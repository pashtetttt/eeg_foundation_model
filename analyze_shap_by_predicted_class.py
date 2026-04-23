"""
SHAP по предсказанным классам: для каждого субъекта берётся вклад признаков
в ответ модели для того класса, который модель предсказала (argmax).

Для всех субъектов с y_pred == «дошкольники» усредняются |SHAP_k| по признакам
(и при необходимости знаковые SHAP для класса k). Так для каждого из 4 классов.

Визуализации: heatmap «признак × предсказанный класс», faceted bar charts, CSV.

Требуется: pip install shap

Примеры:
  .venv/bin/python analyze_shap_by_predicted_class.py --data-dir data --eyes open --features all --model xgb
  .venv/bin/python analyze_shap_by_predicted_class.py --eyes closed --features selected --top-features 30
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eeg_experiment_shared import (
    DATA_DIR,
    DEFAULT_SELECTED_FEATURES_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    load_and_prepare_matrix,
    resolve_feature_indices,
)
from eeg_features import feature_description, get_feature_names


def _feature_names_for_matrix(eyes: str, feature_mode: str, selected_path: Path | None) -> list[str]:
    idx, _ = resolve_feature_indices(eyes, feature_mode, selected_path)
    all_names = get_feature_names(eyes_condition=eyes)
    return [all_names[i] for i in idx]


def _fit_pipeline(model: str, X: np.ndarray, y: np.ndarray, use_balanced_weights: bool):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight

    if model == "xgb":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=3,
            reg_lambda=2.0,
            reg_alpha=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    elif model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            min_samples_split=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError("model must be 'xgb' or 'rf'")

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if use_balanced_weights and model == "xgb":
        sw = compute_sample_weight("balanced", y)
        pipe.fit(X, y, clf__sample_weight=sw)
    else:
        pipe.fit(X, y)
    return pipe


def _as_multiclass_shap_list(sv) -> list[np.ndarray]:
    """Привести вывод TreeExplainer к списку массивов (n_samples, n_features) по классам."""
    if isinstance(sv, list):
        return [np.asarray(a, dtype=float) for a in sv]
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 2:
        return [sv]
    if sv.ndim == 3:
        # (samples, features, classes)
        return [sv[:, :, k] for k in range(sv.shape[2])]
    raise ValueError(f"Unexpected shap_values shape: {getattr(sv, 'shape', None)}")


def _compute_shap_values(pipe, X: np.ndarray, bg_max: int):
    import shap

    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]
    Xs = scaler.transform(X)
    n = Xs.shape[0]
    bg_n = min(bg_max, n)
    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(n, size=bg_n, replace=False)
    bg = Xs[bg_idx]

    explainer = shap.TreeExplainer(clf, data=bg, feature_perturbation="interventional")
    sv = explainer.shap_values(Xs)
    return _as_multiclass_shap_list(sv), Xs


def _mean_shap_per_predicted_class(
    shap_list: list[np.ndarray],
    y_pred: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mean_abs[f,c] = mean(|SHAP_c|) по субъектам с y_pred==c;
    mean_signed[f,c] = mean(SHAP_c) — знак показывает направление вклада в класс c.
    """
    n_feat = shap_list[0].shape[1]
    mean_abs = np.zeros((n_feat, n_classes), dtype=float)
    mean_signed = np.zeros((n_feat, n_classes), dtype=float)
    counts = np.zeros(n_classes, dtype=int)

    for c in range(n_classes):
        mask = y_pred == c
        counts[c] = int(mask.sum())
        if counts[c] == 0:
            continue
        block = shap_list[c][mask]
        mean_abs[:, c] = np.mean(np.abs(block), axis=0)
        mean_signed[:, c] = np.mean(block, axis=0)

    return mean_abs, mean_signed, counts


def plot_heatmap_top_features(
    mean_abs: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    counts: np.ndarray,
    top_features: int,
    out_path: Path,
    title: str,
) -> None:
    """Heatmap: строки = топ признаков по max по столбцам, столбцы = предсказанные классы."""
    n_feat, n_cls = mean_abs.shape
    # топ признаков по сумме по столбцам (важны в нескольких группах)
    score = np.nanmax(mean_abs, axis=1)
    order = np.argsort(score)[::-1][: min(top_features, n_feat)]
    M = mean_abs[order, :]
    labels = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n_cls + 4), max(7, 0.22 * len(order))))
    im = ax.imshow(M.T, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n_cls))
    col_labels = [f"{class_names[c]}\n(n={counts[c]})" for c in range(n_cls)]
    ax.set_yticklabels(col_labels, fontsize=9)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Predicted class (model output)")
    ax.set_title(title + "\nCell = mean |SHAP for that class | subjects predicted as this class", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03, label="mean |SHAP|")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_faceted_bars(
    mean_abs: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    counts: np.ndarray,
    top_k: int,
    out_path: Path,
) -> None:
    """2×2 панели: для каждого предсказанного класса — top-K признаков по mean |SHAP|."""
    n_cls = len(class_names)
    ncols = 2
    nrows = (n_cls + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
    axes = np.atleast_2d(axes)
    for c in range(n_cls):
        r, col = c // ncols, c % ncols
        ax = axes[r, col]
        if counts[c] == 0:
            ax.set_visible(False)
            continue
        vals = mean_abs[:, c]
        order = np.argsort(vals)[::-1][: min(top_k, len(vals))]
        order = order[::-1]
        ax.barh(range(len(order)), vals[order], color="#2c3e50")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feature_names[i] for i in order], fontsize=7)
        ax.set_xlabel("mean |SHAP|")
        ax.set_title(f"Predicted: {class_names[c]}  (n={counts[c]})", fontsize=10)
    # hide empty
    for c in range(n_cls, nrows * ncols):
        r, col = c // ncols, c % ncols
        axes[r, col].set_visible(False)
    fig.suptitle("Top features driving predictions per predicted class", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_csv(
    mean_abs: np.ndarray,
    mean_signed: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    counts: np.ndarray,
    out_path: Path,
    out_signed_path: Path,
) -> None:
    cn = [c.replace(" ", "_") for c in class_names]
    lines = ["feature," + ",".join(f"mean_abs_SHAP_{cn[c]}" for c in range(len(class_names)))]
    lines.append("n_predicted_as," + ",".join(str(int(counts[c])) for c in range(len(class_names))))
    for f in range(len(feature_names)):
        row = [feature_names[f]] + [f"{mean_abs[f, c]:.8f}" for c in range(len(class_names))]
        lines.append(",".join(row))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    lines_s = ["feature," + ",".join(f"mean_signed_SHAP_{cn[c]}" for c in range(len(class_names)))]
    lines_s.append("n_predicted_as," + ",".join(str(int(counts[c])) for c in range(len(class_names))))
    for f in range(len(feature_names)):
        row = [feature_names[f]] + [f"{mean_signed[f, c]:.8f}" for c in range(len(class_names))]
        lines_s.append(",".join(row))
    out_signed_path.write_text("\n".join(lines_s), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SHAP analysis split by predicted class (which features matter when model says class k)."
    )
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "visualizations")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--eyes", type=str, default="open", choices=["closed", "open"])
    ap.add_argument("--features", type=str, default="all", choices=["all", "alpha", "non_alpha", "selected"])
    ap.add_argument("--selected-path", type=Path, default=DEFAULT_SELECTED_FEATURES_PATH)
    ap.add_argument("--model", type=str, default="xgb", choices=["xgb", "rf"])
    ap.add_argument(
        "--balanced-xgb",
        action="store_true",
        help="Use balanced sample_weight when fitting XGB (as in train_xgboost_experiments)",
    )
    ap.add_argument("--shap-background", type=int, default=200, help="Background samples for TreeExplainer")
    ap.add_argument("--top-features", type=int, default=35, help="Rows in heatmap")
    ap.add_argument("--top-bars", type=int, default=18, help="Features per class in faceted plot")
    args = ap.parse_args()

    selected_path = args.selected_path if args.features == "selected" else None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{args.model}_{args.eyes}_{args.features}"

    X, y, class_names, notes = load_and_prepare_matrix(
        eyes_condition=args.eyes,
        feature_mode=args.features,
        max_per_group=args.max,
        selected_path=selected_path,
        data_dir=args.data_dir,
    )
    feat_names = _feature_names_for_matrix(args.eyes, args.features, selected_path)
    if len(feat_names) != X.shape[1]:
        raise RuntimeError(f"Feature names ({len(feat_names)}) != X shape {X.shape[1]}")

    print(f"Data: {args.data_dir.resolve()}")
    print(f"Model: {args.model} | eyes={args.eyes} | features={args.features}")
    for n in notes:
        print("  Feature selection:", n)
    print(feature_description(eyes_condition=args.eyes))
    print(f"Samples: {X.shape[0]}, features: {X.shape[1]}")

    pipe = _fit_pipeline(args.model, X, y, use_balanced_weights=args.balanced_xgb)
    y_pred = pipe.predict(X)
    n_classes = len(class_names)

    print("Computing SHAP (TreeExplainer)...")
    shap_list, _Xs = _compute_shap_values(pipe, X, bg_max=args.shap_background)
    if len(shap_list) != n_classes:
        raise RuntimeError(f"Expected {n_classes} SHAP matrices, got {len(shap_list)}")

    mean_abs, mean_signed, counts = _mean_shap_per_predicted_class(shap_list, y_pred, n_classes)

    print("Predicted class counts:")
    for c, name in enumerate(class_names):
        print(f"  {name}: {counts[c]}")

    base = args.results_dir / f"shap_by_predicted_class_{tag}_{ts}"
    args.results_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmap_top_features(
        mean_abs,
        feat_names,
        class_names,
        counts,
        args.top_features,
        Path(str(base) + "_heatmap.png"),
        title=f"SHAP (mean |·|) by predicted class | {tag}",
    )
    plot_faceted_bars(
        mean_abs,
        feat_names,
        class_names,
        counts,
        args.top_bars,
        Path(str(base) + "_facets.png"),
    )
    write_csv(
        mean_abs,
        mean_signed,
        feat_names,
        class_names,
        counts,
        Path(str(base) + "_mean_abs.csv"),
        Path(str(base) + "_mean_signed.csv"),
    )

    print(f"Wrote: {base}_heatmap.png")
    print(f"Wrote: {base}_facets.png")
    print(f"Wrote: {base}_mean_abs.csv")
    print(f"Wrote: {base}_mean_signed.csv")


if __name__ == "__main__":
    main()
