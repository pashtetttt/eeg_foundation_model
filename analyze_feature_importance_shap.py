"""
Feature importance + SHAP analysis for EEG age-group models.

Works with both datasets:
- data (original folder names with counts)
- data_kids (shorter folder names)

Uses the shared data loader (eeg_experiment_shared.py), so it inherits:
- case-insensitive EDF discovery (.edf/.EDF)
- resilient EDF header loading
- flexible group folder matching for data_kids

Outputs are written into --results-dir (default: results).

Examples:
  ./.venv/bin/python analyze_feature_importance_shap.py --data-dir data_kids --results-dir results_kids --eyes closed --features selected
  ./.venv/bin/python analyze_feature_importance_shap.py --data-dir data --results-dir results --eyes open --features all --model xgb
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from eeg_experiment_shared import (
    DEFAULT_SELECTED_FEATURES_PATH,
    RANDOM_STATE,
    load_and_prepare_matrix,
    resolve_feature_indices,
)
from eeg_features import feature_description, get_feature_names


@dataclass(frozen=True)
class FitResult:
    model: str
    eyes: str
    features: str
    n_samples: int
    n_features: int
    class_names: list[str]
    feature_names: list[str]


def _selected_feature_names(eyes: str, feature_mode: str, selected_path: Path | None) -> list[str]:
    idx, _notes = resolve_feature_indices(eyes, feature_mode, selected_path)
    all_names = get_feature_names(eyes_condition=eyes)
    return [all_names[i] for i in idx]


def _fit_model(model: str, X: np.ndarray, y: np.ndarray):
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

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
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X, y)
        return pipe

    if model == "brf":
        from imblearn.ensemble import BalancedRandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        clf = BalancedRandomForestClassifier(
            n_estimators=250,
            max_depth=25,
            min_samples_leaf=1,
            max_features="sqrt",
            sampling_strategy="auto",
            replacement=False,
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X, y)
        return pipe

    if model == "xgb":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
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
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X, y)
        return pipe

    raise ValueError(f"Unknown model '{model}'. Use: rf, brf, xgb")


def _model_native_importance(pipe, model: str) -> np.ndarray | None:
    clf = pipe.named_steps["clf"]
    if model in ("rf", "brf") and hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_, dtype=float)
    if model == "xgb" and hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_, dtype=float)
    return None


def _write_importance_csv(out_csv: Path, feature_names: list[str], values: np.ndarray) -> None:
    order = np.argsort(values)[::-1]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["rank,feature,importance"]
    for r, i in enumerate(order, 1):
        lines.append(f"{r},{feature_names[i]},{values[i]:.10f}")
    out_csv.write_text("\n".join(lines), encoding="utf-8")


def _plot_barh(out_png: Path, feature_names: list[str], values: np.ndarray, top_k: int, title: str) -> None:
    import matplotlib.pyplot as plt

    order = np.argsort(values)[::-1]
    k = min(top_k, len(order))
    idx = order[:k][::-1]
    names = [feature_names[i] for i in idx]
    vals = values[idx]

    plt.figure(figsize=(10, max(6, 0.22 * k)))
    plt.barh(range(k), vals)
    plt.yticks(range(k), names, fontsize=8)
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()


def _compute_shap(pipe, model: str, X: np.ndarray, max_background: int, max_explain: int):
    """
    Returns (shap_values, X_explain).
    For multiclass, shap_values is a list/array depending on SHAP version.
    """
    import shap

    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps.get("scaler")
    Xs = scaler.transform(X) if scaler is not None else X

    bg = Xs[: min(max_background, Xs.shape[0])]
    Xexp = Xs[: min(max_explain, Xs.shape[0])]

    explainer = shap.TreeExplainer(clf, data=bg, feature_perturbation="interventional")
    shap_values = explainer.shap_values(Xexp)
    return shap_values, Xexp


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute feature importance + SHAP and save into results dir.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--max", type=int, default=None, help="Max samples per class (default: all)")
    ap.add_argument("--eyes", type=str, default="closed", choices=["closed", "open"])
    ap.add_argument("--features", type=str, default="selected", choices=["all", "alpha", "non_alpha", "selected"])
    ap.add_argument("--selected-path", type=Path, default=DEFAULT_SELECTED_FEATURES_PATH)
    ap.add_argument("--model", type=str, default="rf", choices=["rf", "brf", "xgb"])
    ap.add_argument("--top", type=int, default=50, help="Top-K to plot for importance bars")
    ap.add_argument("--shap", action="store_true", help="Also compute SHAP and save summary plot")
    ap.add_argument("--shap-background", type=int, default=200)
    ap.add_argument("--shap-explain", type=int, default=500)
    args = ap.parse_args()

    selected_path = args.selected_path if args.features == "selected" else None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    X, y, class_names, notes = load_and_prepare_matrix(
        eyes_condition=args.eyes,
        feature_mode=args.features,
        max_per_group=args.max,
        selected_path=selected_path,
        data_dir=args.data_dir,
    )
    feat_names = _selected_feature_names(args.eyes, args.features, selected_path)

    print(f"Data: {args.data_dir.resolve()}")
    print(f"Model: {args.model} | eyes={args.eyes} | features={args.features}")
    for n in notes:
        print("  Feature selection:", n)
    print(feature_description(eyes_condition=args.eyes))
    print(f"Samples: {X.shape[0]}, features: {X.shape[1]}")

    pipe = _fit_model(args.model, X, y)

    native = _model_native_importance(pipe, args.model)
    if native is not None and native.shape[0] == len(feat_names):
        out_csv = args.results_dir / f"importance_native_{args.model}_{args.eyes}_{args.features}_{ts}.csv"
        out_png = args.results_dir / f"importance_native_{args.model}_{args.eyes}_{args.features}_{ts}.png"
        _write_importance_csv(out_csv, feat_names, native)
        _plot_barh(
            out_png,
            feat_names,
            native,
            top_k=args.top,
            title=f"Native importance ({args.model}) | eyes={args.eyes} | features={args.features}",
        )
        print(f"Wrote: {out_csv}")
        print(f"Wrote: {out_png}")
    else:
        print("Native importance not available for this model/pipeline.")

    if args.shap:
        try:
            import shap  # noqa: F401
        except Exception as e:
            raise SystemExit(f"SHAP requested but import failed: {e}")

        shap_values, Xexp = _compute_shap(
            pipe,
            args.model,
            X,
            max_background=args.shap_background,
            max_explain=args.shap_explain,
        )

        import matplotlib.pyplot as plt
        import shap

        out_png = args.results_dir / f"shap_summary_{args.model}_{args.eyes}_{args.features}_{ts}.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)

        # Handle common SHAP outputs for multiclass:
        # - list of (n_samples, n_features) arrays per class
        # - or array with class axis
        if isinstance(shap_values, list) and len(shap_values) > 0:
            # Summarize by mean(|shap|) over classes for a single plot
            sv = np.mean([np.abs(s) for s in shap_values], axis=0)
            shap.summary_plot(sv, Xexp, feature_names=feat_names, show=False, plot_type="bar")
        else:
            shap.summary_plot(shap_values, Xexp, feature_names=feat_names, show=False, plot_type="bar")

        plt.tight_layout()
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()

