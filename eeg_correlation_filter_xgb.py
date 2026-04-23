"""
Pearson correlation between features on `data/` only (single eyes condition),
NOT closed−open difference. Default: `--eyes open` → 364 features (common open/closed layout).

Filter: |r| > threshold → drop feature with lower variance (pairs by descending |r|).
Then XGBoost xgb_regularized + stratified CV + full-fit metrics.

  python eeg_correlation_filter_xgb.py
  python eeg_correlation_filter_xgb.py --eyes closed --features all
  python eeg_correlation_filter_xgb.py --threshold 0.90 --max 200
"""

from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from eeg_experiment_shared import (
    DATA_DIR,
    DEFAULT_SELECTED_FEATURES_PATH,
    N_SPLITS,
    RANDOM_STATE,
    RESULTS_DIR,
    indices_from_name_file,
    load_and_prepare_matrix,
)
from eeg_features import feature_description, get_feature_names, get_feature_indices_by_category
from train_xgboost_experiments import XGB_CONFIGS, build_pipeline, fit_with_balanced_weights

warnings.filterwarnings("ignore", message=".*does not conform to MNE naming conventions.*")

try:
    import seaborn as sns

    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

try:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

VIS_DIR = RESULTS_DIR / "visualizations"

# Порядок групп для матрицы «тип признака × тип признака»
GROUP_ORDER = [
    "band_power",
    "band_ratio",
    "centroid",
    "entropy",
    "envelope_freq",
    "hfd",
    "hjorth",
    "alpha_var",
    "theta_topo",
    "alpha_topo",
    "other",
]


def coarse_group_from_name(name: str) -> str:
    """Грубая категория по префиксу имени (как в eeg_features)."""
    if name.startswith("band_"):
        return "band_power"
    if name.startswith("ratio_"):
        return "band_ratio"
    if name.startswith("centroid_"):
        return "centroid"
    if name.startswith("samp_ent_") or name.startswith("app_ent_"):
        return "entropy"
    if name.startswith("envfreq_"):
        return "envelope_freq"
    if name.startswith("hfd_"):
        return "hfd"
    if name.startswith("hjorth_"):
        return "hjorth"
    if name.startswith("alpha_var_"):
        return "alpha_var"
    if name.startswith("theta_ratio_"):
        return "theta_topo"
    if name.startswith("alpha_power") or name.startswith("pred_"):
        return "alpha_topo"
    return "other"


def inter_group_mean_abs_corr(C: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Матрица G[i,j] = среднее |r| между всеми парами признаков из группы i и группы j
    (на диагонали — только пары с разными индексами внутри группы).
    """
    groups: dict[str, list[int]] = {g: [] for g in GROUP_ORDER}
    for i, name in enumerate(feature_names):
        g = coarse_group_from_name(name)
        groups.setdefault(g, []).append(i)

    labels = [g for g in GROUP_ORDER if groups.get(g)]
    k = len(labels)
    G = np.zeros((k, k), dtype=float)
    for ii, gi in enumerate(labels):
        idx_i = groups[gi]
        for jj, gj in enumerate(labels):
            idx_j = groups[gj]
            if gi == gj:
                vals = [abs(C[a, b]) for a in idx_i for b in idx_j if a < b]
            else:
                vals = [abs(C[a, b]) for a in idx_i for b in idx_j]
            G[ii, jj] = float(np.mean(vals)) if vals else 0.0
    return G, labels


def _short_label(s: str, max_len: int = 44) -> str:
    t = s.replace("_", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def plot_interpretable_correlations(
    C: np.ndarray,
    feature_names: list[str],
    threshold: float,
    out_path: Path,
    top_pairs_n: int = 45,
) -> None:
    """
    Дополнительные фигуры: группы × группы, топ пар по |r|, clustermap с дендрограммами.
    Имена файлов: <stem>_groups.png, <stem>_top_pairs.png, <stem>_clustermap.png рядом с основным PNG.
    """
    parent = out_path.parent
    stem = out_path.stem
    parent.mkdir(parents=True, exist_ok=True)
    n = C.shape[0]

    # 1) Матрица между группами признаков
    G, g_labels = inter_group_mean_abs_corr(C, feature_names)
    if G.size > 0 and _HAS_SNS:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            G,
            xticklabels=g_labels,
            yticklabels=g_labels,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            ax=ax,
            cbar_kws={"label": "mean |r|"},
        )
        ax.set_title(
            "Mean |Pearson r| between feature groups\n"
            "(each cell: average over all cross-pairs between the two groups)",
            fontsize=11,
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        fig.tight_layout()
        fig.savefig(parent / f"{stem}_groups.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    elif G.size > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(G, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(g_labels)))
        ax.set_yticks(range(len(g_labels)))
        ax.set_xticklabels(g_labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(g_labels, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, label="mean |r|")
        ax.set_title("Mean |Pearson r| between feature groups")
        fig.tight_layout()
        fig.savefig(parent / f"{stem}_groups.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 2) Топ пар по |r| (самые сильные связи между отдельными признаками)
    pairs: list[tuple[float, int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = C[i, j]
            if np.isfinite(r):
                pairs.append((abs(r), i, j, float(r)))
    pairs.sort(key=lambda t: -t[0])
    take = pairs[: min(top_pairs_n, len(pairs))]
    if take:
        labels_y = [f"{_short_label(feature_names[i])}  ↔  {_short_label(feature_names[j])}" for _, i, j, _ in take]
        vals = [t[3] for t in take]  # signed r
        y_pos = np.arange(len(take))
        fig, ax = plt.subplots(figsize=(11, max(6.0, 0.22 * len(take))))
        colors = ["#c0392b" if abs(v) > threshold else "#2980b9" for v in vals]
        ax.barh(y_pos, [abs(t[0]) for t in take], color=colors, height=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_y, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("|Pearson r|")
        ax.set_xlim(0, 1.02)
        ax.axvline(threshold, color="gray", linestyle="--", linewidth=1, label=f"filter threshold {threshold}")
        ax.set_title(
            f"Strongest feature–feature correlations (top {len(take)} pairs)\n"
            f"red: |r| > threshold (would be filtered), blue: below threshold",
            fontsize=10,
        )
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(parent / f"{stem}_top_pairs.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 3) Clustermap полной матрицы r (дендрограммы = кластеры похожих признаков)
    if not _HAS_SNS or not _HAS_SCIPY or n < 2:
        return
    try:
        D = 1.0 - np.abs(C)
        np.fill_diagonal(D, 0.0)
        D = np.clip(D, 0.0, None)
        condensed = squareform(D, checks=False)
        condensed = np.nan_to_num(condensed, nan=0.0, posinf=1.0)
        row_linkage = linkage(condensed, method="average")
        col_linkage = row_linkage
        show_labels = n <= 48
        if show_labels:
            lbls = [feature_names[i][:18] + "…" if len(feature_names[i]) > 18 else feature_names[i] for i in range(n)]
        else:
            lbls = False
        fig_w = 12 if n > 80 else min(14, 8 + n * 0.08)
        g = sns.clustermap(
            C,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            row_linkage=row_linkage,
            col_linkage=col_linkage,
            xticklabels=lbls if show_labels else False,
            yticklabels=lbls if show_labels else False,
            figsize=(fig_w, fig_w),
            dendrogram_ratio=0.12,
            cbar_pos=(0.02, 0.4, 0.02, 0.18),
            cbar_kws={"label": "Pearson r"},
        )
        if show_labels:
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=5)
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=5)
        g.fig.suptitle(
            "Feature correlation matrix + hierarchical clustering\n"
            "(rows/cols reordered by similarity of correlation profile)",
            y=1.02,
            fontsize=11,
        )
        g.savefig(parent / f"{stem}_clustermap.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        warnings.warn(f"Clustermap skipped: {e}")


def resolve_feature_name_list(eyes: str, feature_mode: str, selected_path: Path | None) -> list[str]:
    """Имена столбцов X в том же порядке, что и load_and_prepare_matrix."""
    full = get_feature_names(eyes)
    fm = feature_mode.lower()
    if fm == "all":
        return full
    if fm == "alpha":
        idx = get_feature_indices_by_category(eyes_condition=eyes)["alpha"]
        return [full[i] for i in idx]
    if fm == "non_alpha":
        cats = get_feature_indices_by_category(eyes_condition=eyes)
        alpha_set = set(cats["alpha"])
        return [full[i] for i in range(len(full)) if i not in alpha_set]
    if fm == "selected":
        path = selected_path or DEFAULT_SELECTED_FEATURES_PATH
        idx, _ = indices_from_name_file(path, eyes)
        return [full[i] for i in idx]
    raise ValueError(f"Unknown feature_mode {feature_mode!r}")


def pearson_corr_matrix(X: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(X.T)
    C = np.asarray(C, dtype=float)
    np.fill_diagonal(C, 1.0)
    C = np.nan_to_num(C, nan=0.0)
    return C


def feature_variances(X: np.ndarray) -> np.ndarray:
    return np.var(X, axis=0, ddof=0)


def drop_redundant_high_correlation(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float,
) -> tuple[np.ndarray, list[str], list[str], list[tuple[str, str, float, str]]]:
    n = X.shape[1]
    var = feature_variances(X)
    C = pearson_corr_matrix(X)

    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = C[i, j]
            if np.isfinite(r) and abs(r) > threshold:
                pairs.append((abs(r), i, j))
    pairs.sort(key=lambda t: -t[0])

    keep = set(range(n))
    dropped_names: list[str] = []
    log: list[tuple[str, str, float, str]] = []

    for abr, i, j in pairs:
        if i not in keep or j not in keep:
            continue
        vi, vj = var[i], var[j]
        if vi >= vj:
            drop_idx, keep_idx = j, i
        else:
            drop_idx, keep_idx = i, j
        keep.discard(drop_idx)
        dropped_names.append(feature_names[drop_idx])
        log.append(
            (
                feature_names[i],
                feature_names[j],
                float(C[i, j]),
                f"removed={feature_names[drop_idx]} (var={var[drop_idx]:.6g}), kept={feature_names[keep_idx]} (var={var[keep_idx]:.6g})",
            )
        )

    idx_sorted = sorted(keep)
    X_out = X[:, idx_sorted]
    names_kept = [feature_names[i] for i in idx_sorted]
    return X_out, names_kept, dropped_names, log


def _cluster_order(C: np.ndarray) -> np.ndarray:
    if not _HAS_SCIPY or C.shape[0] < 2:
        return np.arange(C.shape[0])
    try:
        D = 1.0 - np.abs(C)
        np.fill_diagonal(D, 0.0)
        D = np.clip(D, 0.0, None)
        condensed = squareform(D, checks=False)
        condensed = np.nan_to_num(condensed, nan=0.0, posinf=1.0)
        Z = linkage(condensed, method="average")
        return np.array(leaves_list(Z))
    except Exception:
        return np.arange(C.shape[0])


def plot_correlation_analysis(
    C_before: np.ndarray,
    C_after: np.ndarray,
    n_before: int,
    n_after: int,
    threshold: float,
    eyes: str,
    features: str,
    out_path: Path,
    feature_names: list[str],
) -> None:
    order_b = _cluster_order(C_before)
    C_b = C_before[order_b][:, order_b]

    if C_after.size > 0:
        order_a = _cluster_order(C_after)
        C_a = C_after[order_a][:, order_a]
    else:
        C_a = np.array([[1.0]])

    fig = plt.figure(figsize=(16, 7), constrained_layout=False)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.15, 0.7], wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    im0 = ax0.imshow(C_b, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", rasterized=True)
    ax0.set_title(f"Pearson r before filter\nn = {n_before}", fontsize=11)
    ax0.set_xlabel("feature (hierarchical order)")
    ax0.set_ylabel("feature (hierarchical order)")
    plt.colorbar(im0, ax=ax0, fraction=0.046, label="r")

    im1 = ax1.imshow(C_a, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", rasterized=True)
    ax1.set_title(f"After removing |r| > {threshold}\nn = {n_after}", fontsize=11)
    ax1.set_xlabel("feature (hierarchical order)")
    ax1.set_ylabel("feature (hierarchical order)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, label="r")

    ax2.axis("off")
    summary = (
        f"data/  eyes={eyes}  features={features}\n"
        f"Threshold: |r| > {threshold}\n"
        f"Features before: {n_before}\n"
        f"After: {n_after}\n"
        f"Removed: {n_before - n_after}\n\n"
        "Rule: keep higher variance\n"
        "per correlated pair\n"
        "(pairs by descending |r|)."
    )
    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes, fontsize=10, verticalalignment="top", family="sans-serif")

    fig.suptitle(f"Pearson correlation filter (single condition, not eyes-diff)", fontsize=13, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    triu = np.triu_indices(C_before.shape[0], k=1)
    vals = np.abs(C_before[triu])
    vals = vals[np.isfinite(vals)]
    fig2, ax = plt.subplots(figsize=(8, 4))
    if _HAS_SNS:
        sns.histplot(vals, bins=40, kde=True, ax=ax, color="steelblue")
    else:
        ax.hist(vals, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(threshold, color="crimson", linestyle="--", label=f"threshold = {threshold}")
    ax.set_xlabel("|Pearson r| (all pairs, before filter)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of |correlation| across feature pairs")
    ax.legend()
    fig2.tight_layout()
    hist_path = out_path.with_name(out_path.stem + "_abs_r_hist.png")
    fig2.savefig(hist_path, dpi=140, bbox_inches="tight")
    plt.close(fig2)

    print(f"Interpretable plots: {out_path.stem}_groups.png, _top_pairs.png, _clustermap.png")
    plot_interpretable_correlations(C_before, feature_names, threshold, out_path)


def _effective_cv_splits(y: np.ndarray, n_classes: int) -> int:
    counts = [int(np.sum(y == i)) for i in range(n_classes)]
    m = min(counts) if counts else 0
    return min(N_SPLITS, m) if m >= 2 else 0


def cross_val_predict_xgb(X: np.ndarray, y: np.ndarray, n_splits: int, pipe_template: Pipeline) -> tuple[np.ndarray, float]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.empty_like(y)
    t0 = time.perf_counter()
    for train_idx, test_idx in cv.split(X, y):
        p = clone(pipe_template)
        fit_with_balanced_weights(p, X[train_idx], y[train_idx])
        y_pred[test_idx] = p.predict(X[test_idx])
    elapsed = time.perf_counter() - t0
    return y_pred, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation filter on data/ (single eyes) + XGBoost.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max", type=int, default=None, help="Max EDF files per class")
    parser.add_argument("--eyes", type=str, default="open", choices=["closed", "open"])
    parser.add_argument(
        "--features",
        type=str,
        default="all",
        choices=["all", "alpha", "non_alpha", "selected"],
    )
    parser.add_argument("--selected-path", type=str, default=str(DEFAULT_SELECTED_FEATURES_PATH))
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    selected_path = Path(args.selected_path) if args.features == "selected" else None

    t0 = datetime.now()
    ts = t0.strftime("%Y%m%d_%H%M%S")

    print("Loading features from data/ (single eyes condition, not closed−open difference)...")
    X, y, class_names, sel_notes = load_and_prepare_matrix(
        args.eyes,
        args.features,
        args.max,
        selected_path,
        data_dir=args.data_dir,
    )
    for n in sel_notes:
        print(f"  Feature selection: {n}")
    print(feature_description(eyes_condition=args.eyes))

    feature_names = resolve_feature_name_list(args.eyes, args.features, selected_path)
    if len(feature_names) != X.shape[1]:
        raise RuntimeError(f"Feature name count {len(feature_names)} != X columns {X.shape[1]}")

    n_before = X.shape[1]
    print(f"Samples: {X.shape[0]}, features: {n_before}")

    C_before = pearson_corr_matrix(X)

    print(f"Filtering pairs with |r| > {args.threshold} (keep higher variance)...")
    X_f, names_kept, names_dropped, drop_log = drop_redundant_high_correlation(
        X, feature_names, args.threshold
    )
    n_after = X_f.shape[1]
    print(f"Kept {n_after} features; dropped {len(names_dropped)}.")

    C_after = pearson_corr_matrix(X_f) if n_after > 0 else np.array([[1.0]])

    tag = f"{args.eyes}_{args.features}"
    vis_path = VIS_DIR / f"corr_filter_data_{tag}_{ts}.png"
    print(f"Saving figure: {vis_path}")
    plot_correlation_analysis(
        C_before,
        C_after,
        n_before,
        n_after,
        args.threshold,
        args.eyes,
        args.features,
        vis_path,
        feature_names,
    )

    log_lines: list[str] = []
    log_lines.append("data/: Pearson correlation filter + XGBoost (xgb_regularized); NOT eyes-difference")
    log_lines.append(f"eyes={args.eyes} features={args.features}")
    log_lines.append(f"Started: {t0.isoformat()}")
    log_lines.append(f"Threshold |r| > {args.threshold}")
    log_lines.append(f"Features: {n_before} -> {n_after} (removed {len(names_dropped)})")
    log_lines.append("")
    log_lines.append("--- Dropped redundant pairs ---")
    for row in drop_log[:500]:
        log_lines.append(f"{row[0]}  |  {row[1]}  r={row[2]:.4f}  {row[3]}")
    if len(drop_log) > 500:
        log_lines.append(f"... ({len(drop_log) - 500} more lines omitted)")

    cfg = next(c for c in XGB_CONFIGS if c["name"] == "xgb_regularized")
    pipe_template = build_pipeline(cfg.copy())

    log_lines.append("")
    log_lines.append(f"--- XGBoost {cfg['name']} ---")

    n_cv = _effective_cv_splits(y, len(class_names))
    if n_cv >= 2:
        log_lines.append(f"Stratified {n_cv}-fold CV (OOF), balanced sample_weight")
        y_oof, cv_t = cross_val_predict_xgb(X_f, y, n_cv, pipe_template)
        log_lines.append(f"CV time: {cv_t:.2f}s")
        oof_line = (
            f"OOF acc={(y_oof == y).mean():.4f}  bal_acc={balanced_accuracy_score(y, y_oof):.4f}  "
            f"macro_f1={f1_score(y, y_oof, average='macro', zero_division=0):.4f}  "
            f"weighted_f1={f1_score(y, y_oof, average='weighted', zero_division=0):.4f}"
        )
        log_lines.append(oof_line)
        print(oof_line)
        log_lines.append(classification_report(y, y_oof, target_names=class_names, zero_division=0))
        log_lines.append(str(confusion_matrix(y, y_oof)))
    else:
        log_lines.append(f"Skip CV (effective_splits={n_cv}).")

    log_lines.append("")
    log_lines.append("--- Fit on ALL (in-sample) ---")
    pipe_full = build_pipeline(cfg.copy())
    fit_with_balanced_weights(pipe_full, X_f, y)
    y_hat = pipe_full.predict(X_f)
    log_lines.append(
        f"acc={(y_hat == y).mean():.4f}  bal_acc={balanced_accuracy_score(y, y_hat):.4f}  "
        f"macro_f1={f1_score(y, y_hat, average='macro', zero_division=0):.4f}"
    )
    log_lines.append(classification_report(y, y_hat, target_names=class_names, zero_division=0))
    log_lines.append(str(confusion_matrix(y, y_hat)))
    log_lines.append(f"Ended: {datetime.now().isoformat()}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_txt = args.results_dir / f"results_corr_filter_xgb_{tag}_{ts}.txt"
    out_txt.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
