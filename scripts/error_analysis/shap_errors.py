from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shap

from .feature_clusters import aggregate_cluster_matrix, build_clusters


@dataclass(frozen=True)
class ShapSummary:
    # mean absolute SHAP per cluster (aligned to cluster_keys)
    cluster_keys: list[str]
    mean_abs_shap: np.ndarray


def shap_on_error_subset(
    *,
    fitted_pipeline,
    X_raw: np.ndarray,
    feature_names: list[str],
    canonical_ch_names: list[str] | None,
    error_indices: np.ndarray,
    out_dir: Path,
    tag: str,
    max_samples: int = 400,
    dpi: int = 300,
) -> ShapSummary:
    """
    Compute SHAP values for XGBoost model in the provided fitted pipeline, only on error_indices.

    No leakage responsibility is on the caller: fitted_pipeline must be trained without these samples
    if this is a validation analysis (OOF).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    error_indices = np.asarray(error_indices, dtype=int)
    if error_indices.size == 0:
        return ShapSummary(cluster_keys=[], mean_abs_shap=np.array([], dtype=float))

    take = error_indices[: max_samples]
    X_err = X_raw[take]

    # Extract model + transformed X as the pipeline sees it.
    # We use the full pipeline to transform X to match feature filtering/scaling.
    # For clustering we will still map back by raw feature names only if no corr filter is used;
    # for corr-filter case, we cluster remaining raw features by kept indices (handled outside).
    X_trans = fitted_pipeline[:-1].transform(X_err) if hasattr(fitted_pipeline, "__getitem__") else X_err
    model = fitted_pipeline.named_steps["clf"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # multiclass: list[n_classes] of (n_samples, n_features_trans)
    # We summarize overall magnitude across predicted class contributions by taking mean over classes.
    if isinstance(shap_values, list):
        abs_vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        abs_vals = np.abs(shap_values)

    # Feature-level mean|shap|
    mean_abs = np.mean(abs_vals, axis=0)

    # If transformed features count differs from raw, we can't reliably map to raw names here.
    # The caller should provide feature_names already aligned to the transformed space if needed.
    if mean_abs.shape[0] != len(feature_names):
        # best-effort: truncate to min length
        n = min(mean_abs.shape[0], len(feature_names))
        mean_abs = mean_abs[:n]
        feature_names = feature_names[:n]

    # cluster on feature_names (assumed aligned)
    clusters = build_clusters(feature_names, canonical_ch_names)
    # build pseudo X from mean_abs to aggregate per cluster
    pseudo = mean_abs.reshape(1, -1)
    Xc, keys, _sizes = aggregate_cluster_matrix(pseudo, clusters, agg="mean")
    mean_abs_cluster = Xc[0]

    # Save summary CSV-like text (simple, no pandas dependency)
    order = np.argsort(-mean_abs_cluster)
    lines = ["cluster,mean_abs_shap"]
    for idx in order[: min(200, order.size)]:
        lines.append(f"{keys[idx]},{mean_abs_cluster[idx]:.8g}")
    (out_dir / f"{tag}_shap_mean_abs_by_cluster.csv").write_text("\n".join(lines), encoding="utf-8")

    # Save SHAP summary plots (feature-level)
    try:
        import matplotlib.pyplot as plt

        # bar summary (mean |shap|)
        fig = plt.figure(figsize=(10, 7))
        shap.summary_plot(abs_vals, features=X_trans, feature_names=feature_names, plot_type="bar", show=False, max_display=40)
        plt.title(f"{tag}: SHAP mean(|value|) on errors (bar)")
        plt.tight_layout()
        fig.savefig(out_dir / f"{tag}_shap_summary_bar.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    return ShapSummary(cluster_keys=keys, mean_abs_shap=mean_abs_cluster)

