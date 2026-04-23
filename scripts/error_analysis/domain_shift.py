from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .feature_clusters import aggregate_cluster_matrix, build_clusters
from .stats_utils import ks_two_sample
from .viz import save_kde_overlay

try:
    import seaborn as sns

    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


def analyze_domain_shift(
    *,
    tag: str,
    X_train: np.ndarray,
    X_patients: np.ndarray,
    feature_names: list[str],
    canonical_ch_names_healthy: list[str] | None,
    out_dir: Path,
    ks_p_thr: float = 0.01,
    dpi: int = 300,
) -> list[dict]:
    """
    Step 6: Kolmogorov–Smirnov between healthy (train) and patients per cluster.
    Significant shift if p_ks < ks_p_thr (default 0.01).
    Table: Cluster | KS_stat | p-value | Mean_Healthy | Mean_Patients | Shift_Direction
    Mean difference: mean_patients - mean_healthy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    clusters = build_clusters(feature_names, canonical_ch_names_healthy)
    Xh, keys, sizes = aggregate_cluster_matrix(X_train, clusters, agg="median")
    Xk, _, _ = aggregate_cluster_matrix(X_patients, clusters, agg="median")

    rows: list[dict] = []
    for j, k in enumerate(keys):
        ah = Xh[:, j]
        ak = Xk[:, j]
        ah = ah[np.isfinite(ah)]
        ak = ak[np.isfinite(ak)]
        ks_stat, p_ks = ks_two_sample(ah, ak)
        mh = float(np.mean(ah)) if ah.size else float("nan")
        mk = float(np.mean(ak)) if ak.size else float("nan")
        diff = mk - mh
        if diff > 1e-12:
            direction = "patients>healthy"
        elif diff < -1e-12:
            direction = "patients<healthy"
        else:
            direction = "equal"

        sig = p_ks is not None and p_ks < ks_p_thr
        rows.append(
            {
                "cluster": k,
                "n_features": int(sizes[j]),
                "ks_statistic": ks_stat,
                "p_value": p_ks,
                "mean_healthy": mh,
                "mean_patients": mk,
                "mean_diff_patients_minus_healthy": diff,
                "shift_direction": direction,
                "significant_ks_p_lt_0.01": bool(sig),
            }
        )

    rows.sort(key=lambda r: (-r["significant_ks_p_lt_0.01"], -(r["ks_statistic"] or 0)))

    (out_dir / f"{tag}_domain_shift_ks_table.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # CSV for thesis
    lines = ["cluster,ks_stat,p_value,mean_healthy,mean_patients,shift_direction,significant"]
    for r in rows:
        lines.append(
            f"{r['cluster']},{r['ks_statistic']},{r['p_value']},{r['mean_healthy']},{r['mean_patients']},"
            f"{r['shift_direction']},{r['significant_ks_p_lt_0.01']}"
        )
    (out_dir / f"{tag}_domain_shift_ks_table.csv").write_text("\n".join(lines), encoding="utf-8")

    # Top-10 by |mean_patients - mean_healthy| among finite
    scored = [(r["cluster"], abs(r["mean_diff_patients_minus_healthy"]), r) for r in rows]
    scored.sort(key=lambda t: -t[1])
    top10 = [t[2] for t in scored[:10]]

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r["cluster"][:40] + ("…" if len(r["cluster"]) > 40 else "") for r in top10]
    vals = [r["mean_diff_patients_minus_healthy"] for r in top10]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals]
    y_pos = np.arange(len(top10))
    ax.barh(y_pos, vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("mean_patients − mean_healthy (cluster median)")
    ax.set_title(f"{tag}: top-10 clusters by |shift|")
    fig.tight_layout()
    fig.savefig(plots / f"{tag}_domain_shift_top10_shift_hist.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # KDE for top significant KS clusters (up to 8)
    for r in [x for x in rows if x["significant_ks_p_lt_0.01"]][:8]:
        j = keys.index(r["cluster"])
        a = Xh[:, j][np.isfinite(Xh[:, j])]
        b = Xk[:, j][np.isfinite(Xk[:, j])]
        if a.size < 5 or b.size < 5:
            continue
        save_kde_overlay(
            a=a,
            b=b,
            label_a="Healthy (train)",
            label_b="Patients",
            title=f"{tag}\n{r['cluster']} KS p={r['p_value']:.2e} D={r['ks_statistic']:.3f}",
            xlabel="Cluster aggregated value (median)",
            out_path=plots / f"{tag}_domain_shift_kde_{r['cluster']}.png",
            dpi=dpi,
        )

    return rows
