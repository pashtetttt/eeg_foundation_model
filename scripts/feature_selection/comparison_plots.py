from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


def _read_summary_csv(path: Path) -> list[dict]:
    import csv

    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def plot_comparison(summary_csv: Path, out_path: Path, dpi: int = 300) -> None:
    """
    - bar chart: accuracy_mean and macro_f1_mean per experiment
    - scatter: n_features vs macro_f1_mean
    """
    rows = _read_summary_csv(summary_csv)
    names = [r["experiment"] for r in rows]
    nfeat = np.array([int(float(r["n_features"])) for r in rows], dtype=int)
    acc = np.array([float(r["accuracy_mean"]) for r in rows], dtype=float)
    f1 = np.array([float(r["macro_f1_mean"]) for r in rows], dtype=float)
    acc_std = np.array([float(r.get("accuracy_std", 0.0) or 0.0) for r in rows], dtype=float)
    f1_std = np.array([float(r.get("macro_f1_std", 0.0) or 0.0) for r in rows], dtype=float)

    fig = plt.figure(figsize=(14, 6.5))
    ax1 = fig.add_subplot(1, 2, 1)
    x = np.arange(len(names))
    w = 0.38
    err_kw = {"elinewidth": 1.2, "capsize": 4, "capthick": 1.2, "alpha": 0.9}
    ax1.bar(
        x - w / 2,
        acc,
        width=w,
        yerr=acc_std,
        error_kw=err_kw,
        label="Accuracy",
    )
    ax1.bar(
        x + w / 2,
        f1,
        width=w,
        yerr=f1_std,
        error_kw=err_kw,
        label="F1-macro",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Metrics by experiment")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.2)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(nfeat, f1, s=70)
    for i, nm in enumerate(names):
        ax2.annotate(nm, (nfeat[i], f1[i]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax2.set_xlabel("N features")
    ax2.set_ylabel("F1-macro (mean)")
    ax2.set_title("N features vs quality")
    ax2.grid(alpha=0.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

