"""Shared CSV / JSON / matplotlib artifacts for EEGPT fine-tuning scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def plot_training_curves(metrics_csv: Path, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows: list[dict[str, str]] = []
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    def fcol(r: dict[str, str], k: str, default: float = float("nan")) -> float:
        v = r.get(k)
        if v is None or str(v).strip() == "":
            return default
        return float(v)

    epochs = [int(float(r["epoch"])) for r in rows]
    trn_loss = [fcol(r, "train_loss") for r in rows]
    val_loss = [fcol(r, "val_loss") for r in rows]
    val_bal = [fcol(r, "val_balanced_accuracy") for r in rows]
    val_f1 = [fcol(r, "val_macro_f1") for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axes[0].plot(epochs, trn_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_bal, color="C2", label="val_balanced_accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Val balanced accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    if np.any(np.isfinite(np.asarray(val_f1, dtype=float))):
        axes[2].plot(epochs, val_f1, color="C3", label="val_macro_f1")
        axes[2].set_xlabel("epoch")
        axes[2].set_title("Val macro F1")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "val_macro_f1 n/a", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_axis_off()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: list[list[int]],
    labels: list[str],
    out_png: Path,
    *,
    title: str = "Confusion matrix (test)",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(max(5.0, len(labels) * 1.2), max(4.0, len(labels))))
    im = ax.imshow(arr, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(arr.shape[1]),
        yticks=np.arange(arr.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = arr.max() / 2.0 if arr.size else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j,
                i,
                format(int(arr[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if arr[i, j] > thresh else "black",
                fontsize=10,
            )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_readme_plot(out_dir: Path, metrics_name: str = "metrics_per_epoch.csv") -> None:
    p = out_dir / "README_plot.txt"
    p.write_text(
        "Plots generated automatically:\n"
        f"  - training_curves.png  (from {metrics_name})\n"
        "  - confusion_matrix_test.png\n\n"
        "Re-plot curves after editing CSV:\n"
        f"  python scripts/plot_eegpt_curves.py --metrics-csv {out_dir / metrics_name}\n",
        encoding="utf-8",
    )
