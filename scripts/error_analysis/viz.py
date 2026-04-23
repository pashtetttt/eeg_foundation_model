from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


def save_confusion_matrix(
    *,
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str,
    normalize_rows: bool = True,
    dpi: int = 300,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(cm, dtype=float)
    if normalize_rows:
        row_sum = np.maximum(1.0, cm.sum(axis=1, keepdims=True))
        mat = 100.0 * cm / row_sum
        fmt = ".1f"
        cbar_label = "% (row-normalized)"
    else:
        mat = cm
        fmt = "g"
        cbar_label = "count"

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    if _HAS_SNS:
        sns.heatmap(
            mat,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": cbar_label},
            vmin=0,
        )
    else:
        im = ax.imshow(mat, cmap="Blues", vmin=0)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right")
        ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_confusion_pair_side_by_side(
    *,
    cm_left: np.ndarray,
    cm_right: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title_left: str,
    title_right: str,
    normalize_rows: bool = True,
    dpi: int = 300,
) -> None:
    """Step 5: two confusion matrices (same format) for comparison."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _norm(cm: np.ndarray) -> tuple[np.ndarray, str, str]:
        cm = np.asarray(cm, dtype=float)
        if normalize_rows:
            row_sum = np.maximum(1.0, cm.sum(axis=1, keepdims=True))
            mat = 100.0 * cm / row_sum
            return mat, ".1f", "% (rows)"
        return cm, "g", "count"

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.0))
    for ax, cm, ttl in zip(axes, (cm_left, cm_right), (title_left, title_right)):
        mat, fmt, cbar_label = _norm(cm)
        if _HAS_SNS:
            sns.heatmap(
                mat,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                cbar_kws={"label": cbar_label},
                vmin=0,
            )
        else:
            im = ax.imshow(mat, cmap="Blues", vmin=0)
            for (i, j), v in np.ndenumerate(mat):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=35, ha="right")
            ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(ttl)
    fig.suptitle("Healthy (OOF) vs Patients (test)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_kde_overlay(
    *,
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
    xlabel: str,
    out_path: Path,
    dpi: int = 300,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    if _HAS_SNS:
        import seaborn as sns

        sns.kdeplot(a, ax=ax, label=label_a, linewidth=2)
        sns.kdeplot(b, ax=ax, label=label_b, linewidth=2)
    else:
        ax.hist(a, bins=30, density=True, alpha=0.4, label=label_a)
        ax.hist(b, bins=30, density=True, alpha=0.4, label=label_b)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

