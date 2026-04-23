from __future__ import annotations

"""
Шаг 3: визуализация DFA результатов.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


AGE_ORDER = ["preschooler", "primary", "teenager", "adolescence"]
REGION_ORDER = ["frontal", "central", "parietal", "occipital"]


def _to_arrays(rows: list[dict]) -> tuple[list[str], list[str], list[str], np.ndarray]:
    age = [r["age_class"] for r in rows]
    group = [r["group"] for r in rows]
    region = [r["region"] for r in rows]
    val = np.asarray([float(r["dfa_alpha"]) for r in rows], dtype=float)
    return age, group, region, val


def boxplots_by_age(
    rows: list[dict],
    *,
    group_label: str,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Boxplot DFA по возрастам внутри заданной группы (healthy или patients), фасеты по регионам."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [r for r in rows if r["group"] == group_label and np.isfinite(float(r["dfa_alpha"]))]
    if not data:
        return
    if not _HAS_SNS:
        return
    import pandas as pd

    df = pd.DataFrame(data)
    df["age_class"] = pd.Categorical(df["age_class"], categories=AGE_ORDER, ordered=True)
    df["region"] = pd.Categorical(df["region"], categories=REGION_ORDER, ordered=True)

    g = sns.catplot(
        data=df,
        x="age_class",
        y="dfa_alpha",
        col="region",
        kind="box",
        col_order=REGION_ORDER,
        order=AGE_ORDER,
        height=4.2,
        aspect=0.9,
        showfliers=False,
    )
    for ax in g.axes.flat:
        sns.stripplot(data=df[df["region"] == ax.get_title().split(" = ")[-1]], x="age_class", y="dfa_alpha", order=AGE_ORDER, ax=ax, color="black", alpha=0.35, size=3)
        ax.set_xlabel("Age class")
        ax.set_ylabel("DFA alpha exponent")
        ax.grid(axis="y", alpha=0.2)
    g.fig.suptitle(f"DFA by age ({group_label})", y=1.02)
    g.fig.tight_layout()
    g.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)


def boxplots_health_within_age(
    rows: list[dict],
    *,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Сравнение healthy vs patients внутри каждого возраста (фасеты по region×age)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not _HAS_SNS:
        return
    import pandas as pd

    df = pd.DataFrame([r for r in rows if np.isfinite(float(r["dfa_alpha"]))])
    if df.empty:
        return
    df["age_class"] = pd.Categorical(df["age_class"], categories=AGE_ORDER, ordered=True)
    df["region"] = pd.Categorical(df["region"], categories=REGION_ORDER, ordered=True)

    g = sns.catplot(
        data=df,
        x="group",
        y="dfa_alpha",
        col="age_class",
        row="region",
        kind="box",
        order=["healthy", "patients"],
        col_order=AGE_ORDER,
        row_order=REGION_ORDER,
        height=2.6,
        aspect=1.1,
        showfliers=False,
    )
    for ax in g.axes.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("")
        ax.set_ylabel("DFA alpha")
    g.fig.suptitle("Healthy vs Patients DFA within age groups", y=1.01)
    g.fig.tight_layout()
    g.fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(g.fig)


def heatmap_significance(stats_rows: list[dict], *, out_path: Path, dpi: int = 300) -> None:
    """Тепловая карта -log10(p_adj) для health_within_age по region×age."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not _HAS_SNS:
        return
    import pandas as pd

    filt = [r for r in stats_rows if r.get("comparison_type") == "health_within_age" and r.get("p_adj") is not None]
    if not filt:
        return
    df = pd.DataFrame(filt)
    df["neglog10_p"] = -np.log10(np.maximum(1e-300, df["p_adj"].astype(float)))
    pivot = df.pivot(index="region", columns="age_class", values="neglog10_p").reindex(index=REGION_ORDER, columns=AGE_ORDER)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={"label": "-log10(p_adj)"}, ax=ax)
    ax.set_xlabel("Age class")
    ax.set_ylabel("Region")
    ax.set_title("Health status differences (FDR-adjusted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def trend_plot(rows: list[dict], *, out_path: Path, dpi: int = 300) -> None:
    """Линейный тренд: средний DFA по возрастам для healthy и patients, отдельно по регионам."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not _HAS_SNS:
        return
    import pandas as pd

    df = pd.DataFrame([r for r in rows if np.isfinite(float(r["dfa_alpha"]))])
    if df.empty:
        return
    df["age_class"] = pd.Categorical(df["age_class"], categories=AGE_ORDER, ordered=True)
    df["region"] = pd.Categorical(df["region"], categories=REGION_ORDER, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharey=True)
    axes = axes.flatten()
    for ax, reg in zip(axes, REGION_ORDER):
        sub = df[df["region"] == reg]
        sns.pointplot(data=sub, x="age_class", y="dfa_alpha", hue="group", order=AGE_ORDER, ax=ax, errorbar=("ci", 95))
        ax.set_title(reg)
        ax.set_xlabel("Age class")
        ax.set_ylabel("DFA alpha")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("DFA trend by age and health status", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

