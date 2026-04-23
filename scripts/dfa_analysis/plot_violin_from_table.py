from __future__ import annotations

import argparse
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
GROUP_ORDER = ["healthy", "patients"]


def plot_violin_subject_level(
    dfa_table_csv: Path,
    out_path: Path,
    dpi: int = 300,
    add_points: bool = True,
) -> None:
    if not _HAS_SNS:
        raise RuntimeError("seaborn is required for violin plot (pip install seaborn).")

    import pandas as pd

    if not dfa_table_csv.is_file():
        raise FileNotFoundError(f"dfa_table.csv not found: {dfa_table_csv}")

    df = pd.read_csv(dfa_table_csv)
    if df.empty:
        raise ValueError(f"Input table is empty: {dfa_table_csv}")

    required = {"subject_id", "group", "age_class", "region", "dfa_alpha"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {dfa_table_csv}: {sorted(missing)}")

    df = df[np.isfinite(df["dfa_alpha"].astype(float))].copy()
    if df.empty:
        raise ValueError("No finite dfa_alpha values in input table.")

    df["age_class"] = pd.Categorical(df["age_class"], categories=AGE_ORDER, ordered=True)
    df["region"] = pd.Categorical(df["region"], categories=REGION_ORDER, ordered=True)
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    df = df.sort_values(["region", "age_class", "group", "subject_id"]).reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), sharey=True)
    axes = axes.flatten()

    for ax, reg in zip(axes, REGION_ORDER):
        sub = df[df["region"] == reg]
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.violinplot(
            data=sub,
            x="age_class",
            y="dfa_alpha",
            hue="group",
            order=AGE_ORDER,
            hue_order=GROUP_ORDER,
            cut=0,
            inner="quartile",
            linewidth=1.0,
            ax=ax,
        )
        if add_points:
            sns.stripplot(
                data=sub,
                x="age_class",
                y="dfa_alpha",
                hue="group",
                order=AGE_ORDER,
                hue_order=GROUP_ORDER,
                dodge=True,
                jitter=0.18,
                alpha=0.28,
                size=2.2,
                palette="dark:k",
                ax=ax,
            )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            uniq = {}
            for h, lab in zip(handles, labels):
                if lab not in uniq and lab in GROUP_ORDER:
                    uniq[lab] = h
            ax.legend(
                [uniq[g] for g in GROUP_ORDER if g in uniq],
                [g for g in GROUP_ORDER if g in uniq],
                loc="best",
                fontsize=8,
                title="group",
                title_fontsize=8,
            )

        ax.set_title(reg)
        ax.set_xlabel("Age class")
        ax.set_ylabel("DFA alpha")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("DFA violin plot by age and health status (subject-level)", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot subject-level DFA violin chart from existing dfa_table.csv (no DFA recomputation)."
    )
    ap.add_argument("--dfa-table", type=Path, required=True, help="Path to dfa_table.csv from DFA run")
    ap.add_argument("--out", type=Path, default=None, help="Output image path (.png). Default: next to table")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--no-points",
        action="store_true",
        help="Disable subject-level jittered points overlay.",
    )
    args = ap.parse_args()

    out = args.out
    if out is None:
        out = args.dfa_table.parent / "dfa_violin_subject_level.png"

    plot_violin_subject_level(
        dfa_table_csv=args.dfa_table,
        out_path=out,
        dpi=args.dpi,
        add_points=not args.no_points,
    )
    print(f"[dfa-violin] saved: {out}")


if __name__ == "__main__":
    main()
