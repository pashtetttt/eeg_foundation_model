#!/usr/bin/env python3
"""Plot HEEGNet learning curves from metrics_per_epoch.csv produced by train_heegnet.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict[str, float | int] = {}
            for k, v in raw.items():
                if k is None or v is None or str(v).strip() == "":
                    continue
                if k == "epoch":
                    row[k] = int(float(v))
                else:
                    row[k] = float(v)
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot HEEGNet trn/val metrics vs epoch.")
    ap.add_argument("--metrics-csv", type=Path, required=True, help="metrics_per_epoch.csv from a training run.")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: same directory as CSV, stem + _curves.png).",
    )
    args = ap.parse_args()

    rows = _read_rows(args.metrics_csv.resolve())
    if not rows:
        raise SystemExit(f"No data rows in {args.metrics_csv}")

    import matplotlib.pyplot as plt

    epochs = [int(r["epoch"]) for r in rows]

    def series(key: str) -> list[float]:
        return [float(r[key]) if key in r else float("nan") for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

    def plot_pair(ax, trn_k: str, val_k: str, title: str) -> None:
        if any(trn_k in r for r in rows):
            ax.plot(epochs, series(trn_k), label=trn_k)
        if any(val_k in r for r in rows):
            ax.plot(epochs, series(val_k), label=val_k)
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plot_pair(axes[0], "trn_loss", "val_loss", "Loss")
    plot_pair(axes[1], "trn_score", "val_score", "Balanced accuracy")
    plot_pair(axes[2], "trn_macro_f1", "val_macro_f1", "Macro F1")

    out = args.output
    if out is None:
        out = args.metrics_csv.with_name(args.metrics_csv.stem + "_curves.png")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
