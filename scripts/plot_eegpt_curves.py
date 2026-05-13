#!/usr/bin/env python3
"""Plot EEGPT learning curves from metrics_per_epoch.csv (same layout as training scripts write)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eegpt_run_artifacts import plot_training_curves


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot EEGPT train/val curves from metrics CSV.")
    ap.add_argument("--metrics-csv", type=Path, required=True)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: same dir as CSV, training_curves.png)",
    )
    args = ap.parse_args()
    out = args.output
    if out is None:
        out = args.metrics_csv.parent / "training_curves.png"
    plot_training_curves(args.metrics_csv.resolve(), out.resolve())
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
