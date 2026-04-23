from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.error_analysis.data_loading import load_dataset_with_subjects
from scripts.error_analysis.feature_clusters import build_clusters
from .comparison_plots import plot_comparison
from .experiment_runner import ExperimentConfig, FeatureSelectionExperiments


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature Selection Experiments (XGBoost regularized, GroupKFold).")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--eyes", type=str, default="open", choices=["closed", "open"])
    ap.add_argument("--max", type=int, default=None, help="Debug cap per class")
    ap.add_argument("--out-dir", type=Path, default=Path("results/feature_selection"))
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--rep-criterion", type=str, default="variance", choices=["variance", "corr_y"])
    ap.add_argument("--topn", type=int, default=100)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"run_{args.data_dir.name}_{args.eyes}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fs] loading dataset: {args.data_dir} eyes={args.eyes}")
    ds = load_dataset_with_subjects(data_dir=args.data_dir, eyes=args.eyes, max_per_group=args.max)
    X = ds.X
    y = ds.y
    groups = np.asarray(ds.subject_ids)

    clusters_dict = build_clusters(ds.feature_names, ds.canonical_ch_names)

    cfg = ExperimentConfig(
        random_state=args.random_state,
        n_splits=args.splits,
        rep_criterion=args.rep_criterion,
        topn_keep=args.topn,
    )
    runner = FeatureSelectionExperiments(cfg)
    runner.run_all_experiments(X=X, y=y, groups=groups, clusters_dict=clusters_dict, feature_names=ds.feature_names)

    runner.save_results(out_dir)
    runner.save_cluster_membership(output_dir=out_dir, clusters_dict=clusters_dict, original_feature_names=ds.feature_names)

    # per-experiment selected feature dumps
    for key, r in runner.results.items():
        (out_dir / f"selected_features_{r.name}.json").write_text(
            __import__("json").dumps({"experiment": r.name, "n_features": r.n_features, "feature_names": r.feature_names}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    plot_comparison(out_dir / "experiment_summary.csv", out_dir / "comparison_plots.png", dpi=300)
    (out_dir / "stable_features_report.txt").write_text(
        "TODO: stable features/cluster report will be added next.\n",
        encoding="utf-8",
    )

    print(f"[fs] done. outputs in {out_dir}")


if __name__ == "__main__":
    main()

