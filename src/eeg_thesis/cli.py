"""Unified CLI entrypoints for main workflows."""

from __future__ import annotations

import argparse

from .legacy import run_legacy
from .workflows.train_xgboost_experiments import main as train_xgb_main


def run_train() -> None:
    train_xgb_main()


def run_feature_selection() -> None:
    run_legacy("scripts/feature_selection/run_experiments.py")


def run_error_analysis() -> None:
    run_legacy("scripts/error_analysis/run_error_analysis.py")


def run_dfa() -> None:
    run_legacy("scripts/dfa_analysis/run_pipeline.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG thesis workflow launcher.")
    parser.add_argument(
        "workflow",
        choices=["train", "feature-selection", "error-analysis", "dfa"],
        help="Workflow to run.",
    )
    args = parser.parse_args()

    if args.workflow == "train":
        run_train()
    elif args.workflow == "feature-selection":
        run_feature_selection()
    elif args.workflow == "error-analysis":
        run_error_analysis()
    elif args.workflow == "dfa":
        run_dfa()


if __name__ == "__main__":
    main()
