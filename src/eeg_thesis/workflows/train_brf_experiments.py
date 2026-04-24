from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eeg_features import feature_description
from eeg_experiment_shared import (
    DEFAULT_SELECTED_FEATURES_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    N_SPLITS,
    load_and_prepare_matrix,
)

warnings.filterwarnings("ignore", message=".*does not conform to MNE naming conventions.*")

BRF_CONFIGS: list[dict] = [
    {"name": "brf_baseline", "n_estimators": 200, "max_depth": 20, "min_samples_leaf": 2, "sampling_strategy": "auto", "replacement": False, "bootstrap": True},
    {"name": "brf_deep", "n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "sampling_strategy": "auto", "replacement": False, "bootstrap": True},
    {"name": "brf_shallow_many", "n_estimators": 400, "max_depth": 12, "min_samples_leaf": 4, "sampling_strategy": "auto", "replacement": False, "bootstrap": True},
    {"name": "brf_with_replacement", "n_estimators": 200, "max_depth": 20, "min_samples_leaf": 2, "sampling_strategy": "auto", "replacement": True, "bootstrap": True},
    {"name": "brf_sqrt_features", "n_estimators": 250, "max_depth": 25, "min_samples_leaf": 1, "max_features": "sqrt", "sampling_strategy": "auto", "replacement": False, "bootstrap": True},
]


def build_pipeline(cfg: dict) -> Pipeline:
    cfg = cfg.copy()
    cfg.pop("name", None)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("brf", BalancedRandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **cfg)),
        ]
    )


def evaluate_split(pipeline: Pipeline, X_train, y_train, X_test, y_test, class_names: list[str]) -> dict:
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    y_pred = pipeline.predict(X_test)
    return {
        "train_time_s": train_s,
        "accuracy": float((y_pred == y_test).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "y_pred": y_pred,
    }


def evaluate_cv(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    elapsed = time.perf_counter() - t0
    return {
        "cv_time_s": elapsed,
        "accuracy": float((y_pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "y_pred": y_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Balanced RF experiments (imbalanced-learn).")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root directory (default: data)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Where to save results_*.txt (default: results)")
    parser.add_argument("--eyes", type=str, default="closed", choices=["closed", "open"])
    parser.add_argument("--features", type=str, default="all", choices=["all", "alpha", "non_alpha", "selected"])
    parser.add_argument("--selected-path", type=str, default=str(DEFAULT_SELECTED_FEATURES_PATH))
    args = parser.parse_args()
    selected_path = Path(args.selected_path) if args.features == "selected" else None

    experiment_start = datetime.now()
    log_lines: list[str] = []

    def log(s: str) -> None:
        log_lines.append(s)
        print(s)

    log("=" * 72)
    log("Experiments: BalancedRandomForestClassifier (imblearn)")
    log(f"Started: {experiment_start.isoformat()}")
    log(f"Eyes: {args.eyes} | Features: {args.features}")
    if args.features == "selected":
        log(f"Selected file: {args.selected_path}")
    log("=" * 72)

    print("Loading data...")
    X, y, class_names, sel_notes = load_and_prepare_matrix(args.eyes, args.features, args.max, selected_path, data_dir=args.data_dir)
    for n in sel_notes:
        log(f"  Feature selection: {n}")
    log(feature_description(eyes_condition=args.eyes))
    log(f"Samples: {X.shape[0]}, features: {X.shape[1]}")
    for i, name in enumerate(class_names):
        log(f"  Class {name}: {(y == i).sum()}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    log("")
    log("--- Hold-out (stratified 80/20) ---")
    for cfg_template in BRF_CONFIGS:
        cfg = cfg_template.copy()
        label = cfg["name"]
        pipeline = build_pipeline(cfg)
        metrics = evaluate_split(pipeline, X_train, y_train, X_test, y_test, class_names)
        y_pred = metrics.pop("y_pred")
        log("")
        log(f">> [{label}] {cfg_template}")
        log(f"    acc={metrics['accuracy']:.4f}  bal_acc={metrics['balanced_accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  train_s={metrics['train_time_s']:.2f}")
        log(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        log(str(confusion_matrix(y_test, y_pred)))

    log("")
    log("--- Stratified 5-fold CV ---")
    for cfg_template in BRF_CONFIGS:
        cfg = cfg_template.copy()
        label = cfg["name"]
        pipeline = build_pipeline(cfg)
        m = evaluate_cv(pipeline, X, y)
        y_cv = m.pop("y_pred")
        log("")
        log(f">> [{label}] CV time={m['cv_time_s']:.2f}s")
        log(f"    acc={m['accuracy']:.4f}  bal_acc={m['balanced_accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")
        log(classification_report(y, y_cv, target_names=class_names, zero_division=0))
        log(str(confusion_matrix(y, y_cv)))

    log("")
    log(f"Ended: {datetime.now().isoformat()}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    ts = experiment_start.strftime("%Y%m%d_%H%M%S")
    out = args.results_dir / f"results_brf_experiments_{args.eyes}_{args.features}_{ts}.txt"
    out.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
