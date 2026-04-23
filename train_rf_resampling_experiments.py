"""
Random Forest resampling experiments for class imbalance.

Focus setup:
- eyes = closed (default)
- features = selected (default)
- model = rf_balanced_subsample (best config from prior RF grid)

Compares:
- no resampling
- random undersampling
- random oversampling
- SMOTE

Run:
  python train_rf_resampling_experiments.py
  python train_rf_resampling_experiments.py --eyes closed --features selected
  python train_rf_resampling_experiments.py --max 120
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler

from eeg_experiment_shared import (
    DEFAULT_SELECTED_FEATURES_PATH,
    N_SPLITS,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
    load_and_prepare_matrix,
)
from eeg_features import feature_description


RF_BEST = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
}


def build_pipeline(strategy: str) -> ImbPipeline:
    steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
    if strategy == "under":
        steps.append(("sampler", RandomUnderSampler(random_state=RANDOM_STATE)))
    elif strategy == "over":
        steps.append(("sampler", RandomOverSampler(random_state=RANDOM_STATE)))
    elif strategy == "smote":
        steps.append(("sampler", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)))
    elif strategy == "none":
        pass
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    steps.append(
        (
            "rf",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **RF_BEST,
            ),
        )
    )
    return ImbPipeline(steps)


def evaluate_holdout(pipeline: ImbPipeline, X_train, y_train, X_test, y_test) -> dict:
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


def evaluate_cv(pipeline: ImbPipeline, X: np.ndarray, y: np.ndarray) -> dict:
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
    parser = argparse.ArgumentParser(description="RF imbalance-resampling experiments (undersample/oversample/SMOTE).")
    parser.add_argument("--max", type=int, default=None, help="Max samples per class (default: all)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root directory (default: data)")
    parser.add_argument("--eyes", type=str, default="closed", choices=["closed", "open"])
    parser.add_argument(
        "--features",
        type=str,
        default="selected",
        choices=["selected", "all", "alpha", "non_alpha"],
        help="Feature mode (default: selected)",
    )
    parser.add_argument(
        "--selected-path",
        type=str,
        default=str(DEFAULT_SELECTED_FEATURES_PATH),
        help="Text file with selected feature names (used when --features selected)",
    )
    args = parser.parse_args()
    selected_path = Path(args.selected_path) if args.features == "selected" else None

    experiment_start = datetime.now()
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        print(msg)

    log("=" * 72)
    log("Experiments: RF best config with resampling strategies")
    log(f"Started: {experiment_start.isoformat()}")
    log(f"Eyes: {args.eyes} | Features: {args.features}")
    if selected_path is not None:
        log(f"Selected file: {selected_path}")
    log(f"RF config: {RF_BEST}")
    log("=" * 72)

    print("Loading data...")
    X, y, class_names, sel_notes = load_and_prepare_matrix(
        eyes_condition=args.eyes,
        feature_mode=args.features,
        max_per_group=args.max,
        selected_path=selected_path,
        data_dir=args.data_dir,
    )
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

    strategies = [
        ("none", "No resampling"),
        ("under", "Random undersampling"),
        ("over", "Random oversampling"),
        ("smote", "SMOTE"),
    ]

    log("")
    log("--- Hold-out (stratified 80/20) ---")
    for key, title in strategies:
        pipe = build_pipeline(key)
        m = evaluate_holdout(pipe, X_train, y_train, X_test, y_test)
        y_pred = m.pop("y_pred")
        log("")
        log(f">> [{key}] {title}")
        log(
            f"    acc={m['accuracy']:.4f}  bal_acc={m['balanced_accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  train_s={m['train_time_s']:.2f}"
        )
        log(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        log("Confusion matrix:")
        log(str(confusion_matrix(y_test, y_pred)))

    log("")
    log("--- Stratified 5-fold CV ---")
    for key, title in strategies:
        pipe = build_pipeline(key)
        m = evaluate_cv(pipe, X, y)
        y_cv = m.pop("y_pred")
        log("")
        log(f">> [{key}] {title} | CV time={m['cv_time_s']:.2f}s")
        log(f"    acc={m['accuracy']:.4f}  bal_acc={m['balanced_accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")
        log(classification_report(y, y_cv, target_names=class_names, zero_division=0))
        log("Confusion matrix (CV predictions):")
        log(str(confusion_matrix(y, y_cv)))

    log("")
    log(f"Ended: {datetime.now().isoformat()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = experiment_start.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"results_rf_resampling_{args.eyes}_{args.features}_{ts}.txt"
    out.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
