from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from eeg_experiment_shared import (
    DEFAULT_SELECTED_FEATURES_PATH,
    N_SPLITS,
    RANDOM_STATE,
    TEST_SIZE,
    load_and_prepare_matrix,
)
from eeg_features import feature_description

warnings.filterwarnings("ignore", message=".*does not conform to MNE naming conventions.*")

XGB_BINARY_CONFIGS: list[dict] = [
    {"name": "xgb_bin_baseline", "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0},
    {"name": "xgb_bin_shallow_lr", "n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 1, "reg_lambda": 1.0},
    {"name": "xgb_bin_regularized", "n_estimators": 250, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 3, "reg_lambda": 2.0, "reg_alpha": 0.1},
    {"name": "xgb_bin_fast_lr", "n_estimators": 150, "max_depth": 8, "learning_rate": 0.15, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 1, "reg_lambda": 1.0},
]


def to_adolescence_binary(y_multiclass: np.ndarray, class_names: list[str]) -> tuple[np.ndarray, list[str]]:
    if "adolescence" not in class_names:
        raise ValueError(f"Class 'adolescence' not found in class_names: {class_names}")
    pos_idx = class_names.index("adolescence")
    y_bin = (y_multiclass == pos_idx).astype(int)
    return y_bin, ["rest", "adolescence"]


def build_classifier(cfg: dict) -> XGBClassifier:
    cfg = cfg.copy()
    cfg.pop("name", None)
    return XGBClassifier(objective="binary:logistic", random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss", **cfg)


def build_pipeline(cfg: dict) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", build_classifier(cfg))])


def fit_with_balanced_weights(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    sw = compute_sample_weight("balanced", y)
    pipe.fit(X, y, clf__sample_weight=sw)


def evaluate_split(cfg: dict, X_train, y_train, X_test, y_test) -> dict:
    pipe = build_pipeline(cfg)
    t0 = time.perf_counter()
    fit_with_balanced_weights(pipe, X_train, y_train)
    train_s = time.perf_counter() - t0
    y_pred = pipe.predict(X_test).astype(int)
    return {
        "train_time_s": train_s,
        "accuracy": float((y_pred == y_test).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_adolescence": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "y_pred": y_pred,
    }


def cross_val_predict_balanced(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple[np.ndarray, float]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.empty_like(y)
    t0 = time.perf_counter()
    for train_idx, test_idx in cv.split(X, y):
        pipe = build_pipeline(cfg)
        sw = compute_sample_weight("balanced", y[train_idx])
        pipe.fit(X[train_idx], y[train_idx], clf__sample_weight=sw)
        y_pred[test_idx] = pipe.predict(X[test_idx]).astype(int)
    elapsed = time.perf_counter() - t0
    return y_pred, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost binary experiments: adolescence vs rest.")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root directory (default: data)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Where to save results_*.txt")
    parser.add_argument("--eyes", type=str, default="closed", choices=["closed", "open"])
    parser.add_argument("--features", type=str, default="all", choices=["all", "all_plus_complexity", "alpha", "non_alpha", "selected"], help="all_plus_complexity = legacy features + literature block.")
    parser.add_argument("--selected-path", type=str, default=str(DEFAULT_SELECTED_FEATURES_PATH))
    parser.add_argument("--literature-surrogate-iters", type=int, default=None, help="Phase-shuffle iterations per channel for literature block.")
    args = parser.parse_args()
    selected_path = Path(args.selected_path) if args.features == "selected" else None

    experiment_start = datetime.now()
    log_lines: list[str] = []

    def log(s: str) -> None:
        log_lines.append(s)
        print(s)

    log("=" * 72)
    log("Experiments: XGB binary (adolescence vs rest, balanced sample_weight)")
    log(f"Started: {experiment_start.isoformat()}")
    log(f"Eyes: {args.eyes} | Features: {args.features}")
    if args.features == "selected":
        log(f"Selected file: {args.selected_path}")
    log("=" * 72)

    print("Loading data...")
    X, y_multi, class_names, sel_notes = load_and_prepare_matrix(
        args.eyes,
        args.features,
        args.max,
        selected_path,
        data_dir=args.data_dir,
        literature_surrogate_iters=args.literature_surrogate_iters,
    )
    y, bin_names = to_adolescence_binary(y_multi, class_names)

    for n in sel_notes:
        log(f"  Feature selection: {n}")
    log(feature_description(eyes_condition=args.eyes, include_literature=(args.features == "all_plus_complexity")))
    log(f"Samples: {X.shape[0]}, features: {X.shape[1]}")
    log(f"  Class {bin_names[0]}: {(y == 0).sum()}")
    log(f"  Class {bin_names[1]}: {(y == 1).sum()}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    log("")
    log("--- Hold-out (stratified 80/20) ---")
    for cfg_template in XGB_BINARY_CONFIGS:
        cfg = cfg_template.copy()
        label = cfg["name"]
        metrics = evaluate_split(cfg, X_train, y_train, X_test, y_test)
        y_pred = metrics.pop("y_pred")
        log("")
        log(f">> [{label}] {cfg_template}")
        log("    " f"acc={metrics['accuracy']:.4f}  " f"bal_acc={metrics['balanced_accuracy']:.4f}  " f"macro_f1={metrics['macro_f1']:.4f}  " f"f1_adolescence={metrics['f1_adolescence']:.4f}  " f"train_s={metrics['train_time_s']:.2f}")
        log(classification_report(y_test, y_pred, target_names=bin_names, zero_division=0))
        log(str(confusion_matrix(y_test, y_pred)))

    log("")
    log("--- Stratified 5-fold CV (balanced weights per train fold) ---")
    for cfg_template in XGB_BINARY_CONFIGS:
        cfg = cfg_template.copy()
        label = cfg["name"]
        y_cv, cv_time = cross_val_predict_balanced(X, y, cfg)
        log("")
        log(f">> [{label}] CV time={cv_time:.2f}s")
        log("    " f"acc={float((y_cv == y).mean()):.4f}  " f"bal_acc={balanced_accuracy_score(y, y_cv):.4f}  " f"macro_f1={f1_score(y, y_cv, average='macro', zero_division=0):.4f}  " f"f1_adolescence={f1_score(y, y_cv, pos_label=1, zero_division=0):.4f}")
        log(classification_report(y, y_cv, target_names=bin_names, zero_division=0))
        log(str(confusion_matrix(y, y_cv)))

    log("")
    log(f"Ended: {datetime.now().isoformat()}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    ts = experiment_start.strftime("%Y%m%d_%H%M%S")
    out = args.results_dir / f"results_xgb_binary_adolescence_{args.eyes}_{args.features}_{ts}.txt"
    out.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
