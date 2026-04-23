"""
Обучение XGBoost на полном датасете `data`, оценка на `data_kids` (внешняя выборка).

Восемь конфигураций: eyes closed/open × (all, alpha, selected, all_plus_complexity).
Для каждой пары (eyes, features) берётся лучший по CV XGB-конфиг из
`results/experiment_summary_20260327_125519.csv` (те же имена, что в train_xgboost_experiments),
кроме all_plus_complexity — для него по умолчанию xgb_regularized.

Пример:
  python train_xgb_data_kids_holdout.py
  python train_xgb_data_kids_holdout.py --max 5
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from eeg_experiment_shared import (
    DATA_DIR,
    DEFAULT_SELECTED_FEATURES_PATH,
    GROUPS_DATA_KIDS,
    RESULTS_DIR,
    load_and_prepare_matrix,
)
from eeg_features import feature_description
from train_xgboost_experiments import XGB_CONFIGS, build_pipeline, fit_with_balanced_weights

warnings.filterwarnings("ignore", message=".*does not conform to MNE naming conventions.*")

# Лучший xgb_* по macro_f1 (5-fold CV) из experiment_summary_20260327_125519.csv
BEST_XGB_CONFIG_NAME: dict[tuple[str, str], str] = {
    ("closed", "all"): "xgb_regularized",
    ("closed", "alpha"): "xgb_baseline",
    ("closed", "selected"): "xgb_shallow_lr",
    ("closed", "all_plus_complexity"): "xgb_regularized",
    ("open", "all"): "xgb_baseline",
    ("open", "alpha"): "xgb_baseline",
    ("open", "selected"): "xgb_shallow_lr",
    ("open", "all_plus_complexity"): "xgb_regularized",
}

RUNS: list[tuple[str, str]] = [
    ("closed", "all"),
    ("closed", "alpha"),
    ("closed", "selected"),
    ("closed", "all_plus_complexity"),
    ("open", "all"),
    ("open", "alpha"),
    ("open", "selected"),
    ("open", "all_plus_complexity"),
]


def get_xgb_cfg(name: str) -> dict:
    for c in XGB_CONFIGS:
        if c["name"] == name:
            return c.copy()
    raise ValueError(f"Unknown XGB config name: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGB on data/, evaluate on data_kids/ (6 eyes×features runs)."
    )
    parser.add_argument("--max", type=int, default=None, help="Max EDF files per class (debug)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Training root (default: data)")
    parser.add_argument("--test-dir", type=Path, default=Path("data_kids"), help="Test root (default: data_kids)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Output directory")
    parser.add_argument("--selected-path", type=str, default=str(DEFAULT_SELECTED_FEATURES_PATH))
    args = parser.parse_args()

    experiment_start = datetime.now()
    ts = experiment_start.strftime("%Y%m%d_%H%M%S")
    log_lines: list[str] = []
    csv_rows: list[dict[str, object]] = []

    def log(s: str) -> None:
        log_lines.append(s)
        print(s)

    log("=" * 72)
    log("XGB: train on data/ → test on data_kids/ (best CV config per eyes×features)")
    log(f"Started: {experiment_start.isoformat()}")
    log(f"Train dir: {args.data_dir.resolve()}")
    log(f"Test dir:  {args.test_dir.resolve()}")
    log("=" * 72)

    for eyes, features in RUNS:
        cfg_name = BEST_XGB_CONFIG_NAME[(eyes, features)]
        cfg = get_xgb_cfg(cfg_name)
        selected_path = Path(args.selected_path) if features == "selected" else None
        include_lit = features == "all_plus_complexity"

        log("")
        log("-" * 72)
        log(f"Eyes: {eyes} | Features: {features} | Model: {cfg_name}")
        log(f"Params: {cfg}")

        log("Loading train (data)...")
        X_tr, y_tr, class_names_tr, sel_notes = load_and_prepare_matrix(
            eyes,
            features,
            args.max,
            selected_path,
            data_dir=args.data_dir,
            groups=None,
        )
        for n in sel_notes:
            log(f"  Feature selection: {n}")
        log(feature_description(eyes_condition=eyes, include_literature=include_lit))
        log(f"Train samples: {X_tr.shape[0]}, features: {X_tr.shape[1]}")
        for i, name in enumerate(class_names_tr):
            log(f"  Class {name}: {(y_tr == i).sum()}")

        log("Loading test (data_kids)...")
        X_te, y_te, class_names_te, _ = load_and_prepare_matrix(
            eyes,
            features,
            args.max,
            selected_path,
            data_dir=args.test_dir,
            groups=GROUPS_DATA_KIDS,
        )
        log(f"Test samples: {X_te.shape[0]}, features: {X_te.shape[1]}")
        for i, name in enumerate(class_names_te):
            log(f"  Class {name}: {(y_te == i).sum()}")

        if class_names_tr != class_names_te:
            raise RuntimeError(f"Class name mismatch train vs test: {class_names_tr} vs {class_names_te}")

        pipe = build_pipeline(cfg)
        t0 = time.perf_counter()
        fit_with_balanced_weights(pipe, X_tr, y_tr)
        train_s = time.perf_counter() - t0

        y_pred = pipe.predict(X_te)
        acc = float((y_pred == y_te).mean())
        bal_acc = float(balanced_accuracy_score(y_te, y_pred))
        macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))

        log("")
        log(f"Train time: {train_s:.2f}s")
        log(
            f"Test metrics: acc={acc:.4f}  bal_acc={bal_acc:.4f}  macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}"
        )
        log(classification_report(y_te, y_pred, target_names=class_names_tr, zero_division=0))
        log(str(confusion_matrix(y_te, y_pred)))

        csv_rows.append(
            {
                "eyes": eyes,
                "features": features,
                "best_config": cfg_name,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "train_time_s": train_s,
                "train_n": X_tr.shape[0],
                "test_n": X_te.shape[0],
                "train_features": X_tr.shape[1],
            }
        )

    log("")
    log(f"Ended: {datetime.now().isoformat()}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_txt = args.results_dir / f"results_xgb_data_kids_holdout_{ts}.txt"
    out_txt.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nSaved log: {out_txt}")

    out_csv = args.results_dir / f"experiment_summary_data_kids_holdout_{ts}.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
