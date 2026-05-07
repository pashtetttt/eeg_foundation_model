#!/usr/bin/env python3
"""
Train XGBoost on cached REVE embeddings for age-group classification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from scripts.features.feature_utils import embeddings_cache_path
from scripts.utils.data_handling import load_yaml_config, resolve_data_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Train XGBoost on REVE embeddings.")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "reve_xgboost.yaml")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    _ = resolve_data_dir(cfg)
    results_dir = Path(cfg.get("results_dir", "results")).resolve()
    condition = str(cfg.get("eyes_condition", "closed"))
    cohort_name = str(cfg.get("cohort_name", "cohort"))
    train_cfg = cfg.get("train", {}) or {}
    out_cfg = cfg.get("output", {}) or {}

    emb_path = embeddings_cache_path(results_dir, model="reve", condition=condition, cohort_name=cohort_name)
    if not emb_path.is_file():
        raise FileNotFoundError(f"REVE embeddings not found: {emb_path} (run extract_reve_embeddings.py first)")

    pack = np.load(emb_path)
    X = pack["embeddings"].astype(np.float32)
    y = pack["y"].astype(np.int64)

    rs = int(train_cfg.get("random_state", 42))
    ts = float(train_cfg.get("test_size", 0.2))
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts, random_state=rs, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts, random_state=rs, stratify=None)

    clf = xgb.XGBClassifier(
        n_estimators=int(train_cfg.get("xgb_n_estimators", 400)),
        max_depth=int(train_cfg.get("xgb_max_depth", 8)),
        learning_rate=float(train_cfg.get("xgb_learning_rate", 0.05)),
        subsample=float(train_cfg.get("xgb_subsample", 0.9)),
        colsample_bytree=float(train_cfg.get("xgb_colsample_bytree", 0.9)),
        reg_lambda=float(train_cfg.get("xgb_reg_lambda", 1.0)),
        objective="multi:softprob",
        num_class=int(len(np.unique(y))),
        random_state=rs,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    report = classification_report(y_te, y_pred, zero_division=0)

    metrics = {
        "macro_f1": macro_f1,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "embedding_dim": int(X.shape[1]),
    }
    out_json = Path(out_cfg.get("metrics_json", "results/classification/reve_xgb_metrics.json"))
    out_json = out_json if out_json.is_absolute() else (ROOT / out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"metrics": metrics, "report": report}, indent=2), encoding="utf-8")

    print(metrics)
    print(report)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()

