#!/usr/bin/env python3
"""
Train XGBoost on pre-merged cached features (handcrafted + optional DFA + optional embeddings).

Loads ``results/features/merged_{condition}_{cohort}.npz`` and ``merged_metadata_*.csv``,
subsets columns from YAML toggles, and writes grouped feature-importance plot by source.
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
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from scripts.features.feature_utils import merged_cache_path
from scripts.utils.data_handling import load_yaml_config, resolve_data_dir
from scripts.utils.runtime_diag import log_library_versions

EXPERIMENT_PRESETS: dict[str, dict[str, bool]] = {
    "handcrafted_only": {
        "use_handcrafted": True,
        "use_dfa": False,
        "use_embeddings": False,
    },
    "handcrafted_dfa": {
        "use_handcrafted": True,
        "use_dfa": True,
        "use_embeddings": False,
    },
    "handcrafted_embeddings": {
        "use_handcrafted": True,
        "use_dfa": False,
        "use_embeddings": True,
    },
    "all_combined": {
        "use_handcrafted": True,
        "use_dfa": True,
        "use_embeddings": True,
    },
}


def _select_columns(meta: pd.DataFrame, feat_cfg: dict) -> np.ndarray:
    use_h = bool(feat_cfg.get("use_handcrafted", True))
    use_d = bool(feat_cfg.get("use_dfa", True))
    use_e = bool(feat_cfg.get("use_embeddings", True))
    idx = []
    for _, row in meta.iterrows():
        s = str(row["source"]).lower()
        j = int(row["column_index"])
        if s == "handcrafted" and use_h:
            idx.append(j)
        elif s == "dfa" and use_d:
            idx.append(j)
        elif s == "embedding" and use_e:
            idx.append(j)
    return np.array(sorted(set(idx)), dtype=int)


def _coarse_source(s: str) -> str:
    s = s.lower()
    if s == "embedding":
        return "embedding"
    if s == "dfa":
        return "dfa"
    return "handcrafted"


def main() -> None:
    ap = argparse.ArgumentParser(description="XGBoost on merged cached features.")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "xgboost_training.yaml")
    ap.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=sorted(EXPERIMENT_PRESETS.keys()),
        help="Ablations: override features.* toggles (YAML values used if omitted).",
    )
    ap.add_argument("--data-dir", type=Path, default=None, help="Override config data_dir.")
    ap.add_argument("--condition", type=str, default=None, help="Override eyes_condition.")
    ap.add_argument("--cohort-name", type=str, default=None, help="Override cohort_name.")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    if args.data_dir is not None:
        cfg["data_dir"] = str(args.data_dir.expanduser().resolve())
    if args.condition is not None:
        cfg["eyes_condition"] = args.condition
    if args.cohort_name is not None:
        cfg["cohort_name"] = args.cohort_name

    log_library_versions("numpy", "pandas", "sklearn", "xgboost")

    _ = resolve_data_dir(cfg)
    results_dir = Path(cfg.get("results_dir", "results")).resolve()
    condition = str(cfg.get("eyes_condition", "closed"))
    cohort_name = str(cfg.get("cohort_name", "cohort"))
    feat_cfg = dict(cfg.get("features") or {})
    if args.experiment is not None:
        feat_cfg.update(EXPERIMENT_PRESETS[args.experiment])
        print(f"[train_xgboost] experiment={args.experiment} -> features: {feat_cfg}")
    train_cfg = cfg.get("train") or {}
    out_cfg = cfg.get("output") or {}

    merged_path = merged_cache_path(results_dir, condition=condition, cohort_name=cohort_name)
    meta_path = results_dir / "features" / f"merged_metadata_{condition}_{cohort_name}.csv"
    if not merged_path.is_file():
        raise FileNotFoundError(f"Merged cache not found: {merged_path} (run merge_features.py)")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Merged metadata not found: {meta_path}")

    pack = np.load(merged_path)
    X = pack["X"]
    y = pack["y"]
    meta = pd.read_csv(meta_path)

    cols = _select_columns(meta, feat_cfg)
    if cols.size == 0:
        raise ValueError("No columns selected; check features.* toggles in config.")
    Xs = X[:, cols]

    # Remap metadata rows to local column indices
    col_to_meta: dict[int, pd.Series] = {}
    for _, row in meta.iterrows():
        col_to_meta[int(row["column_index"])] = row
    sources_local: list[str] = []
    for c in cols:
        r = col_to_meta.get(int(c))
        sources_local.append(_coarse_source(str(r["source"])) if r is not None else "unknown")

    rs = int(train_cfg.get("random_state", 42))
    ts = float(train_cfg.get("test_size", 0.2))
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=ts, random_state=rs, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=ts, random_state=rs, stratify=None)

    clf = xgb.XGBClassifier(
        n_estimators=int(train_cfg.get("xgb_n_estimators", 300)),
        max_depth=int(train_cfg.get("xgb_max_depth", 8)),
        learning_rate=float(train_cfg.get("xgb_learning_rate", 0.05)),
        subsample=float(train_cfg.get("xgb_subsample", 0.9)),
        colsample_bytree=float(train_cfg.get("xgb_colsample_bytree", 0.85)),
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

    imp = clf.feature_importances_
    # Map local importances back to global column index for grouping
    group_gain = {"handcrafted": 0.0, "dfa": 0.0, "embedding": 0.0}
    for i_local, g in enumerate(sources_local):
        if g in group_gain:
            group_gain[g] += float(imp[i_local])

    metrics = {
        "macro_f1": macro_f1,
        "grouped_importance_sum": group_gain,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "feature_columns_used": int(cols.size),
    }
    out_json = Path(out_cfg.get("metrics_json", "results/classification/xgb_metrics.json"))
    out_json = out_json if out_json.is_absolute() else (ROOT / out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"metrics": metrics, "report": report}, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 4))
    labs = list(group_gain.keys())
    vals = [group_gain[k] for k in labs]
    ax.bar(labs, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Sum of XGBoost feature importances")
    ax.set_title("Contribution by feature source (subset)")
    fig.tight_layout()
    out_plot = Path(out_cfg.get("importance_plot", "results/classification/xgb_importance_by_source.png"))
    out_plot = out_plot if out_plot.is_absolute() else (ROOT / out_plot).resolve()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    print(metrics)
    print(report)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_plot}")


if __name__ == "__main__":
    main()
