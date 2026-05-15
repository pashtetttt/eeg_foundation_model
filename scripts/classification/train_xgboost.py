#!/usr/bin/env python3
"""
Train XGBoost on cached features.

Sources (config ``feature_source``):
  - ``merged`` — ``merged_*.npz`` from merge_features.py (handcrafted + DFA + optional embeddings).
  - ``dfa_cache`` — ``dfa_*.npz`` from prepare_dfa_cache.py (DFA columns only, no FM embeddings).
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from scripts.features.feature_utils import merged_cache_path
from scripts.features.prepare_dfa_cache import dfa_cache_path
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
    "dfa_only": {
        "use_handcrafted": False,
        "use_dfa": True,
        "use_embeddings": False,
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


def _plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(5.0, len(labels) * 1.1), max(4.0, len(labels))))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = float(cm.max()) / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_dfa_importance(names: list[str], importances: np.ndarray, out_path: Path, top_k: int = 25) -> None:
    order = np.argsort(importances)[::-1][:top_k]
    labels = [names[i] if i < len(names) else f"f{i}" for i in order]
    vals = importances[order]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(order))))
    ax.barh(range(len(order)), vals[::-1], color="#55A868")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels[::-1], fontsize=7)
    ax.set_xlabel("XGBoost importance")
    ax.set_title(f"Top {len(order)} DFA features")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="XGBoost on cached features.")
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
    feature_source = str(cfg.get("feature_source", "merged")).lower().strip()

    class_names: list[str] | None = cfg.get("class_names")
    if class_names is not None:
        class_names = [str(c) for c in class_names]

    if feature_source == "dfa_cache":
        data_path = dfa_cache_path(results_dir, condition=condition, cohort_name=cohort_name)
        meta_path = results_dir / "features" / f"dfa_metadata_{condition}_{cohort_name}.csv"
        if not data_path.is_file():
            raise FileNotFoundError(f"DFA cache not found: {data_path} (run prepare_dfa_cache.py)")
        if not meta_path.is_file():
            raise FileNotFoundError(f"DFA metadata not found: {meta_path}")
        pack = np.load(data_path)
        Xs = pack["X"]
        y = pack["y"]
        meta = pd.read_csv(meta_path)
        feat_names = meta["name"].astype(str).tolist()
        sources_local = ["dfa"] * Xs.shape[1]
    else:
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
        col_to_meta: dict[int, pd.Series] = {}
        for _, row in meta.iterrows():
            col_to_meta[int(row["column_index"])] = row
        sources_local = []
        feat_names = []
        for c in cols:
            r = col_to_meta.get(int(c))
            sources_local.append(_coarse_source(str(r["source"])) if r is not None else "unknown")
            feat_names.append(str(r["name"]) if r is not None and "name" in r else f"col_{c}")

    rs = int(train_cfg.get("random_state", 42))
    ts = float(train_cfg.get("test_size", 0.2))
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=ts, random_state=rs, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=ts, random_state=rs, stratify=None)

    n_classes = int(len(np.unique(y)))
    clf = xgb.XGBClassifier(
        n_estimators=int(train_cfg.get("xgb_n_estimators", 300)),
        max_depth=int(train_cfg.get("xgb_max_depth", 8)),
        learning_rate=float(train_cfg.get("xgb_learning_rate", 0.05)),
        subsample=float(train_cfg.get("xgb_subsample", 0.9)),
        colsample_bytree=float(train_cfg.get("xgb_colsample_bytree", 0.85)),
        reg_lambda=float(train_cfg.get("xgb_reg_lambda", 1.0)),
        objective="multi:softprob",
        num_class=n_classes,
        random_state=rs,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    acc = float((y_pred == y_te).mean())
    labels_idx = list(range(n_classes))
    if class_names is None or len(class_names) != n_classes:
        class_names = [str(i) for i in labels_idx]
    report = classification_report(y_te, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=labels_idx)

    imp = clf.feature_importances_
    group_gain = {"handcrafted": 0.0, "dfa": 0.0, "embedding": 0.0}
    for i_local, g in enumerate(sources_local):
        if g in group_gain:
            group_gain[g] += float(imp[i_local])

    run_tag = f"{condition}_{cohort_name}"
    base_out = str(out_cfg.get("run_subdir", "classification/xgb_dfa"))
    out_sub = (ROOT / base_out / run_tag).resolve()
    out_sub.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "grouped_importance_sum": group_gain,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "feature_columns_used": int(Xs.shape[1]),
        "feature_source": feature_source,
        "cohort_name": cohort_name,
        "eyes_condition": condition,
    }
    out_json = out_sub / "test_metrics.json"
    out_json.write_text(
        json.dumps(
            {"metrics": metrics, "report": report, "confusion_matrix": cm.tolist(), "class_names": class_names},
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    labs = list(group_gain.keys())
    vals = [group_gain[k] for k in labs]
    ax.bar(labs, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Sum of XGBoost feature importances")
    ax.set_title(f"Contribution by source ({run_tag})")
    fig.tight_layout()
    out_plot = out_sub / "importance_by_source.png"
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    _plot_confusion(cm, class_names, out_sub / "confusion_matrix_test.png", title=f"Test confusion ({run_tag})")

    if feature_source == "dfa_cache" and feat_names:
        _plot_dfa_importance(feat_names, imp, out_sub / "importance_dfa_top25.png")

    print(metrics)
    print(report)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_plot}")
    print(f"Wrote {out_sub / 'confusion_matrix_test.png'}")
    print(f"XGB_OUTPUT_DIR={out_sub.resolve()}")


if __name__ == "__main__":
    main()
