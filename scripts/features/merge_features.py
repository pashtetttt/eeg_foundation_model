#!/usr/bin/env python3
"""
Join cached handcrafted+DFA matrix with embeddings on ``subject_id``.

Fits ``StandardScaler`` on a stratified train fraction of subjects (default 80%)
separately for the (handcrafted+DFA) block and the embedding block, then transforms
all rows for a single design matrix used downstream. Saves joblib scalers for inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.features.feature_utils import (
    embeddings_cache_path,
    features_cache_path,
    merged_cache_path,
)
from scripts.utils.data_handling import load_yaml_config, resolve_data_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge cached features + embeddings.")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "merge_features.yaml")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    _ = resolve_data_dir(cfg)
    results_dir = Path(cfg.get("results_dir", "results")).resolve()
    condition = str(cfg.get("eyes_condition", "closed"))
    cohort_name = str(cfg.get("cohort_name", "cohort"))
    model_name = str(cfg.get("embedding_model", "eegpt"))
    scaler_train_frac = float(cfg.get("scaler_fit_train_fraction", 0.8))
    seed = int(cfg.get("seed", 42))

    feat_path = features_cache_path(results_dir, condition=condition, cohort_name=cohort_name)
    emb_path = embeddings_cache_path(results_dir, model=model_name, condition=condition, cohort_name=cohort_name)
    meta_path = results_dir / "features" / f"feature_metadata_{condition}_{cohort_name}.csv"
    map_path = results_dir / "features" / f"subject_mapping_{condition}_{cohort_name}.csv"

    if not feat_path.is_file():
        raise FileNotFoundError(f"Features cache missing: {feat_path}")
    if not emb_path.is_file():
        raise FileNotFoundError(f"Embeddings cache missing: {emb_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Feature metadata missing: {meta_path}")

    Xf = np.load(feat_path)
    map_df = pd.read_csv(map_path)
    # Backward compatible:
    # - new embeddings files store subject_ids as unicode (no pickle needed)
    # - older files may have dtype=object and require allow_pickle=True
    emb = np.load(emb_path)
    E = emb["embeddings"]
    try:
        sid_raw = emb["subject_ids"]
    except ValueError:
        emb = np.load(emb_path, allow_pickle=True)
        sid_raw = emb["subject_ids"]
    sid_emb = [str(s) for s in sid_raw]
    emb_dict = {s: E[i] for i, s in enumerate(sid_emb)}

    sid_feat = map_df["subject_id"].astype(str).tolist()
    n0 = len(sid_feat)
    rows_keep: list[int] = []
    missing_emb = 0
    for i, s in enumerate(sid_feat):
        if s not in emb_dict:
            missing_emb += 1
            continue
        rows_keep.append(i)

    if missing_emb:
        print(f"Dropped {missing_emb} / {n0} rows missing embeddings (inner join).")

    if not rows_keep:
        raise RuntimeError("No rows left after joining features with embeddings.")

    X_h = Xf[rows_keep]
    y = map_df["y"].values[rows_keep] if "y" in map_df.columns else map_df["label_idx"].values[rows_keep]
    subj = np.asarray([sid_feat[i] for i in rows_keep], dtype=str)
    X_e = np.stack([emb_dict[str(sid_feat[i])] for i in rows_keep], axis=0)

    meta = pd.read_csv(meta_path)
    n_hc = int((meta["type"] == "handcrafted").sum())
    n_dfa = int((meta["type"] == "dfa").sum())
    if n_hc + n_dfa != X_h.shape[1]:
        print(
            f"Warning: feature matrix cols {X_h.shape[1]} != metadata handcrafted+dfa {n_hc + n_dfa}; "
            "using full X_h as single block for scaling."
        )
        n_block_a = X_h.shape[1]
        X_a = X_h.astype(np.float64)
    else:
        n_block_a = n_hc + n_dfa
        X_a = X_h[:, :n_block_a].astype(np.float64)
    X_b = X_e.astype(np.float64)

    idx = np.arange(len(y))
    try:
        train_idx, _test_idx = train_test_split(
            idx,
            train_size=scaler_train_frac,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        train_idx, _test_idx = train_test_split(
            idx,
            train_size=scaler_train_frac,
            random_state=seed,
            stratify=None,
        )

    sc_a = StandardScaler()
    sc_b = StandardScaler()
    sc_a.fit(X_a[train_idx])
    sc_b.fit(X_b[train_idx])
    X_a_t = sc_a.transform(X_a)
    X_b_t = sc_b.transform(X_b)
    X_full = np.hstack([X_a_t, X_b_t]).astype(np.float32)

    out_npz = merged_cache_path(results_dir, condition=condition, cohort_name=cohort_name)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(sc_a, out_npz.with_name(out_npz.stem + "_scaler_handcrafted.joblib"))
    joblib.dump(sc_b, out_npz.with_name(out_npz.stem + "_scaler_embedding.joblib"))
    np.savez_compressed(out_npz, X=X_full, y=y.astype(np.int64), subject_ids=subj)

    merged_rows = []
    for j in range(X_a.shape[1]):
        if j < len(meta):
            r = meta.iloc[j]
            merged_rows.append({"column_index": j, "name": str(r["name"]), "source": str(r["type"])})
        else:
            merged_rows.append({"column_index": j, "name": f"col_{j}", "source": "handcrafted"})
    off = X_a.shape[1]
    for j in range(X_b.shape[1]):
        merged_rows.append(
            {"column_index": off + j, "name": f"emb_{j}", "source": "embedding"}
        )
    pd.DataFrame(merged_rows).to_csv(
        results_dir / "features" / f"merged_metadata_{condition}_{cohort_name}.csv",
        index=False,
    )
    print(f"Saved merged matrix {X_full.shape} -> {out_npz}")


if __name__ == "__main__":
    main()
