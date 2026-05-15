#!/usr/bin/env python3
"""
Build a DFA-only feature matrix from cached ``features_*.npy`` + ``feature_metadata_*.csv``.

One row per recording (aligned with ``subject_mapping_*.csv``). Applies StandardScaler
fit on a stratified train fraction. No embeddings required.

Outputs:
  results/features/dfa_{condition}_{cohort_name}.npz  — X, y, subject_ids
  results/features/dfa_metadata_{condition}_{cohort_name}.csv
  results/features/dfa_{condition}_{cohort_name}_scaler.joblib
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

from scripts.features.feature_utils import features_cache_path
from scripts.utils.data_handling import load_yaml_config, resolve_data_dir
from scripts.utils.runtime_diag import log_library_versions


def dfa_cache_path(results_dir: Path, *, condition: str, cohort_name: str) -> Path:
    return results_dir / "features" / f"dfa_{condition}_{cohort_name}.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description="Export DFA columns from cached feature matrix for XGBoost.")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "feature_extraction.yaml")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--condition", type=str, default=None)
    ap.add_argument("--cohort-name", type=str, default=None)
    ap.add_argument("--force", action="store_true", help="Overwrite existing dfa_*.npz")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    if args.data_dir is not None:
        cfg["data_dir"] = str(args.data_dir.expanduser().resolve())
    if args.condition is not None:
        cfg["eyes_condition"] = args.condition
    if args.cohort_name is not None:
        cfg["cohort_name"] = args.cohort_name

    log_library_versions("numpy", "pandas", "sklearn")

    _ = resolve_data_dir(cfg)
    results_dir = Path(cfg.get("results_dir", "results")).resolve()
    condition = str(cfg.get("eyes_condition", "closed"))
    cohort_name = str(cfg.get("cohort_name", "cohort"))
    scaler_train_frac = float(cfg.get("scaler_fit_train_fraction", 0.8))
    seed = int(cfg.get("seed", 42))

    feat_path = features_cache_path(results_dir, condition=condition, cohort_name=cohort_name)
    meta_path = results_dir / "features" / f"feature_metadata_{condition}_{cohort_name}.csv"
    map_path = results_dir / "features" / f"subject_mapping_{condition}_{cohort_name}.csv"
    out_npz = dfa_cache_path(results_dir, condition=condition, cohort_name=cohort_name)

    if out_npz.is_file() and not args.force:
        print(f"Exists, skipping: {out_npz} (use --force)")
        return

    for p, label in [(feat_path, "features"), (meta_path, "metadata"), (map_path, "subject_mapping")]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing {label}: {p} (run compute_all_features.py first)")

    X_all = np.load(feat_path)
    meta = pd.read_csv(meta_path)
    map_df = pd.read_csv(map_path)

    dfa_mask = meta["type"].astype(str).str.lower() == "dfa"
    if not dfa_mask.any():
        raise ValueError(f"No DFA columns in {meta_path}")
    dfa_cols = np.where(dfa_mask.values)[0]
    if X_all.shape[1] != len(meta):
        print(
            f"Warning: feature matrix cols {X_all.shape[1]} != metadata rows {len(meta)}; "
            "using metadata index positions."
        )
    X_dfa = X_all[:, dfa_cols].astype(np.float64)

    if "y" in map_df.columns:
        y = map_df["y"].values.astype(np.int64)
    else:
        y = map_df["label_idx"].values.astype(np.int64)
    if len(y) != X_dfa.shape[0]:
        raise ValueError(f"Row count mismatch: X {X_dfa.shape[0]} vs mapping {len(y)}")

    sid = map_df["subject_id"].astype(str).tolist()

    idx = np.arange(len(y))
    try:
        train_idx, _ = train_test_split(idx, train_size=scaler_train_frac, random_state=seed, stratify=y)
    except ValueError:
        train_idx, _ = train_test_split(idx, train_size=scaler_train_frac, random_state=seed, stratify=None)

    sc = StandardScaler()
    sc.fit(X_dfa[train_idx])
    X_scaled = sc.transform(X_dfa).astype(np.float32)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=X_scaled,
        y=y,
        subject_ids=np.asarray(sid, dtype=str),
    )
    joblib.dump(sc, out_npz.with_name(out_npz.stem + "_scaler.joblib"))

    meta_out = meta.loc[dfa_mask].reset_index(drop=True).copy()
    meta_out["column_index"] = np.arange(len(meta_out), dtype=int)
    meta_out["source"] = "dfa"
    meta_path_out = results_dir / "features" / f"dfa_metadata_{condition}_{cohort_name}.csv"
    meta_out.to_csv(meta_path_out, index=False)

    print(f"Saved DFA matrix {X_scaled.shape} -> {out_npz}")
    print(f"Saved metadata ({len(meta_out)} features) -> {meta_path_out}")
    print(f"DFA_CACHE_PATH={out_npz.resolve()}")


if __name__ == "__main__":
    main()
