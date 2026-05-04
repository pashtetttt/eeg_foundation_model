#!/usr/bin/env python3
"""
Compute and cache handcrafted EEG features + DFA block for all recordings under data_dir.

Outputs (under results_dir from config):
  - results/features/features_{condition}_{cohort_name}.npy  — float32 matrix (n_samples, n_features)
  - results/features/feature_metadata.csv
  - results/features/subject_mapping.csv

Skips if the .npy exists unless --force.
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

import mne
import numpy as np
import pandas as pd

from eeg_features import N_CHANNELS, extract_all_features, get_feature_names
from edf_loader import load_raw_edf_resilient

from scripts.features.dfa_utils import compute_dfa_feature_block
from scripts.features.feature_utils import features_cache_path, save_subject_mapping_csv
from scripts.utils.data_handling import build_recording_manifest, load_yaml_config, resolve_data_dir


def _eeg_data_19(raw: mne.io.BaseRaw, canonical_ch_names: list[str] | None) -> tuple[np.ndarray, list[str]]:
    """Match extract_all_features channel selection: up to N_CHANNELS EEG picks."""
    ch_names = raw.ch_names
    if canonical_ch_names is None:
        picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(picks) == 0:
            picks = list(range(min(N_CHANNELS, raw.info["nchan"])))
        picks = picks[:N_CHANNELS]
        canonical = [ch_names[i] for i in picks]
        while len(canonical) < N_CHANNELS:
            canonical.append("")
        canonical = canonical[:N_CHANNELS]
    else:
        picks = []
        for name in canonical_ch_names:
            if name and name in ch_names:
                picks.append(ch_names.index(name))
            else:
                break
        if len(picks) == 0:
            return np.zeros((N_CHANNELS, int(raw.n_times)), dtype=float), list(canonical_ch_names or [])
        picks = picks[:N_CHANNELS]
        canonical = canonical_ch_names
    data, _ = raw.get_data(picks=picks, return_times=True)
    if data.shape[0] < N_CHANNELS:
        data = np.pad(data, ((0, N_CHANNELS - data.shape[0]), (0, 0)), constant_values=0.0)
    data = np.asarray(data[:N_CHANNELS], dtype=float)
    return data, canonical


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache handcrafted + DFA features for all EDFs under data_dir.")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "feature_extraction.yaml")
    ap.add_argument("--force", action="store_true", help="Recompute even if output .npy exists.")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    data_dir = resolve_data_dir(cfg)
    results_dir = Path(cfg.get("results_dir", "results")).resolve()
    condition = str(cfg.get("eyes_condition", "closed"))
    cohort_name = str(cfg.get("cohort_name", "cohort"))
    include_literature = bool(cfg.get("include_literature", False))
    max_per_group = cfg.get("max_per_group")
    max_per_group = int(max_per_group) if max_per_group is not None else None

    dfa_scales = cfg.get("dfa_scales")
    if dfa_scales is None:
        dfa_scales = list(range(1, 21))
    dfa_poly = int(cfg.get("dfa_polynomial_order", 1))

    out_npy = features_cache_path(results_dir, condition=condition, cohort_name=cohort_name)
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    if out_npy.is_file() and not args.force:
        print(f"Exists, skipping: {out_npy} (use --force to overwrite)")
        return

    manifest = build_recording_manifest(
        data_dir,
        eyes_condition=condition,
        max_per_group=max_per_group,
    )
    if not manifest:
        raise FileNotFoundError(f"No recordings found under {data_dir}")

    X_rows: list[np.ndarray] = []
    y_list: list[int] = []
    mapping_rows: list[dict] = []
    canonical_ch_names: list[str] | None = None
    dfa_names: list[str] = []

    for row in manifest:
        try:
            raw = load_raw_edf_resilient(row.edf_path, preload=True, verbose=False)
            if not np.all(np.isfinite(raw.get_data())):
                raise ValueError("non-finite raw")

            out = extract_all_features(
                raw,
                canonical_ch_names,
                eyes_condition=condition,
                include_literature=include_literature,
            )
            if canonical_ch_names is None:
                handcrafted, canonical_ch_names = out  # type: ignore[misc]
            else:
                handcrafted = out  # type: ignore[assignment]
            handcrafted = np.asarray(handcrafted, dtype=float).ravel()

            data_19, _ = _eeg_data_19(raw, canonical_ch_names)
            sfreq = float(raw.info["sfreq"])
            dfa_vec, dfa_names = compute_dfa_feature_block(
                data_19,
                sfreq,
                scales=dfa_scales,
                poly_order=dfa_poly,
            )

            feat = np.concatenate([handcrafted, dfa_vec]).astype(np.float32)
            X_rows.append(feat)
            y_list.append(row.label_idx)
            mapping_rows.append(
                {
                    "subject_id": row.subject_id,
                    "edf_path": str(row.edf_path),
                    "label_idx": row.label_idx,
                    "label_name": row.label_name,
                    "group_folder": row.group_folder,
                }
            )
        except Exception as e:
            print(f"  skip {row.subject_id}: {e}")

    if not X_rows:
        raise RuntimeError("No rows successfully processed.")

    X = np.stack(X_rows, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    np.save(out_npy, X)

    for i, r in enumerate(mapping_rows):
        r["y"] = int(y[i])

    hc_names = get_feature_names(condition, include_literature=include_literature)

    meta_rows = []
    for name in hc_names:
        meta_rows.append({"name": name, "type": "handcrafted", "cluster": _cluster_handcrafted(name)})
    for name in dfa_names:
        meta_rows.append({"name": name, "type": "dfa", "cluster": "dfa"})

    meta_path = results_dir / "features" / f"feature_metadata_{condition}_{cohort_name}.csv"
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    map_path = results_dir / "features" / f"subject_mapping_{condition}_{cohort_name}.csv"
    save_subject_mapping_csv(map_path, mapping_rows)

    print(f"Saved X shape {X.shape} -> {out_npy}")
    print(f"Saved metadata -> {meta_path}")
    print(f"Saved mapping ({len(mapping_rows)} rows) -> {map_path}")


def _cluster_handcrafted(name: str) -> str:
    if name.startswith("ratio_"):
        return "band_ratio"
    if name.startswith("centroid"):
        return "centroid"
    if "ent" in name or "samp" in name or "app" in name:
        return "entropy"
    if name.startswith("envfreq"):
        return "envelope"
    if name.startswith("hfd"):
        return "hfd"
    if "hjorth" in name:
        return "hjorth"
    if "alpha_var" in name:
        return "alpha_var"
    if "theta" in name or "alpha_power" in name or "pred_" in name:
        return "topography"
    return "band_or_other"


if __name__ == "__main__":
    main()
