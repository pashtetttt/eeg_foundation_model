#!/usr/bin/env python3
"""
Extract frozen foundation-model embeddings for each recording listed in subject_mapping.

Outputs:
  results/embeddings/embeddings_{model}_{condition}_{cohort_name}.npz
    arrays: embeddings (N, D), subject_ids (object), y (int64, optional)

Can be imported: ``from scripts.embeddings.extract_embeddings import load_window_tensor``.
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

import numpy as np
import pandas as pd
import torch

from edf_loader import load_raw_edf_resilient
from eeg_thesis.eegpt_adapter import DEFAULT_19_CH, remap_channels_to_target, resample_window
from eeg_thesis.eegpt_data import _normalize_per_channel  # reuse

from scripts.embeddings.model_loader import (
    build_channel_adapter_if_needed,
    eegpt_forward_embeddings,
    heegnet_domain_batch,
    heegnet_forward_embeddings,
    load_eegpt_encoder,
    load_heegnet_encoder,
    numpy_window_to_heegnet_tensor,
)
from scripts.features.feature_utils import embeddings_cache_path
from scripts.utils.data_handling import load_yaml_config, resolve_data_dir


def load_window_tensor(
    edf_path: Path,
    *,
    target_sfreq: float,
    window_seconds: float,
    zscore_per_channel: bool = True,
) -> np.ndarray:
    """
    Return (C=19, T) float32 window aligned with ``load_eegpt_dataset_from_edf`` preprocessing.
    """
    raw = load_raw_edf_resilient(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    if not np.all(np.isfinite(data)):
        raise ValueError("non-finite raw")
    source_names = [ch.upper().strip(".") for ch in raw.ch_names]
    x = remap_channels_to_target(data, source_names, DEFAULT_19_CH)
    x = resample_window(
        x,
        src_sfreq=float(raw.info["sfreq"]),
        target_sfreq=target_sfreq,
        window_seconds=window_seconds,
    )
    if zscore_per_channel:
        x = _normalize_per_channel(x)
    return x.astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract frozen FM embeddings for recordings in subject_mapping.")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "embedding_extraction.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    data_dir = resolve_data_dir(cfg)
    results_dir = Path(cfg.get("results_dir", "results")).resolve()
    condition = str(cfg.get("eyes_condition", "closed"))
    cohort_name = str(cfg.get("cohort_name", "cohort"))
    model_name = str(cfg.get("model", "eegpt")).lower().strip()
    target_sfreq = float(cfg.get("target_sfreq", 256.0))
    window_seconds = float(cfg.get("window_seconds", 4.0))
    zscore = bool(cfg.get("zscore_per_channel", True))
    batch_size = int(cfg.get("batch_size", 8))
    seed = int(cfg.get("seed", 42))
    device_s = str(cfg.get("device", "cuda"))
    use_cuda = device_s == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    map_rel = cfg.get(
        "subject_mapping_csv",
        f"features/subject_mapping_{condition}_{cohort_name}.csv",
    )
    mapping_path = (results_dir / str(map_rel)).resolve()
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Subject mapping not found: {mapping_path}. Run compute_all_features.py first with matching cohort/condition."
        )

    out_npz = embeddings_cache_path(results_dir, model=model_name, condition=condition, cohort_name=cohort_name)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    if out_npz.is_file() and not args.force:
        print(f"Exists, skipping: {out_npz}")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(mapping_path)
    if "edf_path" not in df.columns or "subject_id" not in df.columns:
        raise ValueError(f"Mapping CSV missing columns: {mapping_path}")

    vecs: list[np.ndarray] = []
    ids: list[str] = []
    ys: list[int] = []

    if model_name == "eegpt":
        ckpt = cfg.get("eegpt_checkpoint")
        ckpt_path = Path(ckpt) if ckpt else None
        if ckpt_path and not ckpt_path.is_absolute():
            ckpt_path = (ROOT / ckpt_path).resolve()
        adapter, _exp_c = load_eegpt_encoder(ROOT, ckpt_path, device=device)
        for _, row in df.iterrows():
            p = Path(row["edf_path"])
            if not p.is_file():
                print(f"  skip missing file: {p}")
                continue
            try:
                x = load_window_tensor(p, target_sfreq=target_sfreq, window_seconds=window_seconds, zscore_per_channel=zscore)
                xt = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)
                z = eegpt_forward_embeddings(adapter, xt)
                vecs.append(z.detach().cpu().numpy().reshape(-1))
                ids.append(str(row["subject_id"]))
                ys.append(int(row.get("y", row.get("label_idx", 0))))
            except Exception as e:
                print(f"  skip {row['subject_id']}: {e}")

    elif model_name == "heegnet":
        ckpt = cfg.get("heegnet_checkpoint")
        if not ckpt:
            raise ValueError("heegnet_checkpoint required in config for model=heegnet")
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = (ROOT / ckpt_path).resolve()
        model, meta, exp_c = load_heegnet_encoder(ROOT, ckpt_path, device=device, dtype=torch.float64)
        in_c = len(DEFAULT_19_CH)
        ch_adapt = build_channel_adapter_if_needed(in_c, exp_c, device=device, dtype=torch.float64)

        batch_x: list[torch.Tensor] = []
        batch_meta: list[tuple[str, int]] = []

        def flush_batch() -> None:
            nonlocal batch_x, batch_meta, vecs, ids, ys
            if not batch_x:
                return
            bx = torch.cat(batch_x, dim=0)
            dom = heegnet_domain_batch(bx.shape[0], meta, device)
            z = heegnet_forward_embeddings(model, bx, dom, channel_adapter=None)
            zn = z.detach().cpu().numpy()
            for i in range(zn.shape[0]):
                vecs.append(zn[i])
            for sid, yv in batch_meta:
                ids.append(sid)
                ys.append(yv)
            batch_x = []
            batch_meta = []

        for _, row in df.iterrows():
            p = Path(row["edf_path"])
            if not p.is_file():
                print(f"  skip missing file: {p}")
                continue
            try:
                x32 = load_window_tensor(p, target_sfreq=target_sfreq, window_seconds=window_seconds, zscore_per_channel=zscore)
                t = torch.from_numpy(x32.astype(np.float64)).unsqueeze(0).to(device=device, dtype=torch.float64)
                if ch_adapt is not None:
                    t = ch_adapt(t)
                else:
                    t = numpy_window_to_heegnet_tensor(x32, expected_c=exp_c, device=device, dtype=torch.float64)
                batch_x.append(t)
                batch_meta.append((str(row["subject_id"]), int(row.get("y", row.get("label_idx", 0)))))
                if len(batch_x) >= batch_size:
                    flush_batch()
            except Exception as e:
                print(f"  skip {row['subject_id']}: {e}")
        flush_batch()

    else:
        raise ValueError(f"Unknown model '{model_name}' (use eegpt or heegnet)")

    if not vecs:
        raise RuntimeError("No embeddings extracted.")

    E = np.stack(vecs, axis=0).astype(np.float32)
    y_arr = np.asarray(ys, dtype=np.int64)
    sid = np.asarray(ids, dtype=object)
    np.savez_compressed(out_npz, embeddings=E, subject_ids=sid, y=y_arr)
    print(f"Saved {out_npz} shape={E.shape}")


if __name__ == "__main__":
    main()
