#!/usr/bin/env python3
"""Extract EEGPT embeddings for hybrid modeling with handcrafted features."""

from __future__ import annotations

from _bootstrap import *  # noqa: F401,F403

import argparse
from pathlib import Path

import numpy as np

from eeg_thesis.eegpt_adapter import DEFAULT_19_CH, EEGPTAdapter
from eeg_thesis.eegpt_data import load_eegpt_dataset_from_edf, to_binary_adolescence


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract EEGPT embeddings.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--eyes", type=str, default="closed", choices=["closed", "open"])
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--target-sfreq", type=float, default=256.0)
    ap.add_argument("--window-seconds", type=float, default=4.0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mode", type=str, default="age4", choices=["age4", "adolescence_binary"])
    ap.add_argument("--out", type=Path, default=Path("results/eegpt_features.npz"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_eegpt_dataset_from_edf(
        data_dir=args.data_dir,
        eyes_condition=args.eyes,
        max_per_group=args.max,
        target_channels=DEFAULT_19_CH,
        target_sfreq=args.target_sfreq,
        window_seconds=args.window_seconds,
    )
    y = ds.y
    class_names = ds.class_names
    num_classes = 4
    if args.mode == "adolescence_binary":
        y, class_names = to_binary_adolescence(ds.y, ds.class_names)
        num_classes = 1

    adapter = EEGPTAdapter(
        repo_root=Path("."),
        checkpoint_path=args.checkpoint,
        channels=ds.channels,
        num_classes=num_classes,
        target_sfreq=args.target_sfreq,
        window_seconds=args.window_seconds,
        device=args.device,
    )

    feats = []
    for i in range(0, ds.X.shape[0], args.batch_size):
        xb = ds.X[i : i + args.batch_size]
        emb = adapter.extract_embeddings(xb).cpu().numpy()
        emb = emb.reshape((emb.shape[0], -1))
        feats.append(emb)
    X_emb = np.concatenate(feats, axis=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X_emb.astype(np.float32),
        y=y.astype(np.int64),
        class_names=np.array(class_names, dtype=object),
        source_paths=np.array([str(p) for p in ds.source_paths], dtype=object),
    )
    print(f"Saved embeddings: {args.out}")
    print(f"Shape: X={X_emb.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
