#!/usr/bin/env python3
"""
Extract frozen REVE embeddings for each recording in subject mapping.

Outputs
-------
results/embeddings/embeddings_reve_{condition}_{cohort_name}.npz
  - embeddings: (N, D) float32
  - subject_ids: (N,) unicode
  - y: (N,) int64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REVE_SRC = ROOT / "reve" / "reve_eeg" / "src"
if str(REVE_SRC) not in sys.path:
    sys.path.insert(0, str(REVE_SRC))

import numpy as np
import pandas as pd
import torch

from edf_loader import load_raw_edf_resilient
from scripts.features.feature_utils import embeddings_cache_path
from scripts.utils.data_handling import load_yaml_config, resolve_data_dir


def _normalize_name(ch: str) -> str:
    c = ch.upper().strip().replace(".", "")
    c = c.replace("EEG ", "").replace("-REF", "").replace("REF-", "")
    return c


def _window_tensor_from_edf(
    edf_path: Path,
    *,
    target_sfreq: float,
    window_seconds: float,
    min_hz: float,
    max_hz: float,
    notch_hz: float | None,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns windows in shape (Nw, C, T) and channel names list of length C.
    """
    import mne

    raw = load_raw_edf_resilient(edf_path, preload=True, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) == 0:
        raise ValueError("no EEG channels")
    raw.pick(picks)
    raw.resample(target_sfreq)
    if min_hz > 0 or max_hz > 0:
        raw.filter(l_freq=min_hz, h_freq=max_hz, verbose=False)
    if notch_hz is not None and notch_hz > 0:
        raw.notch_filter(notch_hz, verbose=False)
    data_uv = raw.get_data(units="uV")
    if not np.all(np.isfinite(data_uv)):
        raise ValueError("non-finite EEG")
    names = [_normalize_name(ch) for ch in raw.ch_names]

    # z-score per channel
    mu = data_uv.mean(axis=1, keepdims=True)
    sd = data_uv.std(axis=1, keepdims=True)
    data_uv = (data_uv - mu) / (sd + 1e-6)

    win = int(round(target_sfreq * window_seconds))
    if win < 200:
        win = 200  # REVE patch size
    if data_uv.shape[1] < win:
        data_uv = np.pad(data_uv, ((0, 0), (0, win - data_uv.shape[1])), constant_values=0.0)
    n_win = data_uv.shape[1] // win
    data_uv = data_uv[:, : n_win * win]
    x = data_uv.reshape(data_uv.shape[0], n_win, win).transpose(1, 0, 2)
    return x.astype(np.float32, copy=False), names


@torch.no_grad()
def _encode_subject(
    windows: np.ndarray,
    names: list[str],
    *,
    encoder: torch.nn.Module,
    load_positions_fn,
    device: torch.device,
    batch_size: int,
    pooling: str,
) -> np.ndarray:
    pos_c3 = load_positions_fn(electrode_names=names).float()  # (C,3)
    feats = []
    for i in range(0, windows.shape[0], batch_size):
        xb = torch.from_numpy(windows[i : i + batch_size]).to(device=device, dtype=torch.float32)
        pb = pos_c3.unsqueeze(0).expand(xb.shape[0], -1, -1).to(device=device, dtype=torch.float32)
        tok = encoder(xb, pb)  # (B, L, D)
        if pooling == "token_mean":
            emb = tok.mean(dim=1)
        elif pooling == "token_max":
            emb = tok.amax(dim=1)
        else:
            raise ValueError(f"unknown pooling={pooling}")
        feats.append(emb.detach().cpu().numpy())
    all_w = np.concatenate(feats, axis=0)
    # recording embedding = mean over windows
    return np.mean(all_w, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract frozen REVE embeddings from EDF recordings.")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "reve_embedding_extraction.yaml")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-recordings", type=int, default=None, help="Optional cap for quick smoke tests.")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    _ = resolve_data_dir(cfg)
    results_dir = Path(cfg.get("results_dir", "results")).resolve()
    condition = str(cfg.get("eyes_condition", "closed"))
    cohort_name = str(cfg.get("cohort_name", "cohort"))
    model_id = str(cfg.get("model_id", "brain-bzh/reve-base"))
    batch_size = int(cfg.get("batch_size", 32))
    target_sfreq = float(cfg.get("target_sfreq", 200.0))
    window_seconds = float(cfg.get("window_seconds", 5.0))
    min_hz = float(cfg.get("filter_min_hz", 0.3))
    max_hz = float(cfg.get("filter_max_hz", 30.0))
    notch_hz = cfg.get("notch_hz", 50.0)
    notch_hz = float(notch_hz) if notch_hz is not None else None
    pooling = str(cfg.get("pooling", "token_mean"))
    max_recordings = cfg.get("max_recordings")
    if max_recordings is None:
        max_recordings = args.max_recordings
    max_recordings = int(max_recordings) if max_recordings is not None else None
    device_s = str(cfg.get("device", "cuda"))
    use_cuda = device_s == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    out_npz = embeddings_cache_path(results_dir, model="reve", condition=condition, cohort_name=cohort_name)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    if out_npz.is_file() and not args.force:
        print(f"Exists, skipping: {out_npz}")
        return

    mapping_rel = cfg.get("subject_mapping_csv", f"features/subject_mapping_{condition}_{cohort_name}.csv")
    mapping_path = (results_dir / str(mapping_rel)).resolve()
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing mapping: {mapping_path} (run compute_all_features first)")
    df = pd.read_csv(mapping_path)
    if max_recordings is not None and max_recordings > 0:
        df = df.head(max_recordings).copy()
        print(f"Smoke mode: processing first {len(df)} recordings")

    try:
        from models.encoder import REVE  # type: ignore
        from downstream_tasks.position_utils import load_positions  # type: ignore
    except Exception as e:
        raise ImportError(
            f"Failed REVE imports from {REVE_SRC}. Install dependencies from reve/reve_eeg/pyproject.toml "
            f"(notably einops/transformers). Original error: {e}"
        ) from e

    encoder, _cls = REVE.from_pretrained(model_id=model_id, cache_dir=str(cfg.get("hf_cache_dir", ".cache")))
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    vecs: list[np.ndarray] = []
    ids: list[str] = []
    ys: list[int] = []
    t0 = time.perf_counter()
    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        p = Path(getattr(row, "edf_path"))
        if not p.is_file():
            print(f"  skip missing: {p}")
            continue
        try:
            windows, names = _window_tensor_from_edf(
                p,
                target_sfreq=target_sfreq,
                window_seconds=window_seconds,
                min_hz=min_hz,
                max_hz=max_hz,
                notch_hz=notch_hz,
            )
            emb = _encode_subject(
                windows,
                names,
                encoder=encoder,
                load_positions_fn=load_positions,
                device=device,
                batch_size=batch_size,
                pooling=pooling,
            )
            vecs.append(emb)
            ids.append(str(getattr(row, "subject_id")))
            ys.append(int(getattr(row, "y", getattr(row, "label_idx", 0))))
            if i % 20 == 0 or i == total:
                elapsed = time.perf_counter() - t0
                rate = i / max(elapsed, 1e-9)
                eta = (total - i) / max(rate, 1e-9)
                print(f"[extract_reve] {i}/{total} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m ok={len(vecs)}")
        except Exception as e:
            print(f"  skip {getattr(row, 'subject_id')}: {e}")

    if not vecs:
        raise RuntimeError("No embeddings extracted.")
    E = np.stack(vecs, axis=0).astype(np.float32)
    sid = np.asarray(ids, dtype=str)
    y = np.asarray(ys, dtype=np.int64)
    np.savez_compressed(out_npz, embeddings=E, subject_ids=sid, y=y)
    print(f"Saved {out_npz} shape={E.shape}")


if __name__ == "__main__":
    main()

