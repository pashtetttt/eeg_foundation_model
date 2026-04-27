"""Dataset conversion helpers: existing EDF loaders -> EEGPT-ready tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from edf_loader import load_raw_edf_resilient
from eeg_experiment_shared import GROUPS, find_edf_files
from .eegpt_adapter import DEFAULT_19_CH, remap_channels_to_target, resample_window


@dataclass
class EEGPTDataset:
    X: np.ndarray  # [N, C, T]
    y: np.ndarray  # [N]
    class_names: list[str]
    source_paths: list[Path]
    channels: list[str]
    sfreq: float


def _normalize_per_channel(x_ct: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x_ct.mean(axis=1, keepdims=True)
    std = x_ct.std(axis=1, keepdims=True)
    return ((x_ct - mean) / (std + eps)).astype(np.float32)


def load_eegpt_dataset_from_edf(
    data_dir: Path,
    eyes_condition: str = "closed",
    max_per_group: int | None = None,
    groups: dict[str, str] | None = None,
    target_channels: Iterable[str] = DEFAULT_19_CH,
    target_sfreq: float = 256.0,
    window_seconds: float = 4.0,
    zscore_per_channel: bool = True,
) -> EEGPTDataset:
    """
    Build EEGPT-ready tensor dataset directly from EDF.

    Output tensor shape: [N, C, T] where C=len(target_channels), T=target_sfreq*window_seconds.
    """
    groups = groups or GROUPS
    class_names = list(groups.values())
    target_channels = [c.upper().strip(".") for c in target_channels]

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    path_list: list[Path] = []

    total_skipped = 0
    for group_folder, label_name in groups.items():
        paths = find_edf_files(data_dir, group_folder, max_per_group, eyes_condition=eyes_condition)
        if not paths:
            continue
        label_idx = class_names.index(label_name)
        print(f"  {label_name}: loading {len(paths)} files...")
        for p in paths:
            try:
                raw = load_raw_edf_resilient(p, preload=True, verbose=False)
                data = raw.get_data()
                if not np.all(np.isfinite(data)):
                    raise ValueError("Non-finite values in EEG")
                source_names = [ch.upper().strip(".") for ch in raw.ch_names]
                x = remap_channels_to_target(data, source_names, target_channels)
                x = resample_window(
                    x,
                    src_sfreq=float(raw.info["sfreq"]),
                    target_sfreq=target_sfreq,
                    window_seconds=window_seconds,
                )
                if zscore_per_channel:
                    x = _normalize_per_channel(x)
                X_list.append(x)
                y_list.append(label_idx)
                path_list.append(p)
            except Exception as e:
                total_skipped += 1
                print(f"    ⚠️ skipped {p.name}: {e}")

    if not X_list:
        raise FileNotFoundError(f"No usable EDF samples found under {data_dir}")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    print(f"Loaded EEG dataset tensor: N={X.shape[0]} C={X.shape[1]} T={X.shape[2]} skipped={total_skipped}")
    return EEGPTDataset(
        X=X,
        y=y,
        class_names=class_names,
        source_paths=path_list,
        channels=target_channels,
        sfreq=float(target_sfreq),
    )


def to_binary_adolescence(y_multiclass: np.ndarray, class_names: list[str]) -> tuple[np.ndarray, list[str]]:
    if "adolescence" not in class_names:
        raise ValueError(f"'adolescence' class not found in {class_names}")
    pos = class_names.index("adolescence")
    y_bin = (y_multiclass == pos).astype(np.int64)
    return y_bin, ["rest", "adolescence"]
