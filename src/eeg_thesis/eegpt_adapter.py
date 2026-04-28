"""Adapter utilities for using EEGPT models in this repository."""

from __future__ import annotations

import pickle
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch


# EEGPT default 19-channel montage used in this thesis project.
DEFAULT_19_CH = [
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
]


def _ensure_eegpt_importable(repo_root: Path) -> None:
    eegpt_root = repo_root / "eegpt" / "EEGPT" / "downstream"
    if not eegpt_root.is_dir():
        raise FileNotFoundError(f"EEGPT downstream path not found: {eegpt_root}")
    path_str = str(eegpt_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def remap_channels_to_target(
    data: np.ndarray,
    source_names: Iterable[str],
    target_names: Iterable[str] = DEFAULT_19_CH,
) -> np.ndarray:
    """
    Reorder/pad channels from source -> target.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_channels, n_times).
    source_names : Iterable[str]
        Channel names corresponding to rows in `data`.
    target_names : Iterable[str]
        Desired output order.
    """
    source_names = [s.upper().strip(".") for s in source_names]
    src_to_idx = {name: i for i, name in enumerate(source_names)}
    target_names = [t.upper().strip(".") for t in target_names]

    out = np.zeros((len(target_names), data.shape[1]), dtype=np.float32)
    for i, name in enumerate(target_names):
        if name in src_to_idx:
            out[i] = data[src_to_idx[name]]
    return out


def _resample_numpy_1d(signal: np.ndarray, new_len: int) -> np.ndarray:
    """Simple linear interpolation resampling for 1D signal."""
    old_len = signal.shape[-1]
    if old_len == new_len:
        return signal.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, old_len, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, new_len, dtype=np.float64)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def resample_window(
    data: np.ndarray,
    src_sfreq: float,
    target_sfreq: float = 256.0,
    window_seconds: float = 4.0,
) -> np.ndarray:
    """
    Resample signal and enforce fixed window length.

    - resample each channel from `src_sfreq` to `target_sfreq`
    - crop or right-pad to `window_seconds * target_sfreq`
    """
    data = np.asarray(data, dtype=np.float32)
    target_len = int(round(window_seconds * target_sfreq))
    if target_len <= 0:
        raise ValueError("target_len must be positive")

    # Resample first if needed.
    if abs(src_sfreq - target_sfreq) > 1e-6:
        new_len = int(round(data.shape[1] * (target_sfreq / src_sfreq)))
        resampled = np.zeros((data.shape[0], new_len), dtype=np.float32)
        for ch in range(data.shape[0]):
            resampled[ch] = _resample_numpy_1d(data[ch], new_len)
        data = resampled

    # Fixed-length crop/pad.
    if data.shape[1] >= target_len:
        return data[:, :target_len].astype(np.float32, copy=False)

    out = np.zeros((data.shape[0], target_len), dtype=np.float32)
    out[:, : data.shape[1]] = data
    return out


def _extract_state_dict(checkpoint_obj: dict) -> dict:
    # Common checkpoint formats:
    # - {"state_dict": ...}
    # - {"model": ...}
    # - raw state_dict
    if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
        return checkpoint_obj["state_dict"]
    if "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
        return checkpoint_obj["model"]
    return checkpoint_obj


def _strip_prefixes(state_dict: dict, prefixes: tuple[str, ...]) -> dict:
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _remap_pretrain_keys_for_finetune(state_dict: dict) -> dict:
    """
    Map common pretraining checkpoint keys to finetune model keys.
    Example: encoder.* -> target_encoder.*
    """
    remapped = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("encoder."):
            nk = "target_encoder." + nk[len("encoder.") :]
        remapped[nk] = v
    return remapped


class EEGPTAdapter:
    """
    Thin wrapper around EEGPTClassifier for:
    - checkpoint loading
    - standardized prediction API
    - embedding extraction API
    """

    def __init__(
        self,
        repo_root: Path,
        checkpoint_path: Path | None,
        channels: list[str] | None = None,
        num_classes: int = 4,
        use_predictor: bool = True,
        use_chan_conv: bool = True,
        target_sfreq: float = 256.0,
        window_seconds: float = 4.0,
        device: str = "cpu",
    ) -> None:
        self.repo_root = Path(repo_root)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.channels = [c.upper().strip(".") for c in (channels or DEFAULT_19_CH)]
        self.num_classes = int(num_classes)
        self.target_sfreq = float(target_sfreq)
        self.window_seconds = float(window_seconds)
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None

        _ensure_eegpt_importable(self.repo_root)
        from Modules.models.EEGPT_mcae_finetune import EEGPTClassifier  # type: ignore

        time_len = int(round(self.target_sfreq * self.window_seconds))
        self.model = EEGPTClassifier(
            num_classes=self.num_classes,
            in_channels=len(self.channels),
            img_size=[len(self.channels), time_len],
            use_channels_names=self.channels,
            use_chan_conv=use_chan_conv,
            use_predictor=use_predictor,
            desired_time_len=time_len,
        ).to(self.device)

        if self.checkpoint_path:
            self.load_checkpoint(self.checkpoint_path)

    def load_checkpoint(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            ckpt = torch.load(path, map_location="cpu")
        except pickle.UnpicklingError as e:
            # PyTorch 2.6+ defaults to weights_only=True. Some older checkpoints
            # require full unpickling and fail unless weights_only=False.
            print(
                "[EEGPTAdapter] weights-only checkpoint load failed; retrying with "
                "weights_only=False. Use only with trusted checkpoints."
            )
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                # Compatibility fallback for older torch versions.
                ckpt = torch.load(path, map_location="cpu")
        state = _extract_state_dict(ckpt)
        state = _strip_prefixes(state, prefixes=("model.", "module."))
        state = _remap_pretrain_keys_for_finetune(state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[EEGPTAdapter] Missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        if unexpected:
            print(
                f"[EEGPTAdapter] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}"
            )

    def prepare_sample(self, data_ct: np.ndarray, source_names: Iterable[str], src_sfreq: float) -> np.ndarray:
        x = remap_channels_to_target(data_ct, source_names, self.channels)
        x = resample_window(x, src_sfreq, self.target_sfreq, self.window_seconds)
        return x

    @torch.no_grad()
    def predict_logits(self, batch_ct: np.ndarray) -> torch.Tensor:
        """
        batch_ct: np.ndarray [B, C, T]
        returns logits: torch.Tensor [B, num_classes] or [B, 1]
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        self.model.eval()
        x = torch.as_tensor(batch_ct, dtype=torch.float32, device=self.device)
        return self.model(x)

    @torch.no_grad()
    def extract_embeddings(self, batch_ct: np.ndarray) -> torch.Tensor:
        """
        Returns latent features from forward_features.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        self.model.eval()
        x = torch.as_tensor(batch_ct, dtype=torch.float32, device=self.device)
        return self.model.forward_features(x)  # type: ignore[attr-defined]
