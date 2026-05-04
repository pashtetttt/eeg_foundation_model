"""
Unified frozen encoders for HEEGNet and EEGPT (feature extraction only).

HEEGNet: loads ``heegnet_best.pt``-style checkpoints with ``meta`` + ``state_dict``.
EEGPT: uses ``EEGPTAdapter`` with pretrained weights; ``forward_features`` before the head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from eeg_thesis.eegpt_adapter import DEFAULT_19_CH, EEGPTAdapter


class ChannelAdapter(nn.Module):
    """1×1 Conv over channels: (B, C_in, T) -> (B, C_out, T). Frozen at init for eval extraction."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def load_eegpt_encoder(
    repo_root: Path,
    checkpoint: Path | None,
    *,
    device: torch.device,
    num_classes: int = 4,
) -> tuple[EEGPTAdapter, int]:
    """Return adapter (eval, frozen) and expected channel count."""
    adapter = EEGPTAdapter(
        repo_root=repo_root,
        checkpoint_path=checkpoint,
        channels=list(DEFAULT_19_CH),
        num_classes=num_classes,
        device=str(device),
    )
    if adapter.model is None:
        raise RuntimeError("EEGPT model failed to initialize")
    adapter.model.eval()
    for p in adapter.model.parameters():
        p.requires_grad_(False)
    return adapter, len(DEFAULT_19_CH)


def load_heegnet_encoder(
    repo_root: Path,
    checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> tuple[nn.Module, dict[str, Any], int]:
    """Load HEEGNet from thesis checkpoint; returns (model, meta, expected_num_electrodes)."""
    heegnet_root = repo_root / "heegnet" / "HEEGNet"
    if not heegnet_root.is_dir():
        raise FileNotFoundError(f"HEEGNet submodule missing: {heegnet_root}")
    import sys

    p = str(heegnet_root)
    if p not in sys.path:
        sys.path.insert(0, p)

    from nets.model import HEEGNet  # type: ignore

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta") or {}
    state = ckpt.get("state_dict")
    if state is None:
        state = ckpt

    n_cls = int(meta.get("n_classes", 4))
    chunk = int(meta.get("chunk_size", 1024))
    n_elec = int(meta.get("num_electrodes", 19))
    dom_ids = list(meta.get("domain_ids") or [0])
    domains = torch.tensor(sorted({int(d) for d in dom_ids}), dtype=torch.long)
    no_da = bool(meta.get("no_domain_adaptation", False))
    import nets.batchnorm as bn  # type: ignore

    model = HEEGNet(
        chunk_size=chunk,
        num_electrodes=n_elec,
        num_classes=n_cls,
        domains=domains,
        domain_adaptation=not no_da,
        bnorm_dispersion=bn.BatchNormDispersion.SCALAR,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta, n_elec


@torch.no_grad()
def eegpt_forward_embeddings(adapter: EEGPTAdapter, x_bct: torch.Tensor) -> torch.Tensor:
    """x_bct: (B, C, T) float32 on adapter device."""
    model = adapter.model
    if model is None:
        raise RuntimeError("EEGPT model missing")
    x_bct = x_bct.to(device=adapter.device, dtype=torch.float32)
    z = model.forward_features(x_bct)  # type: ignore[operator]
    if z.dim() > 2:
        z = z.reshape(z.shape[0], -1)
    return z


@torch.no_grad()
def heegnet_forward_embeddings(
    model: nn.Module,
    x_bct: torch.Tensor,
    domains_b: torch.Tensor,
    *,
    channel_adapter: ChannelAdapter | None,
) -> torch.Tensor:
    """x_bct on same device/dtype as model; domains_b long on same device."""
    if channel_adapter is not None:
        x_bct = channel_adapter(x_bct)
    _logits, feats = model(x_bct, domains_b)
    if feats.dim() > 2:
        feats = feats.reshape(feats.shape[0], -1)
    return feats


def heegnet_domain_batch(batch_size: int, meta: dict[str, Any], device: torch.device) -> torch.Tensor:
    dom_ids = list(meta.get("domain_ids") or [0])
    d0 = int(dom_ids[0])
    return torch.full((batch_size,), d0, device=device, dtype=torch.long)


def numpy_window_to_heegnet_tensor(
    data_ct: np.ndarray,
    *,
    expected_c: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = np.asarray(data_ct, dtype=np.float64)
    if x.shape[0] > expected_c:
        x = x[:expected_c]
    elif x.shape[0] < expected_c:
        x = np.pad(x, ((0, expected_c - x.shape[0]), (0, 0)), mode="constant")
    return torch.from_numpy(x).to(device=device, dtype=dtype).unsqueeze(0)


def build_channel_adapter_if_needed(in_c: int, out_c: int, device: torch.device, dtype: torch.dtype) -> ChannelAdapter | None:
    if in_c == out_c:
        return None
    m = ChannelAdapter(in_c, out_c).to(device=device, dtype=dtype)
    m.eval()
    return m
