#!/usr/bin/env python3
"""Train EEGPT for 4-class age classification."""

from __future__ import annotations

from _bootstrap import *  # noqa: F401,F403

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from eeg_thesis.eegpt_adapter import DEFAULT_19_CH, EEGPTAdapter
from eeg_thesis.eegpt_data import load_eegpt_dataset_from_edf
from eeg_experiment_shared import RANDOM_STATE, TEST_SIZE


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping.")
    return data


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config(pre_args.config)
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}

    ap = argparse.ArgumentParser(description="EEGPT age4 training (multiclass).")
    ap.add_argument("--config", type=Path, default=pre_args.config, help="Path to YAML config.")
    ap.add_argument("--data-dir", type=Path, default=Path(cfg.get("data_dir", "data")))
    ap.add_argument("--eyes", type=str, default=cfg.get("eyes", "closed"), choices=["closed", "open"])
    ap.add_argument("--max", type=int, default=cfg.get("max_per_group"))
    ap.add_argument("--checkpoint", type=Path, default=cfg.get("checkpoint"), help="Optional pretrained EEGPT checkpoint.")
    ap.add_argument("--epochs", type=int, default=train_cfg.get("epochs", 10))
    ap.add_argument("--batch-size", type=int, default=train_cfg.get("batch_size", 16))
    ap.add_argument("--lr", type=float, default=train_cfg.get("lr", 1e-4))
    ap.add_argument("--target-sfreq", type=float, default=cfg.get("target_sfreq", 256.0))
    ap.add_argument("--window-seconds", type=float, default=cfg.get("window_seconds", 4.0))
    ap.add_argument("--device", type=str, default=train_cfg.get("device", "cpu"))
    return ap.parse_args()


def _iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    idx = np.arange(len(y))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        b = idx[i : i + batch_size]
        yield X[b], y[b]


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

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(ds.X, ds.y))
    X_train, X_test = ds.X[train_idx], ds.X[test_idx]
    y_train, y_test = ds.y[train_idx], ds.y[test_idx]

    adapter = EEGPTAdapter(
        repo_root=Path("."),
        checkpoint_path=args.checkpoint,
        channels=ds.channels,
        num_classes=4,
        target_sfreq=args.target_sfreq,
        window_seconds=args.window_seconds,
        device=args.device,
    )
    model = adapter.model
    assert model is not None
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    device = torch.device(args.device)
    for epoch in range(1, args.epochs + 1):
        losses = []
        for xb, yb in _iterate_minibatches(X_train, y_train, args.batch_size, shuffle=True):
            xt = torch.as_tensor(xb, dtype=torch.float32, device=device)
            yt = torch.as_tensor(yb, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xt)
            loss = criterion(logits, yt)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"epoch={epoch} train_loss={np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        logits = adapter.predict_logits(X_test)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    print(f"acc={(y_pred == y_test).mean():.4f}")
    print(f"bal_acc={balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"macro_f1={f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(y_test, y_pred, target_names=ds.class_names, zero_division=0))


if __name__ == "__main__":
    main()
