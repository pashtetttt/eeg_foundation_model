#!/usr/bin/env python3
"""Train EEGPT for 4-class age classification with validation, early stopping, and run artifacts."""

from __future__ import annotations

from _bootstrap import *  # noqa: F401,F403

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from eeg_experiment_shared import RANDOM_STATE, TEST_SIZE
from eeg_thesis.eegpt_adapter import DEFAULT_19_CH, EEGPTAdapter
from eeg_thesis.eegpt_data import load_eegpt_dataset_from_edf

from scripts.eegpt_run_artifacts import (
    plot_confusion_matrix,
    plot_training_curves,
    write_metrics_csv,
    write_readme_plot,
)

ROOT = Path(__file__).resolve().parents[1]


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


def _resolve(p: Path | str | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    idx = np.arange(len(y))
    if shuffle:
        rng = np.random.default_rng(RANDOM_STATE)
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        b = idx[i : i + batch_size]
        yield X[b], y[b]


def _train_epoch_multiclass(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    losses: list[float] = []
    for xb, yb in _iterate_minibatches(X, y, batch_size, shuffle=True):
        xt = torch.as_tensor(xb, dtype=torch.float32, device=device)
        yt = torch.as_tensor(yb, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xt)
        loss = criterion(logits, yt)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _eval_multiclass(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    criterion: torch.nn.Module,
) -> tuple[float, float, float, np.ndarray]:
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in _iterate_minibatches(X, y, batch_size, shuffle=False):
            xt = torch.as_tensor(xb, dtype=torch.float32, device=device)
            yt = torch.as_tensor(yb, dtype=torch.long, device=device)
            logits = model(xt)
            loss = criterion(logits, yt)
            losses.append(float(loss.item()))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_pred = np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.int64)
    vloss = float(np.mean(losses)) if losses else float("nan")
    bal = float(balanced_accuracy_score(y, y_pred)) if len(y_pred) else 0.0
    mf1 = float(f1_score(y, y_pred, average="macro", zero_division=0)) if len(y_pred) else 0.0
    return vloss, bal, mf1, y_pred


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config(pre_args.config)
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}

    ap = argparse.ArgumentParser(description="EEGPT age4 training (multiclass) with logging.")
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
    ap.add_argument("--val-size", type=float, default=float(train_cfg.get("val_size", 0.15)))
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=int(train_cfg.get("early_stopping_patience", 8)),
        help="Stop if val_loss does not improve for this many epochs (after min_epochs). 0 disables.",
    )
    ap.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=int(train_cfg.get("early_stopping_min_epochs", 3)),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metrics, checkpoint, plots. Default: results/eegpt_runs/<timestamp>_age4.",
    )
    ap.add_argument("--run-name", type=str, default=None, help="Optional tag for default output directory.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.run_name or (Path(args.config).stem if args.config else "age4")
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = ROOT / "results" / "eegpt_runs" / f"{ts}_age4_{tag}"
    else:
        out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = _resolve(args.checkpoint)

    ds = load_eegpt_dataset_from_edf(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        eyes_condition=args.eyes,
        max_per_group=args.max,
        target_channels=DEFAULT_19_CH,
        target_sfreq=args.target_sfreq,
        window_seconds=args.window_seconds,
    )

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(sss.split(np.arange(len(ds.y)), ds.y))
    X_trv, X_test = ds.X[train_val_idx], ds.X[test_idx]
    y_trv, y_test = ds.y[train_val_idx], ds.y[test_idx]

    val_size = float(args.val_size)
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be in (0, 1).")
    try:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=RANDOM_STATE)
        train_idx, val_idx = next(sss2.split(np.zeros(len(y_trv)), y_trv))
    except ValueError:
        sh = ShuffleSplit(n_splits=1, test_size=val_size, random_state=RANDOM_STATE)
        train_idx, val_idx = next(sh.split(X_trv))

    X_train, X_val = X_trv[train_idx], X_trv[val_idx]
    y_train, y_val = y_trv[train_idx], y_trv[val_idx]

    adapter = EEGPTAdapter(
        repo_root=ROOT,
        checkpoint_path=ckpt_path,
        channels=ds.channels,
        num_classes=4,
        target_sfreq=args.target_sfreq,
        window_seconds=args.window_seconds,
        device=args.device,
    )
    model = adapter.model
    assert model is not None

    device = torch.device(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    meta = {
        "task": "age4",
        "config_path": str(args.config) if args.config else None,
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "eyes": args.eyes,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "epochs_requested": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "val_size": val_size,
        "test_size": TEST_SIZE,
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_epochs": int(args.early_stopping_min_epochs),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "class_names": list(ds.class_names),
        "random_state": RANDOM_STATE,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    patience = int(args.early_stopping_patience)
    min_epochs = int(args.early_stopping_min_epochs)
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stagnant = 0
    metrics_rows: list[dict] = []
    stopped_early = False

    for epoch in range(1, int(args.epochs) + 1):
        tr_loss = _train_epoch_multiclass(
            model, X_train, y_train, batch_size=args.batch_size, device=device, criterion=criterion, optimizer=optimizer
        )
        val_loss, val_bal, val_f1, _ = _eval_multiclass(
            model, X_val, y_val, batch_size=args.batch_size, device=device, criterion=criterion
        )
        row = {
            "epoch": epoch,
            "train_loss": f"{tr_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_balanced_accuracy": f"{val_bal:.6f}",
            "val_macro_f1": f"{val_f1:.6f}",
        }
        metrics_rows.append(row)
        write_metrics_csv(out_dir / "metrics_per_epoch.csv", metrics_rows)
        print(
            f"epoch={epoch} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"val_bal_acc={val_bal:.4f} val_macro_f1={val_f1:.4f}"
        )

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            stagnant = 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stagnant += 1

        if patience > 0 and epoch >= min_epochs and stagnant >= patience:
            stopped_early = True
            print(f"Early stopping at epoch {epoch} (no val_loss improvement for {patience} epochs).")
            break

    if best_state is None and metrics_rows:
        best_epoch = int(metrics_rows[-1]["epoch"])
        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        print("Warning: no val_loss improvement tracked; saving last epoch weights as best.")

    if best_state is not None:
        model.load_state_dict(best_state)
        device_sd = torch.device(args.device)
        model.to(device_sd)
        print(f"Restored best weights from epoch {best_epoch} (val_loss={best_val_loss:.6f}).")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {**meta, "best_epoch": best_epoch, "best_val_loss": best_val_loss, "stopped_early": stopped_early},
        },
        out_dir / "eegpt_best.pt",
    )

    model.eval()
    _, _, _, y_pred = _eval_multiclass(
        model, X_test, y_test, batch_size=args.batch_size, device=device, criterion=criterion
    )
    acc = float((y_pred == y_test).mean()) if len(y_test) else 0.0
    bal = float(balanced_accuracy_score(y_test, y_pred)) if len(y_test) else 0.0
    mf1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0)) if len(y_test) else 0.0
    report = classification_report(y_test, y_pred, target_names=ds.class_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(4))).tolist()

    test_payload = {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "macro_f1": mf1,
        "confusion_matrix": cm,
        "classification_report": report,
        "epochs_trained": int(metrics_rows[-1]["epoch"]) if metrics_rows else 0,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if np.isfinite(best_val_loss) else None,
        "stopped_early": stopped_early,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(test_payload, indent=2), encoding="utf-8")

    plot_training_curves(out_dir / "metrics_per_epoch.csv", out_dir / "training_curves.png")
    plot_confusion_matrix(cm, list(ds.class_names), out_dir / "confusion_matrix_test.png", title="Test confusion matrix")
    write_readme_plot(out_dir)

    print(report)
    print(f"Wrote run directory: {out_dir}")


if __name__ == "__main__":
    main()
