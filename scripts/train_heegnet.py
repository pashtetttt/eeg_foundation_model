#!/usr/bin/env python3
"""Train HEEGNet on thesis EEG data (age4 or adolescence binary)."""

from __future__ import annotations

try:
    from _bootstrap import *  # noqa: F401,F403
except ModuleNotFoundError:
    from scripts._bootstrap import *  # noqa: F401,F403

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from eeg_thesis.eegpt_adapter import DEFAULT_19_CH
from eeg_thesis.eegpt_data import load_eegpt_dataset_from_edf, to_binary_adolescence


def _ensure_heegnet_importable(repo_root: Path) -> None:
    heegnet_root = repo_root / "heegnet" / "HEEGNet"
    if not heegnet_root.is_dir():
        raise FileNotFoundError(f"HEEGNet path not found: {heegnet_root}")
    p = str(heegnet_root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("HEEGNet config must be a YAML mapping (key-value object).")
    return data


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None, help="Path to YAML config file.")
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config(pre_args.config)

    ap = argparse.ArgumentParser(description="Train HEEGNet on local EDF data.")
    ap.add_argument("--config", type=Path, default=pre_args.config, help="Path to YAML config file.")
    ap.add_argument("--data-dir", type=Path, default=Path(cfg.get("data_dir", "data")))
    ap.add_argument("--eyes", type=str, default=cfg.get("eyes", "closed"), choices=["closed", "open"])
    ap.add_argument("--task", type=str, default=cfg.get("task", "age4"), choices=["age4", "adolescence_binary"])
    ap.add_argument("--max-per-group", type=int, default=cfg.get("max_per_group"), help="Cap per class while loading EDF.")
    ap.add_argument("--limit-total", type=int, default=cfg.get("limit_total"), help="Hard cap total examples for quick smoke test.")
    ap.add_argument("--target-sfreq", type=float, default=cfg.get("target_sfreq", 256.0))
    ap.add_argument("--window-seconds", type=float, default=cfg.get("window_seconds", 4.0))
    ap.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 32))
    ap.add_argument("--domains-per-batch", type=int, default=cfg.get("domains_per_batch", 4))
    ap.add_argument("--epochs", type=int, default=cfg.get("epochs", 40))
    ap.add_argument("--min-epochs", type=int, default=cfg.get("min_epochs", 5))
    ap.add_argument("--patience", type=int, default=cfg.get("patience", 8))
    ap.add_argument("--lr", type=float, default=cfg.get("lr", 1e-3))
    ap.add_argument("--weight-decay", type=float, default=cfg.get("weight_decay", 1e-4))
    ap.add_argument("--swd-weight", type=float, default=cfg.get("swd_weight", 0.01))
    ap.add_argument("--validation-size", type=float, default=cfg.get("validation_size", 0.2))
    ap.add_argument("--test-size", type=float, default=cfg.get("test_size", 0.2))
    ap.add_argument("--input-align", action="store_true", default=bool(cfg.get("input_align", False)), help="Use per-domain Euler alignment.")
    ap.add_argument("--no-domain-adaptation", action="store_true", default=bool(cfg.get("no_domain_adaptation", False)), help="Disable domain-specific BN.")
    ap.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    ap.add_argument("--device", type=str, default=cfg.get("device", "cuda"), choices=["cuda", "cpu"])
    return ap.parse_args()


def _make_domain_labels(paths: list[Path]) -> np.ndarray:
    """
    Domain id for HEEGNet batchnorm/sampler.
    For this project, we use folder-level grouping to keep per-domain sample
    counts high enough for small debug runs.
    """
    tags = [p.parent.name for p in paths]
    return LabelEncoder().fit_transform(tags)


def _prepare_splits(
    X: np.ndarray,
    y: np.ndarray,
    domains: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss_outer.split(X, y))
    X_tv, y_tv = X[train_val_idx], y[train_val_idx]
    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_rel, val_rel = next(sss_inner.split(X_tv, y_tv))
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]
    return train_idx, val_idx, test_idx


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_heegnet_importable(repo_root)

    from nets.callbacks import EarlyStopping, MomentumBatchNormScheduler  # type: ignore
    from nets.model import HEEGNet  # type: ignore
    from nets.trainer import Trainer  # type: ignore
    from nets.utils.data import DomainDataset, StratifiedDomainDataLoader  # type: ignore
    import nets.functionals as fn  # type: ignore
    import nets.batchnorm as bn  # type: ignore

    ds = load_eegpt_dataset_from_edf(
        data_dir=args.data_dir,
        eyes_condition=args.eyes,
        max_per_group=args.max_per_group,
        target_channels=DEFAULT_19_CH,
        target_sfreq=args.target_sfreq,
        window_seconds=args.window_seconds,
    )

    X = ds.X.astype(np.float64)
    y = ds.y.copy()
    class_names = ds.class_names
    if args.task == "adolescence_binary":
        y, class_names = to_binary_adolescence(ds.y, ds.class_names)
    domains = _make_domain_labels(ds.source_paths)

    if args.limit_total is not None and args.limit_total > 0 and len(y) > args.limit_total:
        keep_idx = np.arange(len(y))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(keep_idx)
        keep_idx = np.sort(keep_idx[: args.limit_total])
        X = X[keep_idx]
        y = y[keep_idx]
        domains = domains[keep_idx]

    if args.input_align:
        for d in np.unique(domains):
            X[domains == d] = fn.euler_align(X[domains == d])

    n_classes = len(np.unique(y))
    print(f"Task={args.task} classes={class_names} n_classes={n_classes}")
    print(f"Samples={len(y)} shape={X.shape} domains={len(np.unique(domains))}")

    train_idx, val_idx, test_idx = _prepare_splits(
        X, y, domains, args.test_size, args.validation_size, args.seed
    )

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y.astype(np.int64))
    d_t = torch.from_numpy(domains.astype(np.int64))

    ds_train = DomainDataset(X_t[train_idx], y_t[train_idx], d_t[train_idx])
    ds_val = DomainDataset(X_t[val_idx], y_t[val_idx], d_t[val_idx])
    ds_test = DomainDataset(X_t[test_idx], y_t[test_idx], d_t[test_idx])

    train_domains = d_t[train_idx].detach().cpu().numpy()
    _, per_domain_counts = np.unique(train_domains, return_counts=True)
    can_use_domain_loader = len(per_domain_counts) > 0 and int(per_domain_counts.min()) >= 2
    if can_use_domain_loader:
        domains_per_batch = min(args.domains_per_batch, len(torch.unique(d_t[train_idx])))
        loader_train = StratifiedDomainDataLoader(
            ds_train,
            args.batch_size,
            domains_per_batch=domains_per_batch,
            shuffle=True,
            drop_last=False,
        )
    else:
        print(
            "Warning: insufficient per-domain training samples for "
            "StratifiedDomainDataLoader; falling back to standard shuffled DataLoader."
        )
        loader_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_val = torch.utils.data.DataLoader(ds_val, batch_size=len(ds_val))
    loader_test = torch.utils.data.DataLoader(ds_test, batch_size=len(ds_test))

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using device: {device}")

    model = HEEGNet(
        chunk_size=X.shape[2],
        num_electrodes=X.shape[1],
        num_classes=n_classes,
        domains=torch.unique(d_t[train_idx]),
        domain_adaptation=not args.no_domain_adaptation,
        bnorm_dispersion=bn.BatchNormDispersion.SCALAR,
        device=device,
        dtype=torch.float64,
        lr=args.lr,
        weight_decay=args.weight_decay,
    ).to(device=device, dtype=torch.float64)

    es = EarlyStopping(metric="val_loss", higher_is_better=False, patience=args.patience, verbose=False)
    bn_sched = MomentumBatchNormScheduler(
        epochs=max(args.epochs - 1, 1),
        bs=args.batch_size,
        bs0=args.batch_size,
        tau0=0.85,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        min_epochs=args.min_epochs,
        callbacks=[es, bn_sched],
        loss=torch.nn.CrossEntropyLoss(),
        device=device,
        dtype=torch.float64,
        swd_weight=args.swd_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    trainer.fit(model, train_dataloader=loader_train, val_dataloader=loader_val)

    print(f"Best epoch (ES): {es.best_epoch}")
    test_res = trainer.test(model, dataloader=loader_test)
    print(f"Test metrics: {test_res}")

    # Detailed test report
    model.eval()
    with torch.no_grad():
        feats, y_true = next(iter(loader_test))
        feats["inputs"] = feats["inputs"].to(dtype=torch.float64, device=device)
        y_true = y_true.to(device=device)
        logits, _ = model(**feats)
        y_pred = torch.argmax(logits, dim=1)
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    print(classification_report(yt, yp, target_names=class_names, zero_division=0))
    print(confusion_matrix(yt, yp))


if __name__ == "__main__":
    main()
