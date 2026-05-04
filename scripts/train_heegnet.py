#!/usr/bin/env python3
"""Train HEEGNet on thesis EEG data (age4 or adolescence binary)."""

from __future__ import annotations

try:
    from _bootstrap import *  # noqa: F401,F403
except ModuleNotFoundError:
    from scripts._bootstrap import *  # noqa: F401,F403

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
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
    _sm = cfg.get("sampling_method", "none")
    if _sm is None:
        _sm = "none"
    ap.add_argument(
        "--sampling-method",
        type=str,
        default=str(_sm).lower().strip(),
        choices=["none", "smote"],
        help="Train-only resampling: none (default) or smote (flatten C×T, imblearn SMOTE). From config key sampling_method.",
    )
    _kn = cfg.get("smote_k_neighbors", 5)
    ap.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=int(5 if _kn is None else _kn),
        help="SMOTE k_neighbors (capped by minority count − 1). Config: smote_k_neighbors.",
    )
    ap.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    ap.add_argument("--device", type=str, default=cfg.get("device", "cuda"), choices=["cuda", "cpu"])
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metrics CSV, checkpoint, and test summary. Default: results/heegnet_runs/<timestamp>_<task>.",
    )
    ap.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional tag appended to the default output directory name.",
    )
    return ap.parse_args()


def _make_domain_labels(paths: list[Path]) -> np.ndarray:
    """
    Domain id for HEEGNet batchnorm/sampler.
    For this project, we use folder-level grouping to keep per-domain sample
    counts high enough for small debug runs.
    """
    tags = [p.parent.name for p in paths]
    return LabelEncoder().fit_transform(tags)


def _records_per_epoch(records: list[dict]) -> list[dict]:
    """Trainer logs trn and val as separate dicts per epoch; merge into one row per epoch."""
    merged: dict[int, dict] = {}
    order: list[int] = []
    for rec in records:
        ep = rec.get("epoch")
        if ep is None:
            continue
        if ep not in merged:
            merged[ep] = {"epoch": int(ep)}
            order.append(int(ep))
        merged[ep].update(rec)
    return [merged[ep] for ep in order]


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("epoch\n", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()}, key=lambda k: (k != "epoch", k))
    if "epoch" in keys:
        keys.remove("epoch")
        keys = ["epoch"] + keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _smote_oversample_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    domains_train: np.ndarray,
    *,
    k_neighbors: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Oversample the training set with SMOTE on flattened windows (N, C*T).

    Synthetic rows get the domain label of their nearest neighbor in the original
    training set (same feature space), so domain-specific BN stays consistent.
    """
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "sampling_method=smote requires imbalanced-learn compatible with your scikit-learn. "
            "Install e.g. pip install 'imbalanced-learn==0.11.0' (matches NGC/old sklearn; "
            "avoid 0.12+ which needs sklearn.utils._metadata_requests). See train_heegnet_zhores.slurm."
        ) from e

    y_int = y_train.astype(np.int64, copy=False)
    counts = np.bincount(y_int, minlength=int(y_int.max()) + 1)
    pos = counts[counts > 0]
    min_count = int(pos.min()) if len(pos) else 0
    k_eff = min(int(k_neighbors), min_count - 1)
    if k_eff < 1:
        print(
            f"Warning: SMOTE skipped (smallest class count {min_count}, "
            f"need >= k_neighbors+1 for k_neighbors={k_neighbors}). Using original train set."
        )
        return X_train, y_train, domains_train

    flat = X_train.reshape(len(X_train), -1)
    sm = SMOTE(random_state=seed, k_neighbors=k_eff)
    flat_new, y_new = sm.fit_resample(flat, y_int)
    X_new = flat_new.reshape(-1, X_train.shape[1], X_train.shape[2]).astype(np.float64, copy=False)
    y_new = y_new.astype(np.int64, copy=False)

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(flat)
    neigh_idx = nn.kneighbors(flat_new, return_distance=False)[:, 0]
    dom_new = domains_train[neigh_idx].astype(np.int64, copy=False)

    return X_new, y_new, dom_new


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
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_default = repo_root / "results" / "heegnet_runs" / f"heegnet_{stamp}_{args.task}"
    if args.run_name:
        out_default = out_default.parent / f"{out_default.name}_{args.run_name}"
    output_dir = args.output_dir if args.output_dir is not None else out_default
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts directory: {output_dir}")
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
    print(f"Sampling method (train only): {args.sampling_method}")

    train_idx, val_idx, test_idx = _prepare_splits(
        X, y, domains, args.test_size, args.validation_size, args.seed
    )

    X_train_np = X[train_idx]
    y_train_np = y[train_idx].astype(np.int64, copy=False)
    dom_train_np = domains[train_idx].astype(np.int64, copy=False)

    if args.sampling_method == "smote":
        print("Applying SMOTE on training split only (flattened C×T features).")
        print("Train class counts (before SMOTE):", {class_names[i]: int((y_train_np == i).sum()) for i in range(n_classes)})
        X_train_np, y_train_np, dom_train_np = _smote_oversample_train(
            X_train_np,
            y_train_np,
            dom_train_np,
            k_neighbors=args.smote_k_neighbors,
            seed=args.seed,
        )
        print("Train class counts (after SMOTE):", {class_names[i]: int((y_train_np == i).sum()) for i in range(n_classes)})
        print(f"Train size after SMOTE: {len(y_train_np)} (was {len(train_idx)})")

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y.astype(np.int64))
    d_t = torch.from_numpy(domains.astype(np.int64))

    ds_train = DomainDataset(
        torch.from_numpy(X_train_np),
        torch.from_numpy(y_train_np),
        torch.from_numpy(dom_train_np),
    )
    ds_val = DomainDataset(X_t[val_idx], y_t[val_idx], d_t[val_idx])
    ds_test = DomainDataset(X_t[test_idx], y_t[test_idx], d_t[test_idx])

    train_domains = dom_train_np
    _, per_domain_counts = np.unique(train_domains, return_counts=True)
    can_use_domain_loader = len(per_domain_counts) > 0 and int(per_domain_counts.min()) >= 2
    if can_use_domain_loader:
        domains_per_batch = min(args.domains_per_batch, len(np.unique(train_domains)))
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
        # Register BN domains for all samples so validation/test domain ids exist.
        domains=torch.unique(d_t),
        domain_adaptation=not args.no_domain_adaptation,
        bnorm_dispersion=bn.BatchNormDispersion.SCALAR,
        device=device,
        dtype=torch.float64,
        lr=args.lr,
        weight_decay=args.weight_decay,
    ).to(device=device, dtype=torch.float64)

    es = EarlyStopping(metric="val_loss", higher_is_better=False, patience=args.patience, verbose=False)
    callbacks = [es]
    if args.epochs > 1:
        bn_sched = MomentumBatchNormScheduler(
            epochs=max(args.epochs - 1, 1),
            bs=args.batch_size,
            bs0=args.batch_size,
            tau0=0.85,
        )
        callbacks.append(bn_sched)
    else:
        print("Info: skipping MomentumBatchNormScheduler because epochs <= 1")

    trainer = Trainer(
        max_epochs=args.epochs,
        min_epochs=args.min_epochs,
        callbacks=callbacks,
        loss=torch.nn.CrossEntropyLoss(),
        device=device,
        dtype=torch.float64,
        swd_weight=args.swd_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    trainer.fit(model, train_dataloader=loader_train, val_dataloader=loader_val)

    history_rows = _records_per_epoch(trainer.records)
    metrics_csv = output_dir / "metrics_per_epoch.csv"
    _write_metrics_csv(metrics_csv, history_rows)
    print(f"Wrote training history: {metrics_csv}")

    ckpt_path = output_dir / "heegnet_best.pt"
    meta = {
        "best_epoch": int(es.best_epoch),
        "early_stop_metric": "val_loss",
        "task": args.task,
        "class_names": list(class_names),
        "n_classes": int(n_classes),
        "chunk_size": int(X.shape[2]),
        "num_electrodes": int(X.shape[1]),
        "domain_ids": torch.unique(d_t).cpu().tolist(),
        "seed": int(args.seed),
        "config_path": str(args.config) if args.config else None,
        "no_domain_adaptation": bool(args.no_domain_adaptation),
        "sampling_method": str(args.sampling_method),
        "smote_k_neighbors": int(args.smote_k_neighbors),
    }
    torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt_path)
    print(f"Wrote checkpoint (early-stopping best weights): {ckpt_path}")

    print(f"Best epoch (ES): {es.best_epoch}")
    test_res = trainer.test(model, dataloader=loader_test)
    print(f"Test metrics: {test_res}")

    test_json = output_dir / "test_metrics.json"
    test_json.write_text(json.dumps({"metrics": test_res, "meta": meta}, indent=2), encoding="utf-8")
    print(f"Wrote test summary: {test_json}")

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

    readme = output_dir / "README_plot.txt"
    readme.write_text(
        f"metrics_per_epoch.csv — one row per epoch (train + val columns).\n"
        f"heegnet_best.pt — state_dict + meta (best early-stopping weights).\n\n"
        f"Plot learning curves (from repo root):\n"
        f'  python scripts/plot_heegnet_curves.py --metrics-csv "{metrics_csv}"\n',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
