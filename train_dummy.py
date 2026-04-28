"""
Dummy classifier baseline for EEG age-group classification.

Predicts the most frequent class (majority class) in the training set.
Useful baseline to compare with RF and other models: if your model doesn't
beat this, it's not learning useful patterns.

Same CLI as train_rf.py: --max, --eyes, --features for direct comparison.

Usage:
  python train_dummy.py
  python train_dummy.py --max 50
  python train_dummy.py --eyes open --features alpha
"""

import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*does not conform to MNE naming conventions.*")
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict

from edf_loader import load_raw_edf_resilient
from eeg_features import extract_all_features, feature_description, get_feature_indices_by_category

# --- Config (same as train_rf.py) ---
BASE = Path(".")
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"

GROUPS = {
    "preschooler": "preschooler",
    "primary": "primary",
    "teenager": "teenager",
    "adolescence": "adolescence",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

CLOSED_EYES_SUBSTRINGS = ("_zg", "_ZG", "_зг", "_ЗГ")
OPEN_EYES_SUBSTRINGS = ("_og", "_OG", "_ог", "_ОГ")


def find_edf_files(data_dir: Path, group_folder: str, max_per_group: int | None, eyes_condition: str = "closed") -> list[Path]:
    """Find .edf files in group folder, optionally limit count."""
    folder = data_dir / group_folder
    if not folder.exists():
        # Support alternate dataset folder naming by prefix matching.
        prefix = group_folder.split("(")[0].strip()
        candidates = [p for p in data_dir.iterdir() if p.is_dir() and p.name.strip().startswith(prefix)]
        if len(candidates) == 1:
            folder = candidates[0]
        else:
            return []
    paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".edf")
    if eyes_condition == "closed":
        paths = [p for p in paths if any(s in p.name for s in CLOSED_EYES_SUBSTRINGS)]
    else:
        paths = [p for p in paths if any(s in p.name for s in OPEN_EYES_SUBSTRINGS)]
    if max_per_group is not None:
        paths = paths[:max_per_group]
    return paths


def load_features_and_labels(data_dir: Path, max_per_group: int | None, eyes_condition: str = "closed"):
    """Load features and labels from EDF files."""
    import mne

    X_list, y_list = [], []
    class_names = list(GROUPS.values())
    canonical_ch_names = None
    total_loaded = 0
    total_skipped = 0

    for group_folder, label_name in GROUPS.items():
        paths = find_edf_files(data_dir, group_folder, max_per_group, eyes_condition)
        n_paths = len(paths)
        if n_paths == 0:
            print(f"  ⚠️  No .edf files found for {label_name} in {data_dir / group_folder}")
            continue
        print(f"  {label_name}: loading {n_paths} files ...", end="", flush=True)
        label_idx = class_names.index(label_name)
        group_loaded = 0
        for i, p in enumerate(paths):
            if (i + 1) % 200 == 0 or i == n_paths - 1:
                print(f" {i + 1}/{n_paths}", end="", flush=True)
            try:
                raw = load_raw_edf_resilient(p, preload=True, verbose=False)
                data_check = raw.get_data()
                if not np.all(np.isfinite(data_check)):
                    raise ValueError("Non-finite values in raw data (NaN/Inf detected)")
                out = extract_all_features(raw, canonical_ch_names, eyes_condition=eyes_condition)
                if canonical_ch_names is None:
                    feat, canonical_ch_names = out
                else:
                    feat = out
                X_list.append(feat)
                y_list.append(label_idx)
                group_loaded += 1
            except Exception as e:
                print(f"\n    ⚠️  Failed to load {p.name}: {e}")
                total_skipped += 1
        total_loaded += group_loaded
        print(f" -> {group_loaded} ok")

    if not X_list:
        raise FileNotFoundError(
            f"No .edf files found under {data_dir}. Check folder structure and filenames."
        )

    print(f"  Total: {total_loaded} loaded, {total_skipped} skipped")
    return np.array(X_list), np.array(y_list), class_names


def main():
    parser = argparse.ArgumentParser(description="Dummy classifier (most frequent class) baseline for EEG age classification.")
    parser.add_argument("--max", type=int, default=None, help="Max samples per class (default: all)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root directory (default: data)")
    parser.add_argument("--eyes", type=str, default="closed", choices=["closed", "open"],
                        help="Eyes condition: 'closed' (default) or 'open'")
    parser.add_argument("--features", type=str, default="all",
                        help="Feature subset: 'all' (default), 'alpha', 'band_power', etc. (same as train_rf)")
    args = parser.parse_args()

    experiment_start = datetime.now()
    log_lines = []

    def log(s: str) -> None:
        log_lines.append(s)
        print(s)

    log("=" * 60)
    log("Model: Dummy Classifier (most_frequent)")
    log("Strategy: predicts the most popular class in training set")
    log(f"Experiment started: {experiment_start.isoformat()}")
    log(f"Eyes condition: {args.eyes}")
    log(f"Feature subset: {args.features}")
    log("=" * 60)

    print("Loading data and extracting features...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y, class_names = load_features_and_labels(args.data_dir, max_per_group=args.max, eyes_condition=args.eyes)

    # Select feature subset (same as train_rf)
    feature_indices = get_feature_indices_by_category(eyes_condition=args.eyes)
    if args.features not in feature_indices:
        valid = list(feature_indices.keys())
        raise ValueError(f"Unknown feature subset '{args.features}'. Valid options: {valid}")

    selected_idx = feature_indices[args.features]
    X = X[:, selected_idx]

    log("")
    log(f"Total samples: {X.shape[0]}, features: {X.shape[1]} (selected: {args.features})")
    log(feature_description(eyes_condition=args.eyes))
    log("  (N_CHANNELS=19 = standard 10-20; actual channels per file vary—we use first file's order and pad to 19.)")
    for i, name in enumerate(class_names):
        log(f"  {name}: {(y == i).sum()}")

    # Balanced subset (same as train_rf for fair comparison)
    rng = np.random.default_rng(RANDOM_STATE)
    n_per_class = np.array([(y == i).sum() for i in range(len(class_names))])
    n_balance = int(n_per_class.min())
    balanced_idx = []
    for i in range(len(class_names)):
        idx = np.where(y == i)[0]
        balanced_idx.append(rng.choice(idx, size=n_balance, replace=False))
    balanced_idx = np.concatenate(balanced_idx)
    rng.shuffle(balanced_idx)
    X_bal = X[balanced_idx]
    y_bal = y[balanced_idx]
    log(f"Balanced subset: {len(y_bal)} samples ({n_balance} per class) for training.")

    # Train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X_bal, y_bal))
    X_train, X_test = X_bal[train_idx], X_bal[test_idx]
    y_train, y_test = y_bal[train_idx], y_bal[test_idx]

    # Dummy: most frequent class (ignores X entirely)
    model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # With balanced data there may be a tie; sklearn picks one. Fallback: compute from y_train.
    unique, counts = np.unique(y_train, return_counts=True)
    most_freq_idx = int(unique[np.argmax(counts)])
    most_freq_name = class_names[most_freq_idx]
    log(f"Train time: {train_time:.3f} s (no actual training)")
    log(f"Most frequent class: {most_freq_name}")

    y_pred = model.predict(X_test)
    log("")
    log("--- Test set results (balanced hold-out) ---")
    log(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    log("Confusion matrix:")
    log(str(confusion_matrix(y_test, y_pred)))

    log("")
    log("--- Stratified 5-fold CV (on full dataset) ---")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    y_cv = cross_val_predict(model, X, y, cv=cv)
    cv_time = time.perf_counter() - t0
    log(f"CV time: {cv_time:.3f} s")
    log(classification_report(y, y_cv, target_names=class_names, zero_division=0))
    log("Confusion matrix (CV):")
    log(str(confusion_matrix(y, y_cv)))

    log("")
    log("--- Theoretical baseline ---")
    log(f"Most-frequent baseline accuracy ≈ class frequency of '{most_freq_name}' in data")
    log(f"Random chance (uniform): 25.0% (1/4 classes)")

    experiment_end = datetime.now()
    log("")
    log("=" * 60)
    log(f"Experiment ended: {experiment_end.isoformat()}")
    log("=" * 60)

    # Save to file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = experiment_start.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_dummy_{timestamp}.txt"
    out_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
