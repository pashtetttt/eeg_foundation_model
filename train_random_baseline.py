"""
Random classifier baseline for EEG age-group classification.

Trains a DummyClassifier with random strategy to establish baseline performance.
This provides a lower bound for comparison with RF and SVM models.

Classes (4): preschooler, primary, teenager, adolescence.

Usage:
  python train_random_baseline.py              # use all data
  python train_random_baseline.py --max 50     # use up to 50 files per class
  python train_random_baseline.py --eyes open  # use open-eyes data
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
from sklearn.preprocessing import StandardScaler

from eeg_features import extract_all_features, feature_description

# --- Config ---
BASE = Path(".")
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"

GROUPS = {
    "дошкольники (124)": "preschooler",
    "младшие школьники (177)": "primary",
    "подростки (160)": "teenager",
    "юность(273)": "adolescence",
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
        return []
    paths = sorted(folder.rglob("*.edf"))
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
                raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
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
    parser = argparse.ArgumentParser(description="Random classifier baseline for EEG age classification.")
    parser.add_argument("--max", type=int, default=None, help="Max samples per class (default: all)")
    parser.add_argument("--eyes", type=str, default="closed", choices=["closed", "open"],
                        help="Eyes condition: 'closed' (default) or 'open'")
    args = parser.parse_args()

    experiment_start = datetime.now()
    log_lines = []

    def log(s: str) -> None:
        log_lines.append(s)
        print(s)

    log("=" * 60)
    log("Model: Random Classifier (DummyClassifier)")
    log("Strategy: stratified (random with class priors)")
    log(f"Experiment started: {experiment_start.isoformat()}")
    log(f"Eyes condition: {args.eyes}")
    log("=" * 60)

    print("Loading data and extracting features...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y, class_names = load_features_and_labels(DATA_DIR, max_per_group=args.max, eyes_condition=args.eyes)
    log("")
    log(f"Total samples: {X.shape[0]}, features: {X.shape[1]}")
    log(feature_description(eyes_condition=args.eyes))
    log("  (N_CHANNELS=19 = standard 10-20; actual channels per file vary—we use first file's order and pad to 19.)")
    for i, name in enumerate(class_names):
        log(f"  {name}: {(y == i).sum()}")

    # Balanced subset (same as RF/SVM for fair comparison)
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

    # Random classifier (stratified = random with class priors)
    model = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    log(f"Train time: {train_time:.3f} s (random, no actual training)")

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

    # Theoretical baseline for balanced 4-class: 25% accuracy
    log("")
    log("--- Theoretical baseline ---")
    log(f"Random chance (uniform): 25.0% accuracy (1/{len(class_names)} classes)")
    log(f"Random chance (stratified): ~{100/len(class_names):.1f}% accuracy (varies by run)")

    experiment_end = datetime.now()
    log("")
    log("=" * 60)
    log(f"Experiment ended: {experiment_end.isoformat()}")
    log("=" * 60)

    # Save to file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = experiment_start.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_random_baseline_{timestamp}.txt"
    out_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
