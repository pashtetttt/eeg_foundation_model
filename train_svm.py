"""
Train SVM to classify EEG into 4 age groups using processed .fif data.

Classes (4 only): preschooler, primary, teenager, adolescence.
Labels come only from the top-level group folder; age subfolders (e.g. 6(38), 23(42))
are ignored for labeling — all files in a group get that group's class.
Train/test split is stratified so class proportions are preserved.

Run after process_edf.py --batch. From thesis folder: python train_svm.py

Usage:
  python train_svm.py              # use all processed files
  python train_svm.py --max 50     # use up to 50 files per class (faster)
"""

import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path

import mne
import numpy as np

# Suppress noisy MNE filename convention warnings
warnings.filterwarnings("ignore", message=".*does not conform to MNE naming conventions.*")
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from eeg_features import N_CHANNELS, extract_all_features, feature_description

# --- Config ---
BASE = Path(".")
PROCESSED_DIR = BASE / "processed"
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

# Only closed-eyes recordings (zg/ЗГ/ог/ОГ in filename) to avoid same subject in train and test with different conditions.
CLOSED_EYES_SUBSTRINGS = ("_zg_", "_ZG_", "_зг_", "_ЗГ_", "_ог_", "_ОГ_")


def find_processed_fif(processed_dir: Path, group_folder: str, max_per_group: int | None) -> list[Path]:
    """List *_processed.fif under processed_dir/group_folder (closed-eyes only)."""
    folder = processed_dir / group_folder
    if not folder.exists():
        return []
    paths = sorted(folder.rglob("*_processed.fif"))
    paths = [p for p in paths if any(s in p.name for s in CLOSED_EYES_SUBSTRINGS)]
    if max_per_group is not None:
        paths = paths[:max_per_group]
    return paths


def load_features_and_labels(processed_dir: Path, max_per_group: int | None):
    """Load all processed .fif, extract band-power features, build X and y."""
    X_list = []
    y_list = []
    class_names = list(GROUPS.values())
    total_loaded = 0
    total_skipped = 0

    canonical_ch_names = None
    for group_folder, label_name in GROUPS.items():
        paths = find_processed_fif(processed_dir, group_folder, max_per_group)
        n_paths = len(paths)
        if n_paths == 0:
            continue
        print(f"  {label_name}: loading {n_paths} files ...", end="", flush=True)
        label_idx = class_names.index(label_name)
        group_loaded = 0
        for i, p in enumerate(paths):
            if (i + 1) % 200 == 0 or i == n_paths - 1:
                print(f" {i + 1}/{n_paths}", end="", flush=True)
            try:
                raw = mne.io.read_raw_fif(p, preload=True, verbose=False)
                out = extract_all_features(raw, canonical_ch_names)
                if canonical_ch_names is None:
                    feat, canonical_ch_names = out
                else:
                    feat = out
                X_list.append(feat)
                y_list.append(label_idx)
                group_loaded += 1
            except Exception:
                total_skipped += 1
        total_loaded += group_loaded
        print(f" -> {group_loaded} ok")

    if not X_list:
        raise FileNotFoundError(
            f"No processed .fif found under {processed_dir}. Run process_edf.py --batch first."
        )

    print(f"  Total: {total_loaded} loaded, {total_skipped} skipped")
    print("  (Folder names like '(124)' = subject count; we have more files = multiple recordings per subject, e.g. og/zg.)")
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, class_names


def main():
    parser = argparse.ArgumentParser(description="Train SVM on processed EEG (4 age classes).")
    parser.add_argument("--max", type=int, default=None, help="Max samples per class (default: all)")
    args = parser.parse_args()

    experiment_start = datetime.now()
    log_lines = []

    def log(s: str) -> None:
        log_lines.append(s)
        print(s)

    log("=" * 60)
    log("Model: SVM (RBF kernel)")
    log("Params: C=10.0, gamma=scale, cache_size=500, random_state=42")
    log(f"Experiment started: {experiment_start.isoformat()}")
    log("=" * 60)

    print("Loading processed data and extracting band-power features...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y, class_names = load_features_and_labels(PROCESSED_DIR, max_per_group=args.max)
    log("")
    log(f"Total samples: {X.shape[0]}, features: {X.shape[1]}")
    log(feature_description())
    log("  (N_CHANNELS=19 = standard 10-20; actual channels per file vary—we use first file's order and pad to 19.)")
    for i, name in enumerate(class_names):
        log(f"  {name}: {(y == i).sum()}")

    # Undersample to smallest class so all four classes are balanced (avoids F1=0 for majority)
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

    # Split first (no scaling yet) so scaler is fit only on train data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X_bal, y_bal))
    X_train, X_test = X_bal[train_idx], X_bal[test_idx]
    y_train, y_test = y_bal[train_idx], y_bal[test_idx]

    # Pipeline: scale then SVM. Scaler sees only train; stronger SVM (C=10, cache_size to allow more SVs)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", cache_size=500, random_state=RANDOM_STATE)),
    ])
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    log(f"Train time: {train_time:.3f} s")

    # Test set (on balanced hold-out)
    y_pred = pipeline.predict(X_test)
    log("")
    log("--- Test set results (balanced hold-out) ---")
    log(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    log("Confusion matrix:")
    log(str(confusion_matrix(y_test, y_pred)))

    # Cross-validation on full dataset: use pipeline so each fold scales on its own train split
    log("")
    log("--- Stratified 5-fold CV (on full dataset) ---")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    y_cv = cross_val_predict(pipeline, X, y, cv=cv)
    cv_time = time.perf_counter() - t0
    log(f"CV time: {cv_time:.3f} s")
    log(classification_report(y, y_cv, target_names=class_names, zero_division=0))
    log("Confusion matrix (CV):")
    log(str(confusion_matrix(y, y_cv)))

    experiment_end = datetime.now()
    log("")
    log("=" * 60)
    log(f"Experiment ended: {experiment_end.isoformat()}")
    log("=" * 60)

    # Save to file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = experiment_start.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_svm_{timestamp}.txt"
    out_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
