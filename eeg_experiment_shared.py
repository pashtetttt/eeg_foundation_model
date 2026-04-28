"""
Shared data loading and feature selection for RF / BalancedRF / XGBoost experiment scripts.

- Uses full class-imbalanced data (no per-class subsampling). Stratified splits preserve proportions.
- Feature modes: all, alpha (from get_feature_indices_by_category), or names from a selected-features file
  (e.g. mutual-information top-k list aligned with get_feature_names).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from eeg_features import extract_all_features, feature_description, get_feature_indices_by_category, get_feature_names
from edf_loader import load_raw_edf_resilient

BASE = Path(".")
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"

DEFAULT_SELECTED_FEATURES_PATH = RESULTS_DIR / "selected_features_k150_mutual_info_20260319_205929.txt"

GROUPS = {
    "preschooler": "preschooler",
    "primary": "primary",
    "teenager": "teenager",
    "adolescence": "adolescence",
}

GROUPS_DATA_KIDS = {
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


def load_features_and_labels(
    data_dir: Path,
    max_per_group: int | None,
    eyes_condition: str = "closed",
    groups: dict[str, str] | None = None,
    include_literature: bool = False,
    literature_surrogate_iters: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import mne

    groups = groups or GROUPS
    X_list, y_list = [], []
    class_names = list(groups.values())
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
                out = extract_all_features(
                    raw,
                    canonical_ch_names,
                    eyes_condition=eyes_condition,
                    include_literature=include_literature,
                    literature_surrogate_iters=literature_surrogate_iters,
                )
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
        raise FileNotFoundError(f"No .edf files found under {data_dir}.")

    print(f"  Total: {total_loaded} loaded, {total_skipped} skipped")
    return np.asarray(X_list, dtype=float), np.asarray(y_list, dtype=int), class_names


def resolve_feature_indices(
    eyes_condition: str,
    feature_mode: str,
    selected_path: Path | None,
) -> tuple[np.ndarray, list[str]]:
    """
    Return column indices into the full feature vector for the given mode.

    feature_mode: 'all' | 'alpha' | 'non_alpha' | 'selected' | 'all_plus_complexity'
    """
    feature_mode = feature_mode.lower().strip()
    if feature_mode == "all":
        n = len(get_feature_names(eyes_condition, include_literature=False))
        return np.arange(n, dtype=int), ["all columns (legacy)"]

    if feature_mode == "all_plus_complexity":
        n = len(get_feature_names(eyes_condition, include_literature=True))
        return np.arange(n, dtype=int), ["legacy + literature complexity block"]

    if feature_mode == "alpha":
        cats = get_feature_indices_by_category(eyes_condition=eyes_condition, include_literature=False)
        if "alpha" not in cats:
            raise ValueError("alpha feature group not available")
        idx = np.array(cats["alpha"], dtype=int)
        return idx, [f"alpha category ({len(idx)} features)"]

    if feature_mode == "non_alpha":
        cats = get_feature_indices_by_category(eyes_condition=eyes_condition, include_literature=False)
        if "alpha" not in cats:
            raise ValueError("alpha feature group not available")
        n = len(get_feature_names(eyes_condition))
        alpha = set(cats["alpha"])
        idx = np.array([i for i in range(n) if i not in alpha], dtype=int)
        return idx, [f"non_alpha category ({len(idx)} features) = all - alpha"]

    if feature_mode == "selected":
        path = selected_path or DEFAULT_SELECTED_FEATURES_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Selected features file not found: {path}")
        return indices_from_name_file(path, eyes_condition)

    raise ValueError(
        f"Unknown feature_mode '{feature_mode}'. Use: all, all_plus_complexity, alpha, non_alpha, selected"
    )


def indices_from_name_file(path: Path, eyes_condition: str) -> tuple[np.ndarray, list[str]]:
    """Map one feature name per line to indices using get_feature_names order."""
    canonical = get_feature_names(eyes_condition)
    name_to_idx = {n: i for i, n in enumerate(canonical)}
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    indices: list[int] = []
    missing: list[str] = []
    for name in lines:
        if name in name_to_idx:
            indices.append(name_to_idx[name])
        else:
            missing.append(name)

    notes: list[str] = [f"file={path.name}, matched={len(indices)}, lines={len(lines)}"]
    if missing:
        notes.append(f"skipped {len(missing)} names not in {eyes_condition} layout (e.g. closed-only topo): ...")
        # keep log short
        preview = ", ".join(missing[:8])
        if len(missing) > 8:
            preview += ", ..."
        notes[-1] = f"skipped {len(missing)} names not in {eyes_condition} layout: {preview}"

    if not indices:
        raise ValueError(f"No features from {path} could be mapped for eyes_condition={eyes_condition}")

    uniq = np.array(sorted(set(indices)), dtype=int)
    return uniq, notes


def apply_feature_selection(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return X[:, indices]


def load_and_prepare_matrix(
    eyes_condition: str,
    feature_mode: str,
    max_per_group: int | None,
    selected_path: Path | None,
    data_dir: Path | None = None,
    groups: dict[str, str] | None = None,
    literature_surrogate_iters: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load data, apply feature selection, return X, y, class_names, selection_notes.
    """
    fm = feature_mode.lower().strip()
    include_lit = fm == "all_plus_complexity"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y, class_names = load_features_and_labels(
            data_dir or DATA_DIR,
            max_per_group,
            eyes_condition,
            groups=groups,
            include_literature=include_lit,
            literature_surrogate_iters=literature_surrogate_iters,
        )

    idx, notes = resolve_feature_indices(eyes_condition, feature_mode, selected_path)
    X = apply_feature_selection(X, idx)
    return X, y, class_names, notes
