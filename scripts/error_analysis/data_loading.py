from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edf_loader import load_raw_edf_resilient
from eeg_experiment_shared import GROUPS, GROUPS_DATA_KIDS, find_edf_files
from eeg_features import extract_all_features, get_feature_names
from .subject_ids import subject_id_from_path


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    class_names: list[str]
    feature_names: list[str]
    subject_ids: list[str]
    paths: list[Path]
    canonical_ch_names: list[str] | None


def _resolve_groups_for_dir(data_dir: Path) -> dict[str, str]:
    # heuristic: data_kids uses shorter folder names
    try:
        children = {p.name for p in data_dir.iterdir() if p.is_dir()}
    except Exception:
        children = set()
    if any(name in children for name in GROUPS_DATA_KIDS.keys()):
        return GROUPS_DATA_KIDS
    return GROUPS


def load_dataset_with_subjects(
    *,
    data_dir: Path,
    eyes: str,
    max_per_group: int | None = None,
    groups: dict[str, str] | None = None,
) -> Dataset:
    """
    Load features + labels + subject ids + file paths.

    No splitting here (caller is responsible). This is used for GroupKFold to avoid leakage.
    """
    eyes = eyes.lower().strip()
    if eyes not in ("closed", "open"):
        raise ValueError("eyes must be 'closed' or 'open'")

    groups = groups or _resolve_groups_for_dir(data_dir)
    class_names = list(groups.values())
    feature_names = get_feature_names(eyes_condition=eyes)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    subj_list: list[str] = []
    path_list: list[Path] = []
    canonical_ch_names = None

    for group_folder, label_name in groups.items():
        paths = find_edf_files(data_dir, group_folder, max_per_group, eyes_condition=eyes)
        if not paths:
            continue
        label_idx = class_names.index(label_name)
        for p in paths:
            try:
                raw = load_raw_edf_resilient(p, preload=True, verbose=False)
                data_check = raw.get_data()
                if not np.all(np.isfinite(data_check)):
                    raise ValueError("Non-finite values in raw data (NaN/Inf detected)")
                out = extract_all_features(raw, canonical_ch_names, eyes_condition=eyes)
                if canonical_ch_names is None:
                    feat, canonical_ch_names = out
                else:
                    feat = out
                feat = np.asarray(feat, dtype=float)
                if not np.all(np.isfinite(feat)):
                    raise ValueError("Non-finite values in features (NaN/Inf detected)")
                X_list.append(feat)
                y_list.append(label_idx)
                subj_list.append(subject_id_from_path(p))
                path_list.append(p)
            except Exception:
                # caller wants strict leakage guarantees; skipping bad files is better than guessing
                continue

    if not X_list:
        raise FileNotFoundError(f"No EDF files loaded under {data_dir}")

    X = np.vstack([x.reshape(1, -1) for x in X_list])
    y = np.asarray(y_list, dtype=int)
    return Dataset(
        X=X,
        y=y,
        class_names=class_names,
        feature_names=feature_names,
        subject_ids=subj_list,
        paths=path_list,
        canonical_ch_names=canonical_ch_names,
    )

