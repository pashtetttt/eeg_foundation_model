"""
Shared data discovery helpers for pipeline scripts.

All paths derive from ``data_dir`` in config — no hardcoded cohort directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from eeg_experiment_shared import GROUPS, find_edf_files


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a YAML mapping at top level.")
    return data


def resolve_data_dir(cfg: dict[str, Any]) -> Path:
    raw = cfg.get("data_dir", "data")
    return Path(raw).expanduser().resolve()


def subject_id_for_path(edf_path: Path, data_dir: Path) -> str:
    """Stable string ID: path relative to data_dir (posix, for joins)."""
    edf_path = edf_path.resolve()
    data_dir = data_dir.resolve()
    try:
        rel = edf_path.relative_to(data_dir)
    except ValueError:
        rel = edf_path
    return rel.as_posix()


@dataclass
class RecordingManifestRow:
    subject_id: str
    edf_path: Path
    label_idx: int
    label_name: str
    group_folder: str


def build_recording_manifest(
    data_dir: Path,
    *,
    eyes_condition: str = "closed",
    max_per_group: int | None = None,
    groups: dict[str, str] | None = None,
) -> list[RecordingManifestRow]:
    """
    Enumerate all EDF recordings under ``data_dir`` using the same folder layout as
    ``eeg_experiment_shared.GROUPS`` (class folder names are keys, display names are values).
    """
    groups = groups or GROUPS
    class_names = list(groups.values())
    rows: list[RecordingManifestRow] = []
    for group_folder, label_name in groups.items():
        paths = find_edf_files(data_dir, group_folder, max_per_group, eyes_condition)
        label_idx = class_names.index(label_name)
        for p in paths:
            sid = subject_id_for_path(p, data_dir)
            rows.append(
                RecordingManifestRow(
                    subject_id=sid,
                    edf_path=p.resolve(),
                    label_idx=label_idx,
                    label_name=label_name,
                    group_folder=group_folder,
                )
            )
    return sorted(rows, key=lambda r: r.subject_id)
