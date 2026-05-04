"""Helpers for feature matrix I/O and naming used by the caching pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def features_cache_path(
    results_dir: Path,
    *,
    condition: str,
    cohort_name: str,
) -> Path:
    """Path to cached handcrafted+DFA matrix."""
    return results_dir / "features" / f"features_{condition}_{cohort_name}.npy"


def embeddings_cache_path(
    results_dir: Path,
    *,
    model: str,
    condition: str,
    cohort_name: str,
) -> Path:
    return results_dir / "embeddings" / f"embeddings_{model}_{condition}_{cohort_name}.npz"


def merged_cache_path(
    results_dir: Path,
    *,
    condition: str,
    cohort_name: str,
) -> Path:
    return results_dir / "features" / f"merged_{condition}_{cohort_name}.npz"


def save_subject_mapping_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_feature_matrix(path: Path) -> np.ndarray:
    return np.load(path)
