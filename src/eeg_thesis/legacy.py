"""Adapters to run legacy top-level scripts from package-style wrappers."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_legacy(script_name: str) -> None:
    script_path = ROOT / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Legacy script not found: {script_path}")
    runpy.run_path(str(script_path), run_name="__main__")
