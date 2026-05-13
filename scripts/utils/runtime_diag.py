"""Lightweight runtime diagnostics for reproducibility logs."""

from __future__ import annotations


def log_library_versions(*names: str) -> None:
    """Print import versions for key dependencies (one line each)."""
    for name in names:
        try:
            mod = __import__(name)
        except ImportError as e:
            print(f"[versions] {name}: not importable ({e})")
            continue
        ver = getattr(mod, "__version__", "?")
        print(f"[versions] {name}: {ver}")
