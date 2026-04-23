from __future__ import annotations

import re
from pathlib import Path

from eeg_experiment_shared import CLOSED_EYES_SUBSTRINGS, OPEN_EYES_SUBSTRINGS


_SUFFIX_CLEAN = re.compile(r"[\s._-]+$")


def parse_eyes_from_name(filename: str) -> str | None:
    name = filename
    if any(s in name for s in CLOSED_EYES_SUBSTRINGS):
        return "closed"
    if any(s in name for s in OPEN_EYES_SUBSTRINGS):
        return "open"
    return None


def subject_id_from_path(p: Path) -> str:
    """
    Extract a stable subject id from EDF filename.

    Current datasets typically store two files per subject distinguished only by an eyes-condition substring.
    We strip known eyes substrings and normalize separators/case.
    """
    stem = p.stem
    for s in CLOSED_EYES_SUBSTRINGS + OPEN_EYES_SUBSTRINGS:
        stem = stem.replace(s, "")
    stem = _SUFFIX_CLEAN.sub("", stem)
    stem = stem.strip().lower()
    return stem

