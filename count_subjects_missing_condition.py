"""
Count subjects that have only closed-eyes or only open-eyes recordings.

Heuristic:
- Condition is inferred from filename tokens:
    closed: *_zg*, *_зг*
    open:   *_og*, *_ог*
- A "subject id" is the filename with condition token removed (and extension stripped),
  normalized to lowercase. This groups e.g. "IVANOV_ILYA_4_zg.edf" with
  "IVANOV_ILYA_4_og.EDF".

Usage:
  python count_subjects_missing_condition.py
  python count_subjects_missing_condition.py --data-dir data_kids
  python count_subjects_missing_condition.py --csv-out results/subject_condition_counts.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


CLOSED_TOKENS = ("_zg", "_зг")
OPEN_TOKENS = ("_og", "_ог")


@dataclass(frozen=True)
class SubjectSummary:
    subject_id: str
    any_closed: bool
    any_open: bool
    n_files: int


def detect_condition(name: str) -> str | None:
    n = name.lower()
    if any(t in n for t in CLOSED_TOKENS):
        return "closed"
    if any(t in n for t in OPEN_TOKENS):
        return "open"
    return None


_COND_RE = re.compile(r"(_zg|_og|_зг|_ог)", re.IGNORECASE)


def subject_id_from_filename(filename: str) -> str:
    base = Path(filename).stem
    base = _COND_RE.sub("", base)
    base = re.sub(r"__+", "_", base)
    base = base.strip("_ ").lower()
    return base


def iter_edf_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".edf")


def summarize_subjects(paths: list[Path]) -> dict[str, SubjectSummary]:
    agg: dict[str, dict[str, object]] = {}
    for p in paths:
        cond = detect_condition(p.name)
        if cond is None:
            continue
        sid = subject_id_from_filename(p.name)
        if not sid:
            continue
        a = agg.setdefault(sid, {"closed": False, "open": False, "n": 0})
        a["n"] = int(a["n"]) + 1
        if cond == "closed":
            a["closed"] = True
        elif cond == "open":
            a["open"] = True

    out: dict[str, SubjectSummary] = {}
    for sid, a in agg.items():
        out[sid] = SubjectSummary(
            subject_id=sid,
            any_closed=bool(a["closed"]),
            any_open=bool(a["open"]),
            n_files=int(a["n"]),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Count subjects missing open/closed condition recordings.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root directory (default: data)")
    ap.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path")
    args = ap.parse_args()

    paths = iter_edf_files(args.data_dir)
    subjects = summarize_subjects(paths)

    only_closed = [s for s in subjects.values() if s.any_closed and not s.any_open]
    only_open = [s for s in subjects.values() if s.any_open and not s.any_closed]
    both = [s for s in subjects.values() if s.any_open and s.any_closed]

    print(f"Data dir: {args.data_dir.resolve()}")
    print(f"EDF files scanned: {len(paths)}")
    print(f"Subjects (with detectable condition token): {len(subjects)}")
    print(f"  Both conditions: {len(both)}")
    print(f"  Only closed:     {len(only_closed)}")
    print(f"  Only open:       {len(only_open)}")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(subjects.values(), key=lambda s: (not s.any_open or not s.any_closed, -s.n_files, s.subject_id))
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["subject_id", "any_closed", "any_open", "n_files", "missing"])
            for s in rows:
                if s.any_closed and not s.any_open:
                    missing = "open"
                elif s.any_open and not s.any_closed:
                    missing = "closed"
                else:
                    missing = ""
                w.writerow([s.subject_id, int(s.any_closed), int(s.any_open), s.n_files, missing])
        print(f"Wrote CSV: {args.csv_out.resolve()}")


if __name__ == "__main__":
    main()

