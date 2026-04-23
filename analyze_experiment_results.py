"""
Summarize grid-search experiment logs into tables (best config per run).

Reads files matching:
  results/results_rf_experiments_{closed|open}_{all|alpha|selected}_*.txt
  results/results_brf_experiments_*.txt
  results/results_xgb_experiments_*.txt
  results/results_dummy_*.txt
  results/results_rf_resampling_{closed|open}_{all|alpha|selected}_*.txt

For each file, parses the **Stratified 5-fold CV** section only, finds the hyperparameter
config with the highest macro-F1 (tie-break: balanced accuracy), and records that row.
Also records **weighted F1** from the classification report (``weighted avg`` row), which
reflects support-weighted performance alongside macro-F1.

When to run
-----------
- **After experiments finish:** gives a complete table (recommended).
- **In parallel** with still-running jobs: safe, but only **finished** logs appear;
  re-run this script later to refresh.

Usage:
  python analyze_experiment_results.py
  python analyze_experiment_results.py --results-dir results --format markdown csv
  python analyze_experiment_results.py --include-all-runs   # best per file, not only latest timestamp per condition
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedRun:
    model: str  # rf | brf | xgb | dummy
    eyes: str
    features: str
    timestamp: str
    path: Path
    best_config_name: str
    macro_f1: float
    weighted_f1: float | None  # from sklearn report "weighted avg" f1-score
    balanced_acc: float | None
    accuracy: float


CV_HEADER_RF = "--- Stratified 5-fold CV (full data, same configs) ---"
CV_HEADER_BRF = "--- Stratified 5-fold CV ---"
CV_HEADER_XGB = "--- Stratified 5-fold CV (balanced weights per train fold) ---"
CV_HEADER_DUMMY = "--- Stratified 5-fold CV (on full dataset) ---"
CV_HEADER_RF_RESAMPLING = "--- Stratified 5-fold CV ---"


def _split_cv_section(text: str) -> str | None:
    """Return text after the CV header (first matching)."""
    for marker in (CV_HEADER_RF, CV_HEADER_BRF, CV_HEADER_XGB, CV_HEADER_DUMMY, CV_HEADER_RF_RESAMPLING):
        if marker in text:
            return text.split(marker, 1)[1]
    return None


# >> [rf_baseline] {...}  or  >> [brf_baseline] CV time=...
BLOCK_START = re.compile(r"^>> \[([^\]]+)\]", re.MULTILINE)

# RF CV line:    accuracy=0.4310  balanced_acc=0.4192  macro_f1=0.3264
METRIC_LINE_RF = re.compile(
    r"^\s*accuracy=(?P<acc>[\d.]+)\s+balanced_acc=(?P<bal>[\d.]+)\s+macro_f1=(?P<f1>[\d.]+)",
    re.MULTILINE,
)

# BRF / XGB CV line:    acc=0.4138  bal_acc=0.4026  macro_f1=0.3132
METRIC_LINE_SHORT = re.compile(
    r"^\s*acc=(?P<acc>[\d.]+)\s+bal_acc=(?P<bal>[\d.]+)\s+macro_f1=(?P<f1>[\d.]+)",
    re.MULTILINE,
)

# sklearn classification_report: "weighted avg       0.28      0.41      0.32        58"
WEIGHTED_AVG_F1 = re.compile(
    r"^\s*weighted avg\s+[\d.]+\s+[\d.]+\s+(?P<wf1>[\d.]+)",
    re.MULTILINE,
)
MACRO_AVG_F1 = re.compile(
    r"^\s*macro avg\s+[\d.]+\s+[\d.]+\s+(?P<mf1>[\d.]+)",
    re.MULTILINE,
)
ACCURACY_LINE = re.compile(
    r"^\s*accuracy\s+(?P<acc>[\d.]+)\s+\d+",
    re.MULTILINE,
)


def parse_cv_blocks(cv_text: str) -> list[tuple[str, float, float, float, float | None]]:
    """
    Parse (config_name, accuracy, balanced_acc, macro_f1, weighted_f1) for each config in CV section.
    weighted_f1 is taken from the report's weighted-avg f1-score column when present.
    """
    out: list[tuple[str, float, float, float, float | None]] = []
    # Split by >> [name] starts
    parts = BLOCK_START.split(cv_text)
    if not parts:
        return out
    # parts[0] is preamble before first >>; rest alternate: name, body, name, body...
    it = iter(parts)
    next(it, None)  # drop preamble
    for name, body in zip(it, it):
        name = name.strip()
        m = METRIC_LINE_RF.search(body)
        if not m:
            m = METRIC_LINE_SHORT.search(body)
        if not m:
            continue
        acc = float(m.group("acc"))
        bal = float(m.group("bal"))
        f1 = float(m.group("f1"))
        wm = WEIGHTED_AVG_F1.search(body)
        weighted_f1 = float(wm.group("wf1")) if wm else None
        out.append((name, acc, bal, f1, weighted_f1))
    return out


FILENAME_RE = re.compile(
    r"^results_(?P<model>rf|brf|xgb)_experiments_(?P<eyes>closed|open)_(?P<feat>all|alpha|selected)_(?P<ts>\d{8}_\d{6})\.txt$"
)
DUMMY_FILENAME_RE = re.compile(r"^results_dummy_(?P<ts>\d{8}_\d{6})\.txt$")
RF_RESAMPLING_FILENAME_RE = re.compile(
    r"^results_rf_resampling_(?P<eyes>closed|open)_(?P<feat>all|alpha|selected)_(?P<ts>\d{8}_\d{6})\.txt$"
)


def parse_filename(path: Path) -> tuple[str, str, str, str] | None:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.group("model"), m.group("eyes"), m.group("feat"), m.group("ts")


def parse_dummy_filename(path: Path) -> str | None:
    m = DUMMY_FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.group("ts")


def parse_dummy_meta(text: str) -> tuple[str, str] | None:
    eyes_m = re.search(r"^Eyes condition:\s*(closed|open)\s*$", text, re.MULTILINE)
    feat_m = re.search(r"^Feature subset:\s*([A-Za-z_]+)\s*$", text, re.MULTILINE)
    if not eyes_m or not feat_m:
        return None
    return eyes_m.group(1), feat_m.group(1)


def parse_dummy_cv_metrics(cv_text: str) -> tuple[float, float, float] | None:
    """
    Parse dummy CV metrics from classification_report-style CV section:
    returns (accuracy, macro_f1, weighted_f1).
    """
    am = ACCURACY_LINE.search(cv_text)
    mm = MACRO_AVG_F1.search(cv_text)
    wm = WEIGHTED_AVG_F1.search(cv_text)
    if not am or not mm:
        return None
    acc = float(am.group("acc"))
    macro_f1 = float(mm.group("mf1"))
    weighted_f1 = float(wm.group("wf1")) if wm else None
    if weighted_f1 is None:
        return None
    return acc, macro_f1, weighted_f1


def best_in_file(
    blocks: list[tuple[str, float, float, float, float | None]],
) -> tuple[str, float, float, float, float | None] | None:
    if not blocks:
        return None
    # Maximize macro_f1, then balanced_acc, then accuracy
    best = max(blocks, key=lambda t: (t[3], t[2], t[1]))
    name, acc, bal, f1, wf1 = best
    return name, acc, bal, f1, wf1


def collect_runs(results_dir: Path, latest_only: bool) -> list[ParsedRun]:
    files: list[Path] = []
    for pattern in (
        "results_rf_experiments_*.txt",
        "results_brf_experiments_*.txt",
        "results_xgb_experiments_*.txt",
        "results_dummy_*.txt",
        "results_rf_resampling_*.txt",
    ):
        files.extend(sorted(results_dir.rglob(pattern)))

    grouped: dict[tuple[str, str, str, str], list[tuple[str, Path]]] = {}
    for p in files:
        meta = parse_filename(p)
        if meta:
            model, eyes, feat, ts = meta
            key = (model, eyes, feat)
            grouped.setdefault(key, []).append((ts, p))
            continue
        rm = RF_RESAMPLING_FILENAME_RE.match(p.name)
        if rm:
            model = "rf_resampling"
            eyes = rm.group("eyes")
            feat = rm.group("feat")
            ts = rm.group("ts")
            grouped.setdefault((model, eyes, feat), []).append((ts, p))
            continue
        dummy_ts = parse_dummy_filename(p)
        if dummy_ts is not None:
            # Keep one bucket for dummy per (eyes, features) extracted from file content.
            grouped.setdefault(("dummy", "_pending", "_pending"), []).append((dummy_ts, p))

    runs: list[ParsedRun] = []
    for key, items in grouped.items():
        model, eyes, feat = key
        if model == "dummy":
            # Dummy metadata is in file body, so resolve per-file.
            items.sort(key=lambda x: x[0], reverse=True)
            if latest_only:
                latest_per_condition: dict[tuple[str, str], tuple[str, Path]] = {}
                for ts, path in items:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    meta = parse_dummy_meta(text)
                    if not meta:
                        continue
                    cond = (meta[0], meta[1])
                    if cond not in latest_per_condition:
                        latest_per_condition[cond] = (ts, path)
                to_process = list(latest_per_condition.values())
            else:
                to_process = items
            for ts, path in to_process:
                text = path.read_text(encoding="utf-8", errors="replace")
                dm = parse_dummy_meta(text)
                if not dm:
                    continue
                eyes_d, feat_d = dm
                cv = _split_cv_section(text)
                if not cv:
                    continue
                metrics = parse_dummy_cv_metrics(cv)
                if not metrics:
                    continue
                acc, macro_f1, weighted_f1 = metrics
                runs.append(
                    ParsedRun(
                        model="dummy",
                        eyes=eyes_d,
                        features=feat_d,
                        timestamp=ts,
                        path=path,
                        best_config_name="most_frequent",
                        macro_f1=macro_f1,
                        weighted_f1=weighted_f1,
                        balanced_acc=None,
                        accuracy=acc,
                    )
                )
            continue

        if model == "rf_resampling":
            items.sort(key=lambda x: x[0], reverse=True)
            to_process = [items[0]] if latest_only else items
            for ts, path in to_process:
                text = path.read_text(encoding="utf-8", errors="replace")
                cv = _split_cv_section(text)
                if not cv:
                    continue
                # For resampling logs we keep *all* strategies (one row per block)
                blocks = parse_cv_blocks(cv)
                for name, acc, bal, f1, wf1 in blocks:
                    runs.append(
                        ParsedRun(
                            model="rf_resampling",
                            eyes=eyes,
                            features=feat,
                            timestamp=ts,
                            path=path,
                            best_config_name=name,
                            macro_f1=f1,
                            weighted_f1=wf1,
                            balanced_acc=bal,
                            accuracy=acc,
                        )
                    )
            continue

        items.sort(key=lambda x: x[0], reverse=True)
        to_process = [items[0]] if latest_only else items
        for ts, path in to_process:
            text = path.read_text(encoding="utf-8", errors="replace")
            cv = _split_cv_section(text)
            if not cv:
                continue
            blocks = parse_cv_blocks(cv)
            b = best_in_file(blocks)
            if not b:
                continue
            name, acc, bal, f1, wf1 = b
            runs.append(
                ParsedRun(
                    model=model,
                    eyes=eyes,
                    features=feat,
                    timestamp=ts,
                    path=path,
                    best_config_name=name,
                    macro_f1=f1,
                    weighted_f1=wf1,
                    balanced_acc=bal,
                    accuracy=acc,
                )
            )
    return runs


def sort_key(r: ParsedRun) -> tuple:
    order_eyes = ("closed", "open")
    order_feat = ("all", "alpha", "selected")
    order_model = ("rf", "brf", "xgb", "rf_resampling", "dummy")
    return (
        order_eyes.index(r.eyes) if r.eyes in order_eyes else 99,
        order_feat.index(r.features) if r.features in order_feat else 99,
        order_model.index(r.model) if r.model in order_model else 99,
    )


def _fmt_wf1(r: ParsedRun) -> str:
    return f"{r.weighted_f1:.4f}" if r.weighted_f1 is not None else "—"


def _fmt_bal(r: ParsedRun) -> str:
    return f"{r.balanced_acc:.4f}" if r.balanced_acc is not None else "—"


def print_markdown(runs: list[ParsedRun]) -> None:
    runs = sorted(runs, key=sort_key)
    print(
        "| model | eyes | features | best_config | macro_f1 | weighted_f1 | balanced_acc | accuracy | source |"
    )
    print(
        "|-------|------|----------|-------------|----------|---------------|--------------|----------|--------|"
    )
    for r in runs:
        print(
            f"| {r.model} | {r.eyes} | {r.features} | {r.best_config_name} | "
            f"{r.macro_f1:.4f} | {_fmt_wf1(r)} | {_fmt_bal(r)} | {r.accuracy:.4f} | `{r.path.name}` |"
        )


def print_pivot_macro_f1(runs: list[ParsedRun]) -> None:
    """One row per (eyes, features), columns rf / brf / xgb — best macro_f1 in that cell."""
    cell: dict[tuple[str, str], dict[str, ParsedRun]] = {}
    for r in runs:
        key = (r.eyes, r.features)
        cell.setdefault(key, {})[r.model] = r

    order_eyes = ("closed", "open")
    order_feat = ("all", "alpha", "selected")
    print()
    print("### Pivot: macro_f1 (best config per model, latest log per condition)")
    print()
    print("| eyes | features | rf | brf | xgb |")
    print("|------|------------|----|----|-----|")
    for eyes in order_eyes:
        for feat in order_feat:
            row = cell.get((eyes, feat), {})

            def fmt_macro(m: str) -> str:
                x = row.get(m)
                return f"{x.macro_f1:.4f}" if x else "—"

            print(f"| {eyes} | {feat} | {fmt_macro('rf')} | {fmt_macro('brf')} | {fmt_macro('xgb')} |")

    print()
    print("### Pivot: weighted_f1 (same best-config rows as above)")
    print()
    print("| eyes | features | rf | brf | xgb |")
    print("|------|------------|----|----|-----|")
    for eyes in order_eyes:
        for feat in order_feat:
            row = cell.get((eyes, feat), {})

            def fmt_wf(m: str) -> str:
                x = row.get(m)
                if not x:
                    return "—"
                return f"{x.weighted_f1:.4f}" if x.weighted_f1 is not None else "—"

            print(f"| {eyes} | {feat} | {fmt_wf('rf')} | {fmt_wf('brf')} | {fmt_wf('xgb')} |")


def write_csv(runs: list[ParsedRun], out_path: Path) -> None:
    runs = sorted(runs, key=sort_key)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "eyes",
                "features",
                "best_config",
                "macro_f1",
                "weighted_f1",
                "balanced_acc",
                "accuracy",
                "file",
            ]
        )
        for r in runs:
            w.writerow(
                [
                    r.model,
                    r.eyes,
                    r.features,
                    r.best_config_name,
                    f"{r.macro_f1:.6f}",
                    "" if r.weighted_f1 is None else f"{r.weighted_f1:.6f}",
                    "" if r.balanced_acc is None else f"{r.balanced_acc:.6f}",
                    f"{r.accuracy:.6f}",
                    r.path.name,
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize experiment result logs into tables.")
    ap.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory with results_*.txt")
    ap.add_argument(
        "--include-all-runs",
        action="store_true",
        help="Include every timestamped file per condition (default: only latest file per model/eyes/features)",
    )
    ap.add_argument("--format", nargs="+", choices=("markdown", "csv", "pivot"), default=["markdown", "pivot"])
    ap.add_argument("--csv-out", type=Path, default=None, help="Path for CSV (default: results/experiment_summary_<ts>.csv)")
    args = ap.parse_args()

    runs = collect_runs(args.results_dir, latest_only=not args.include_all_runs)
    if not runs:
        print("No matching experiment logs found or no CV metrics could be parsed.")
        print(f"Looked under: {args.results_dir.resolve()}")
        return

    if "markdown" in args.format:
        print_markdown(runs)
        print()

    if "pivot" in args.format:
        print_pivot_macro_f1(runs)
        print()

    if "csv" in args.format:
        out = args.csv_out
        if out is None:
            from datetime import datetime

            out = args.results_dir / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        write_csv(runs, out)
        print(f"Wrote CSV: {out.resolve()}")


if __name__ == "__main__":
    main()
