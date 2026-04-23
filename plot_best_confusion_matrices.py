"""
Train the best model (auto-selected from results files) and save nicer confusion matrices.

Selection rule:
- Reads experiment result files in --results-dir for the given (eyes, features).
- Uses the same parsing logic as analyze_experiment_results.py:
  picks the config with highest macro_f1 in the CV section for each model file.
- Then selects the single best row across models by macro_f1 (tie-break: balanced_acc, accuracy).

Outputs:
- confusion matrix PNG (counts + normalized) for hold-out split

Examples:
  ./.venv/bin/python plot_best_confusion_matrices.py --data-dir data --results-dir results --eyes closed --features selected
  ./.venv/bin/python plot_best_confusion_matrices.py --data-dir data_kids --results-dir results_kids --eyes open --features all
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from eeg_experiment_shared import (
    DEFAULT_SELECTED_FEATURES_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    load_and_prepare_matrix,
)


@dataclass(frozen=True)
class PickedConfig:
    model: str  # rf|brf|xgb
    config_name: str
    params: dict
    macro_f1: float
    balanced_acc: float
    accuracy: float
    source_file: Path


CFG_LINE = re.compile(r"^>> \[(?P<name>[^\]]+)\]\s+(?P<dict>\{.*\})\s*$", re.MULTILINE)
METRIC_RF = re.compile(
    r"^\s*accuracy=(?P<acc>[\d.]+)\s+balanced_acc=(?P<bal>[\d.]+)\s+macro_f1=(?P<f1>[\d.]+)",
    re.MULTILINE,
)
METRIC_SHORT = re.compile(
    r"^\s*acc=(?P<acc>[\d.]+)\s+bal_acc=(?P<bal>[\d.]+)\s+macro_f1=(?P<f1>[\d.]+)",
    re.MULTILINE,
)

CV_HEADERS = (
    "--- Stratified 5-fold CV (full data, same configs) ---",
    "--- Stratified 5-fold CV ---",
    "--- Stratified 5-fold CV (balanced weights per train fold) ---",
)


def _split_cv(text: str) -> str | None:
    for h in CV_HEADERS:
        if h in text:
            return text.split(h, 1)[1]
    return None


def _parse_best_from_file(path: Path) -> PickedConfig | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    cv = _split_cv(text)
    if not cv:
        return None

    blocks = []
    # Split into chunks by ">> [name]"
    parts = re.split(r"^>> \[([^\]]+)\].*$", cv, flags=re.MULTILINE)
    # parts[0] preamble; then name, body alternating
    it = iter(parts)
    next(it, None)
    for name, body in zip(it, it):
        name = name.strip()
        m = METRIC_RF.search(body) or METRIC_SHORT.search(body)
        if not m:
            continue
        acc = float(m.group("acc"))
        bal = float(m.group("bal"))
        f1 = float(m.group("f1"))
        # find params dict (may be in this block as "{...}" after >> line)
        dm = re.search(r"\{.*\}", body, flags=re.DOTALL)
        params = {}
        if dm:
            try:
                params = ast.literal_eval(dm.group(0))
            except Exception:
                params = {}
        blocks.append((name, params, f1, bal, acc))

    if not blocks:
        return None
    name, params, f1, bal, acc = max(blocks, key=lambda t: (t[2], t[3], t[4]))

    model = "rf" if "rf_experiments" in path.name else "brf" if "brf_experiments" in path.name else "xgb"
    # Some files include "name" field inside dict; normalize away
    if isinstance(params, dict) and "name" in params:
        params = dict(params)
        params.pop("name", None)
    return PickedConfig(model=model, config_name=name, params=params, macro_f1=f1, balanced_acc=bal, accuracy=acc, source_file=path)


def _latest_file(results_dir: Path, pattern: str) -> Path | None:
    files = sorted(results_dir.glob(pattern))
    return files[-1] if files else None


def pick_best(results_dir: Path, eyes: str, features: str) -> PickedConfig:
    candidates: list[PickedConfig] = []
    for model, patt in [
        ("rf", f"results_rf_experiments_{eyes}_{features}_*.txt"),
        ("brf", f"results_brf_experiments_{eyes}_{features}_*.txt"),
        ("xgb", f"results_xgb_experiments_{eyes}_{features}_*.txt"),
    ]:
        p = _latest_file(results_dir, patt)
        if not p:
            continue
        pc = _parse_best_from_file(p)
        if pc:
            candidates.append(pc)
    if not candidates:
        raise FileNotFoundError(f"No experiment results found under {results_dir} for eyes={eyes} features={features}")
    return max(candidates, key=lambda c: (c.macro_f1, c.balanced_acc, c.accuracy))


def build_pipeline(pick: PickedConfig):
    if pick.model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **pick.params)),
            ]
        )
    if pick.model == "brf":
        from imblearn.ensemble import BalancedRandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", BalancedRandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **pick.params)),
            ]
        )
    if pick.model == "xgb":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss", **pick.params)),
            ]
        )
    raise ValueError(pick.model)


def save_confusion_plots(out_dir: Path, title_prefix: str, class_names: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    cmn = cm.astype(float) / np.maximum(1, cm.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax[0], cmap="Blues", values_format="d", colorbar=False)
    ax[0].set_title(f"{title_prefix} (counts)")
    ConfusionMatrixDisplay(cmn, display_labels=class_names).plot(ax=ax[1], cmap="Blues", values_format=".2f", colorbar=True)
    ax[1].set_title(f"{title_prefix} (row-normalized)")
    plt.tight_layout()
    out = out_dir / f"confusion_{title_prefix.replace(' ', '_')}_{ts}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot nicer confusion matrices for the best model (auto-picked).")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to save confusion plots (default: results-dir)")
    ap.add_argument("--max", type=int, default=None, help="Max samples per class (default: all)")
    ap.add_argument("--eyes", type=str, default="closed", choices=["closed", "open"])
    ap.add_argument("--features", type=str, default="selected", choices=["all", "alpha", "selected"])
    ap.add_argument("--selected-path", type=Path, default=DEFAULT_SELECTED_FEATURES_PATH)
    args = ap.parse_args()

    out_dir = args.out_dir or args.results_dir
    selected_path = args.selected_path if args.features == "selected" else None

    pick = pick_best(args.results_dir, args.eyes, args.features)
    print(f"Picked best: model={pick.model} cfg={pick.config_name} macro_f1={pick.macro_f1:.4f} from {pick.source_file.name}")

    X, y, class_names, _notes = load_and_prepare_matrix(
        eyes_condition=args.eyes,
        feature_mode=args.features,
        max_per_group=args.max,
        selected_path=selected_path,
        data_dir=args.data_dir,
    )

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te = next(sss.split(X, y))
    X_train, X_test = X[tr], X[te]
    y_train, y_test = y[tr], y[te]

    pipe = build_pipeline(pick)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    title = f"{pick.model} {args.eyes} {args.features} | {args.data_dir.name}"
    save_confusion_plots(out_dir, title, class_names, y_test, y_pred)


if __name__ == "__main__":
    main()

