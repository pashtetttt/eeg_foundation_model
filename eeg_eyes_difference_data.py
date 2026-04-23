"""
Датасет «разность признаков закрытые − открытые глаза» для одного пациента.

Для каждой пары EDF (закрытые / открытые) из одной подпапки класса извлекаются
признаки как в eeg_features.extract_all_features, затем берётся разность только по
признакам, общим для обоих eyes_condition (364 признака; 12 закрыто-специфичных
alpha-topography исключаются).

Использование:
  python eeg_eyes_difference_data.py
  python eeg_eyes_difference_data.py --max 10
  python eeg_eyes_difference_data.py --train
      # --train: RF + SMOTE (imblearn), stratified K-fold CV + fit на всех (in-sample)
  python eeg_eyes_difference_data.py --train --holdout
      # дополнительно hold-out 80/20
"""

from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from eeg_experiment_shared import DATA_DIR, GROUPS, N_SPLITS, RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from eeg_features import extract_all_features, feature_description, get_feature_names
from edf_loader import load_raw_edf_resilient

warnings.filterwarnings("ignore", message=".*does not conform to MNE naming conventions.*")

# Суффиксы имён файлов (как в eeg_experiment_shared.find_edf_files)
CLOSED_EYES_SUBSTRINGS = ("_zg", "_ZG", "_зг", "_ЗГ")
OPEN_EYES_SUBSTRINGS = ("_og", "_OG", "_ог", "_ОГ")


def parse_stem_eyes(stem: str) -> tuple[str, str] | None:
    """
    По stem без .edf вернуть (base_key, 'closed'|'open') или None, если маркер глаз не найден.
    Сравнение суффикса с конца stem (без учёта регистра).
    """
    lower = stem.lower()
    for s in CLOSED_EYES_SUBSTRINGS:
        ls = s.lower()
        if lower.endswith(ls):
            return stem[: -len(s)], "closed"
    for s in OPEN_EYES_SUBSTRINGS:
        ls = s.lower()
        if lower.endswith(ls):
            return stem[: -len(s)], "open"
    return None


def common_feature_indices() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Индексы признаков, присутствующих и в open, и в closed векторах.
    Порядок — как в get_feature_names('open'); длина 364.
    """
    names_o = get_feature_names("open")
    names_c = get_feature_names("closed")
    set_c = set(names_c)
    common_names = [n for n in names_o if n in set_c]
    if len(common_names) != len(names_o):
        raise RuntimeError("Expected all open feature names to exist in closed layout.")
    idx_o = np.arange(len(names_o), dtype=int)
    idx_c = np.array([names_c.index(n) for n in common_names], dtype=int)
    return idx_c, idx_o, common_names


def find_paired_edf_paths(
    group_folder: str,
    data_dir: Path,
) -> tuple[list[tuple[Path, Path]], dict[str, int]]:
    """
    Внутри data_dir/group_folder найти пары (closed_edf, open_edf) с одним base_key.
    Возвращает список пар и счётчики пропусков: only_closed, only_open, unmarked.
    """
    folder = data_dir / group_folder
    if not folder.exists():
        prefix = group_folder.split("(")[0].strip()
        candidates = [p for p in data_dir.iterdir() if p.is_dir() and p.name.strip().startswith(prefix)]
        if len(candidates) == 1:
            folder = candidates[0]
        else:
            return [], {"only_closed": 0, "only_open": 0, "unmarked": 0}

    paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".edf")
    by_key: dict[str, dict[str, Path]] = {}
    unmarked = 0
    for p in paths:
        parsed = parse_stem_eyes(p.stem)
        if parsed is None:
            unmarked += 1
            continue
        base, cond = parsed
        by_key.setdefault(base, {})[cond] = p

    pairs: list[tuple[Path, Path]] = []
    only_closed = only_open = 0
    for base in sorted(by_key.keys()):
        m = by_key[base]
        if "closed" in m and "open" in m:
            pairs.append((m["closed"], m["open"]))
        elif "closed" in m:
            only_closed += 1
        elif "open" in m:
            only_open += 1

    stats = {"only_closed": only_closed, "only_open": only_open, "unmarked": unmarked}
    return pairs, stats


def load_eyes_difference_features_and_labels(
    data_dir: Path,
    max_pairs: int | None,
    groups: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Загрузить матрицу X: для каждой пары (закр., откр.) строка = feat_closed[common] - feat_open[common].

    Parameters
    ----------
    max_pairs
        Максимум пар на класс (для отладки), иначе None — все пары.
    """
    groups = groups or GROUPS
    idx_c, idx_o, common_names = common_feature_indices()
    diff_names = [f"diff_{n}" for n in common_names]

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    class_names = list(groups.values())

    canonical_ch_names: list[str] | None = None
    total_pairs = 0
    total_skipped = 0

    for group_folder, label_name in groups.items():
        pairs, stats = find_paired_edf_paths(group_folder, data_dir)
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        n_pairs = len(pairs)
        if n_pairs == 0:
            print(f"  ⚠️  No closed+open pairs for {label_name} in {data_dir / group_folder}")
            if stats["only_closed"] or stats["only_open"] or stats["unmarked"]:
                print(f"      (only_closed={stats['only_closed']} only_open={stats['only_open']} unmarked={stats['unmarked']})")
            continue

        label_idx = class_names.index(label_name)
        print(
            f"  {label_name}: {n_pairs} pairs (only_closed={stats['only_closed']} only_open={stats['only_open']} unmarked={stats['unmarked']}) ..."
        )

        group_ok = 0
        for i, (p_closed, p_open) in enumerate(pairs):
            if (i + 1) % 100 == 0 or i == n_pairs - 1:
                print(f"    {i + 1}/{n_pairs}", end="", flush=True)
            try:
                raw_c = load_raw_edf_resilient(p_closed, preload=True, verbose=False)
                if not np.all(np.isfinite(raw_c.get_data())):
                    raise ValueError("Non-finite values in closed recording")
                out_c = extract_all_features(raw_c, canonical_ch_names, eyes_condition="closed")
                if canonical_ch_names is None:
                    feat_c, canonical_ch_names = out_c
                else:
                    feat_c = out_c

                raw_o = load_raw_edf_resilient(p_open, preload=True, verbose=False)
                if not np.all(np.isfinite(raw_o.get_data())):
                    raise ValueError("Non-finite values in open recording")
                feat_o = extract_all_features(raw_o, canonical_ch_names, eyes_condition="open")

                diff = feat_c[idx_c] - feat_o[idx_o]
                if not np.all(np.isfinite(diff)):
                    raise ValueError("Non-finite values in diff vector")

                X_list.append(diff)
                y_list.append(label_idx)
                group_ok += 1
            except Exception as e:
                print(f"\n    ⚠️  Pair failed {p_closed.name} / {p_open.name}: {e}")
                total_skipped += 1

        total_pairs += group_ok
        print(f" -> {group_ok} ok")

    if not X_list:
        raise FileNotFoundError(f"No paired EDF data under {data_dir}.")

    print(f"  Total pairs: {total_pairs}, skipped: {total_skipped}")
    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=int)
    return X, y, class_names, diff_names


# RF + SMOTE (как в train_rf_resampling_experiments, стратегия smote)
RF_SMOTE = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
}


def _smote_k_neighbors(y: np.ndarray) -> int:
    """SMOTE k_neighbors не больше, чем min_class_count - 1."""
    counts = np.bincount(y)
    m = int(counts[counts > 0].min()) if (counts > 0).any() else 0
    return max(1, min(5, m - 1)) if m >= 2 else 1


def _rf_smote_pipeline(y: np.ndarray) -> ImbPipeline:
    k = _smote_k_neighbors(y)
    return ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k)),
            (
                "rf",
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **RF_SMOTE),
            ),
        ]
    )


def _effective_cv_splits(y: np.ndarray, n_classes: int) -> int:
    """StratifiedKFold: в каждом тестовом фолде нужен ≥1 объект на класс → n_splits ≤ min count per class."""
    counts = [int(np.sum(y == i)) for i in range(n_classes)]
    m = min(counts) if counts else 0
    return min(N_SPLITS, m) if m >= 2 else 0


def cross_val_predict_rf_smote(X: np.ndarray, y: np.ndarray, n_splits: int) -> tuple[np.ndarray, float]:
    """Stratified K-fold; в каждом фолде свой SMOTE k_neighbors по обучающей выборке."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.empty_like(y)
    t0 = time.perf_counter()
    for train_idx, test_idx in cv.split(X, y):
        pipe = _rf_smote_pipeline(y[train_idx])
        pipe.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = pipe.predict(X[test_idx])
    elapsed = time.perf_counter() - t0
    return y_pred, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG closed−open feature difference dataset (data/ only).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max", type=int, default=None, help="Max pairs per class (debug)")
    parser.add_argument(
        "--train",
        action="store_true",
        help="RandomForest + SMOTE: stratified K-fold CV + fit на всех; см. --holdout",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Дополнительно stratified 80/20 hold-out (как раньше)",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    t0 = datetime.now()
    print("=" * 72)
    print("Eyes difference features: closed − open (common 364 features)")
    print(f"Started: {t0.isoformat()}")
    print(f"Data: {args.data_dir.resolve()}")
    idx_c, idx_o, common_names = common_feature_indices()
    print(f"Common features: {len(common_names)} (indices closed {idx_c.min()}..{idx_c.max()}, open {idx_o.min()}..{idx_o.max()})")
    print(feature_description(eyes_condition="open"))
    print("(Closed adds 12 alpha-topography features; they are excluded from the difference.)")
    print("=" * 72)

    X, y, class_names, diff_names = load_eyes_difference_features_and_labels(
        args.data_dir, max_pairs=args.max, groups=GROUPS
    )

    print(f"\nMatrix shape: {X.shape}; classes: {class_names}")
    for i, name in enumerate(class_names):
        print(f"  {name}: {(y == i).sum()}")

    if not args.train:
        print("\nDone (load only). Use --train for RF+SMOTE + CV / full-fit metrics.")
        return

    log_lines: list[str] = []

    def log(s: str) -> None:
        log_lines.append(s)
        print(s)

    n_cv = _effective_cv_splits(y, len(class_names))
    log("")
    log("--- Stratified K-fold CV (OOF), StandardScaler + SMOTE + RandomForest ---")
    if n_cv < 2:
        log(
            f"Skip CV: need at least 2 samples in the smallest class for stratified folds; "
            f"effective_splits={n_cv}, n={X.shape[0]}."
        )
    else:
        log(f"K={n_cv} (max {N_SPLITS} if class counts allow); SMOTE k_neighbors={_smote_k_neighbors(y)}")
        y_oof, cv_time = cross_val_predict_rf_smote(X, y, n_cv)
        log(f"CV wall time: {cv_time:.2f}s")
        log(
            f"OOF acc={(y_oof == y).mean():.4f}  bal_acc={balanced_accuracy_score(y, y_oof):.4f}  "
            f"macro_f1={f1_score(y, y_oof, average='macro', zero_division=0):.4f}  "
            f"weighted_f1={f1_score(y, y_oof, average='weighted', zero_division=0):.4f}"
        )
        log(classification_report(y, y_oof, target_names=class_names, zero_division=0))
        log(str(confusion_matrix(y, y_oof)))

    log("")
    log("--- Fit on ALL samples, evaluate on same set (in-sample; optimistic) ---")
    pipe_full = _rf_smote_pipeline(y)
    t_fit = time.perf_counter()
    pipe_full.fit(X, y)
    fit_s = time.perf_counter() - t_fit
    y_hat = pipe_full.predict(X)
    log(f"Train time: {fit_s:.2f}s")
    log(
        f"acc={(y_hat == y).mean():.4f}  bal_acc={balanced_accuracy_score(y, y_hat):.4f}  "
        f"macro_f1={f1_score(y, y_hat, average='macro', zero_division=0):.4f}  "
        f"weighted_f1={f1_score(y, y_hat, average='weighted', zero_division=0):.4f}"
    )
    log(classification_report(y, y_hat, target_names=class_names, zero_division=0))
    log(str(confusion_matrix(y, y_hat)))

    if args.holdout:
        log("")
        log("--- Hold-out (stratified 80/20), StandardScaler + SMOTE + RF ---")
        if X.shape[0] < 20:
            log(f"Skip hold-out: need at least 20 samples, got {X.shape[0]}.")
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            try:
                train_idx, test_idx = next(sss.split(X, y))
            except ValueError as e:
                log(f"Skip hold-out: {e}")
            else:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                pipe = _rf_smote_pipeline(y_train)
                t1 = time.perf_counter()
                pipe.fit(X_train, y_train)
                train_s = time.perf_counter() - t1
                y_pred = pipe.predict(X_test)
                log(f"Train time: {train_s:.2f}s")
                log(
                    f"acc={(y_pred == y_test).mean():.4f}  bal_acc={balanced_accuracy_score(y_test, y_pred):.4f}  "
                    f"macro_f1={f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
                )
                log(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
                log(str(confusion_matrix(y_test, y_pred)))

    log(f"Ended: {datetime.now().isoformat()}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    ts = t0.strftime("%Y%m%d_%H%M%S")
    out = args.results_dir / f"results_eyes_diff_rf_smote_{ts}.txt"
    out.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
