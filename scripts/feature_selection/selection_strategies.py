from __future__ import annotations

"""
Стратегии отбора признаков для сравнительных экспериментов.

ВАЖНО (про утечку):
- Exp3 (Representative) и Exp4 (Top-N Importance/RFE) должны выбирать признаки ТОЛЬКО на train-fold.
- Поэтому функции ниже разделены на fit (на train) и apply (на train/val) части.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    X_train: np.ndarray
    X_val: np.ndarray
    feature_names: list[str]
    # indices into original feature space; None if transformation is not a subset (e.g. aggregation)
    selected_indices: np.ndarray | None


def baseline_all_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: list[str],
) -> SelectionResult:
    return SelectionResult(X_train=X_train, X_val=X_val, feature_names=feature_names, selected_indices=np.arange(X_train.shape[1], dtype=int))


def cluster_aggregation_median(
    X_train: np.ndarray,
    X_val: np.ndarray,
    clusters_dict: dict[str, list[int]],
    feature_names: list[str] | None = None,
) -> SelectionResult:
    # deterministic cluster order
    keys = sorted(clusters_dict.keys())
    Xt = np.zeros((X_train.shape[0], len(keys)), dtype=float)
    Xv = np.zeros((X_val.shape[0], len(keys)), dtype=float)
    for j, k in enumerate(keys):
        cols = clusters_dict[k]
        Xt[:, j] = np.nanmedian(X_train[:, cols], axis=1)
        Xv[:, j] = np.nanmedian(X_val[:, cols], axis=1)
    # aggregated features are clusters
    return SelectionResult(X_train=Xt, X_val=Xv, feature_names=keys, selected_indices=None)


def _safe_abs_corr_with_target(x: np.ndarray, y: np.ndarray) -> float:
    """
    Абсолютная корреляция Пирсона между признаком и y (метка класса как число).
    Это эвристика для Exp3; считается на train-fold, чтобы не было утечки.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(r):
        return 0.0
    return abs(r)


def cluster_representative_one_feature(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    clusters_dict: dict[str, list[int]],
    feature_names: list[str],
    criterion: str = "variance",  # variance | corr_y
) -> SelectionResult:
    """
    Exp3: выбираем один признак из каждого кластера.
    criterion:
      - variance: max variance on train
      - corr_y: max |corr(feature, y)| on train
    """
    crit = criterion.strip().lower()
    if crit not in ("variance", "corr_y"):
        raise ValueError("criterion must be 'variance' or 'corr_y'")

    keys = sorted(clusters_dict.keys())
    chosen: list[int] = []
    chosen_names: list[str] = []
    for k in keys:
        cols = clusters_dict[k]
        if not cols:
            continue
        if crit == "variance":
            v = np.nanvar(X_train[:, cols], axis=0)
            idx_local = int(np.nanargmax(v))
        else:
            scores = [(_safe_abs_corr_with_target(X_train[:, c], y_train), c) for c in cols]
            scores.sort(key=lambda t: (-t[0], t[1]))
            idx_local = cols.index(scores[0][1])
        c = cols[idx_local]
        chosen.append(int(c))
        chosen_names.append(feature_names[int(c)])

    chosen_idx = np.array(chosen, dtype=int)
    Xt = X_train[:, chosen_idx]
    Xv = X_val[:, chosen_idx]
    return SelectionResult(X_train=Xt, X_val=Xv, feature_names=chosen_names, selected_indices=chosen_idx)


def topn_by_xgb_importance(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: list[str],
    n_keep: int,
    random_state: int = 42,
) -> SelectionResult:
    """
    Exp4: отбор по важности XGBoost (на train-fold), игнорируя кластеры.

    Важно: это не RFE (дороже), но по смыслу соответствует 'Feature Importance → Top-N'.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    if n_keep <= 0:
        raise ValueError("n_keep must be > 0")
    n_keep = int(min(n_keep, X_train.shape[1]))

    # небольшая модель для важностей (можно заменить на xgb_regularized при желании)
    clf = XGBClassifier(
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
        n_estimators=250,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        reg_lambda=2.0,
        reg_alpha=0.1,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    sw = compute_sample_weight("balanced", y_train)
    pipe.fit(X_train, y_train, clf__sample_weight=sw)

    importances = pipe.named_steps["clf"].feature_importances_
    if importances is None or len(importances) != X_train.shape[1]:
        # fallback: keep first n_keep
        chosen_idx = np.arange(n_keep, dtype=int)
    else:
        order = np.argsort(-importances)
        chosen_idx = np.sort(order[:n_keep].astype(int))

    chosen_names = [feature_names[i] for i in chosen_idx.tolist()]
    return SelectionResult(
        X_train=X_train[:, chosen_idx],
        X_val=X_val[:, chosen_idx],
        feature_names=chosen_names,
        selected_indices=chosen_idx,
    )

