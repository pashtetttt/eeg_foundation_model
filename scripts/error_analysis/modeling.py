from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


XGB_REGULARIZED_PARAMS = {
    "n_estimators": 250,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_lambda": 2.0,
    "reg_alpha": 0.1,
}


class CorrelationFilterByVariance(TransformerMixin, BaseEstimator):
    """
    Fit-time correlation filter (Pearson) that drops one feature from each highly correlated pair
    (|r| > threshold), keeping the one with higher variance.

    IMPORTANT: must be fit only on training data (no leakage).
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = float(threshold)
        self.keep_indices_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None):
        X = np.asarray(X, dtype=float)
        # guard against NaNs/Infs affecting corr
        X = np.where(np.isfinite(X), X, np.nan)
        # variance ignoring NaN
        var = np.nanvar(X, axis=0)
        C = np.corrcoef(np.nan_to_num(X, nan=0.0).T)
        # constant columns -> NaN correlations; treat as 0 (no actionable correlation)
        if np.any(~np.isfinite(C)):
            C = np.where(np.isfinite(C), C, 0.0)
            np.fill_diagonal(C, 1.0)
        n = C.shape[0]

        keep = np.ones(n, dtype=bool)
        # process pairs by descending |r|
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                r = C[i, j]
                if np.isfinite(r) and abs(r) > self.threshold:
                    pairs.append((abs(float(r)), i, j))
        pairs.sort(key=lambda t: -t[0])

        for _abs_r, i, j in pairs:
            if not keep[i] or not keep[j]:
                continue
            # drop lower variance (ties -> drop j)
            if var[i] >= var[j]:
                keep[j] = False
            else:
                keep[i] = False

        self.keep_indices_ = np.where(keep)[0].astype(int)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.keep_indices_ is None:
            raise RuntimeError("CorrelationFilterByVariance is not fit yet")
        return np.asarray(X)[:, self.keep_indices_]


def build_xgb_regularized_pipeline(
    *,
    use_corr_filter: bool,
    corr_threshold: float = 0.95,
    random_state: int = 42,
) -> Pipeline:
    steps: list[tuple[str, object]] = []
    steps.append(("scaler", StandardScaler()))
    if use_corr_filter:
        steps.append(("corr_filter", CorrelationFilterByVariance(threshold=corr_threshold)))
    steps.append(
        (
            "clf",
            XGBClassifier(
                random_state=random_state,
                n_jobs=-1,
                eval_metric="mlogloss",
                **XGB_REGULARIZED_PARAMS,
            ),
        )
    )
    return Pipeline(steps)


def fit_xgb_balanced(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    sw = compute_sample_weight("balanced", y)
    pipe.fit(X, y, clf__sample_weight=sw)

