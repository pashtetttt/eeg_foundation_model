from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import ks_2samp, mannwhitneyu

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    """Kolmogorov–Smirnov: statistic D and two-sided p-value."""
    a0 = np.asarray(a, dtype=float).reshape(-1)
    b0 = np.asarray(b, dtype=float).reshape(-1)
    a0 = a0[np.isfinite(a0)]
    b0 = b0[np.isfinite(b0)]
    if a0.size < 2 or b0.size < 2 or not _HAS_SCIPY:
        return None, None
    r = ks_2samp(a0, b0, alternative="two-sided")
    return float(r.statistic), float(r.pvalue)


@dataclass(frozen=True)
class TestResult:
    p_mwu: float | None
    p_ks: float | None
    cohens_d: float
    mean_a: float
    mean_b: float
    n_a: int
    n_b: int
    n_drop_naninf: int


def _finite_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return 0.0
    ma = float(np.mean(a))
    mb = float(np.mean(b))
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = math.sqrt(max(1e-12, ((a.size - 1) * va + (b.size - 1) * vb) / max(1, (a.size + b.size - 2))))
    return (mb - ma) / pooled  # b - a


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """
    BH-FDR correction.
    Returns adjusted p-values (same shape).
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty_like(ranked)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty_like(p)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def run_tests(a: np.ndarray, b: np.ndarray) -> TestResult:
    """
    Nonparametric tests between a and b, with NaN/Inf filtering.
    Returns MWU and KS p-values (if SciPy available), plus Cohen's d and means.
    """
    a0 = np.asarray(a, dtype=float).reshape(-1)
    b0 = np.asarray(b, dtype=float).reshape(-1)
    ma = _finite_mask(a0)
    mb = _finite_mask(b0)
    dropped = int((~ma).sum() + (~mb).sum())
    a1 = a0[ma]
    b1 = b0[mb]
    if a1.size == 0 or b1.size == 0:
        return TestResult(None, None, 0.0, float("nan"), float("nan"), int(a1.size), int(b1.size), dropped)

    p_mwu = None
    p_ks = None
    if _HAS_SCIPY:
        # two-sided
        p_mwu = float(mannwhitneyu(a1, b1, alternative="two-sided").pvalue)
        p_ks = float(ks_2samp(a1, b1, alternative="two-sided").pvalue)

    d = float(cohens_d(a1, b1))
    return TestResult(
        p_mwu=p_mwu,
        p_ks=p_ks,
        cohens_d=d,
        mean_a=float(np.mean(a1)),
        mean_b=float(np.mean(b1)),
        n_a=int(a1.size),
        n_b=int(b1.size),
        n_drop_naninf=dropped,
    )

