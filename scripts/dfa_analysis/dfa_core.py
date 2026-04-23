from __future__ import annotations

"""
DFA core implementation.

Предпочтительно использовать библиотеку fathon, но она может отсутствовать в окружении
(GitHub-only). Поэтому здесь есть:
- попытка использовать fathon (если установлен)
- fallback: чистая реализация DFA (Peng et al.) с polynomial detrending.
"""

import numpy as np


def _try_fathon_alpha(x: np.ndarray, min_scale: int, max_scale: int, order: int) -> float | None:
    try:
        import fathon  # type: ignore
    except Exception:
        return None
    # fathon API varies; we keep it best-effort and fall back if it fails.
    try:
        from fathon import DFA  # type: ignore

        dfa = DFA(x)
        # common: dfa.computeFlucVec(scales, order) then dfa.fitFlucVec()
        scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), num=20).astype(int))
        dfa.computeFlucVec(scales, order=order)
        # slope/intercept from linear fit on log-log
        alpha, _ = dfa.fitFlucVec()
        alpha = float(alpha)
        return alpha if np.isfinite(alpha) else None
    except Exception:
        return None


def _poly_detrend(y: np.ndarray, order: int) -> np.ndarray:
    t = np.arange(y.size, dtype=float)
    coef = np.polyfit(t, y, deg=order)
    trend = np.polyval(coef, t)
    return y - trend


def dfa_alpha(
    x: np.ndarray,
    *,
    min_scale: int = 4,
    max_scale: int | None = None,
    order: int = 1,
    n_scales: int = 20,
) -> float:
    """
    Возвращает альфа-экспоненту DFA.

    x: 1D signal
    min_scale: минимальный размер окна (в сэмплах)
    max_scale: максимальный размер окна (по умолчанию len(x)//10)
    order: порядок полинома для детрендинга (1 или 2)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 200:
        return float("nan")

    if max_scale is None:
        max_scale = max(min_scale + 1, x.size // 10)
    max_scale = int(max_scale)
    min_scale = int(min_scale)
    if max_scale <= min_scale + 1:
        return float("nan")

    # Try fathon first (if present)
    alpha = _try_fathon_alpha(x, min_scale=min_scale, max_scale=max_scale, order=order)
    if alpha is not None:
        return alpha

    # Fallback DFA implementation
    x = x - float(np.mean(x))
    y = np.cumsum(x)  # profile
    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), num=n_scales).astype(int))
    scales = scales[scales >= min_scale]
    scales = scales[scales <= max_scale]
    if scales.size < 6:
        return float("nan")

    F = []
    S = []
    for s in scales.tolist():
        n = y.size // s
        if n < 4:
            continue
        seg = y[: n * s].reshape(n, s)
        # detrend each window
        rms = []
        for i in range(n):
            detr = _poly_detrend(seg[i], order=order)
            rms.append(float(np.sqrt(np.mean(detr * detr))))
        f = float(np.sqrt(np.mean(np.square(rms))))
        if np.isfinite(f) and f > 0:
            F.append(f)
            S.append(s)

    if len(F) < 6:
        return float("nan")

    logS = np.log(np.asarray(S, dtype=float))
    logF = np.log(np.asarray(F, dtype=float))
    if not (np.all(np.isfinite(logS)) and np.all(np.isfinite(logF))):
        return float("nan")

    # slope in log-log domain
    slope = float(np.polyfit(logS, logF, deg=1)[0])
    return slope if np.isfinite(slope) else float("nan")


def dfa_backend_name() -> str:
    try:
        import fathon  # type: ignore

        return "fathon"
    except Exception:
        return "fallback_numpy"

