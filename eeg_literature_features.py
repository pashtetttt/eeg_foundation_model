"""
Literature-based EEG features (Vandenbosch et al. 2019; van Noordt & Willoughby 2021).

- 1 Hz PSD bins 1–24 Hz (per canonical channel), Welch 2 s windows
- Multiscale entropy (MSE): coarse-graining + sample entropy (antropy), scales 1–20
- PSD slope (1–30 Hz) in log–log space with robust regression (Huber)
- Phase-shuffled surrogate nonlinearity index (MSE_surrogate − MSE_original), regional

Runtime for surrogate scales with EEG_LITERATURE_SURROGATE_ITERS (default 25; set 100 for paper-style).
"""

from __future__ import annotations

import logging
import os
import time
import numpy as np
from scipy.signal import butter, filtfilt, welch
from sklearn.linear_model import HuberRegressor

try:
    import antropy as ant

    _ANTROPY_AVAILABLE = True
except ImportError:
    _ANTROPY_AVAILABLE = False

logger = logging.getLogger(__name__)

N_CHANNELS = 19
N_1HZ_BINS = 24
MSE_SCALES = tuple(range(1, 21))
MSE_SHORT = tuple(range(1, 6))
MSE_MEDIUM = tuple(range(6, 11))
MSE_LONG = tuple(range(15, 21))
MSE_MIN_SEGMENT_S = 20.0
MSE_M = 2
MSE_R_STD_FACTOR = 0.5

# Welch / PSD
WELCH_NPERSEG_FACTOR = 2.0  # nperseg = int(fs * this)
WELCH_NOVERLAP_FACTOR = 1.0  # noverlap = int(fs * this)
BP_LIT_LOW = 1.0
BP_LIT_HIGH = 30.0
SLOPE_FMIN = 1.0
SLOPE_FMAX = 30.0

SURROGATE_ITERS_DEFAULT = int(os.environ.get("EEG_LITERATURE_SURROGATE_ITERS", "25"))

REGION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "frontal": ("Fp", "F3", "F4", "F7", "F8", "Fz"),
    "central": ("C3", "C4", "Cz"),
    "parietal": ("P3", "P4", "Pz"),
    "occipital": ("O1", "O2"),
}


def _bandpass(signal: np.ndarray, fs: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    lo = max(0.01, fmin / nyq)
    hi = min(0.99, fmax / nyq)
    if lo >= hi:
        return signal
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, signal)


def _channel_regions(canonical_ch_names: list[str]) -> dict[str, list[int]]:
    """Map region name -> list of channel indices 0..n-1 matching canonical names."""
    out: dict[str, list[int]] = {k: [] for k in REGION_KEYWORDS}
    for i, name in enumerate(canonical_ch_names[:N_CHANNELS]):
        if not name:
            continue
        for reg, kws in REGION_KEYWORDS.items():
            if any(kw in name for kw in kws):
                out[reg].append(i)
                break
    return out


def _integrate_psd_bin(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    idx = (freqs >= lo) & (freqs < hi)
    if not np.any(idx):
        return 0.0
    f = freqs[idx]
    p = np.asarray(psd[idx], dtype=float)
    return float(np.trapezoid(p, f))


def power_1hz_bins(
    signal: np.ndarray,
    fs: float,
) -> dict[str, float]:
    """
    Welch PSD (2 s segments, 1 s overlap), band-pass 1–30 Hz, integrate 1 Hz bins 1–2 … 23–24.
    Returns keys pow1hz_bin{k}_ch format filled in flatten order by caller.
    """
    sig = np.asarray(signal, dtype=float).ravel()
    if sig.size < int(fs * 2):
        return {f"pow1hz_bin{b+1}": 0.0 for b in range(N_1HZ_BINS)}
    sig = _bandpass(sig, fs, BP_LIT_LOW, BP_LIT_HIGH)
    nperseg = max(8, int(fs * WELCH_NPERSEG_FACTOR))
    noverlap = min(nperseg - 1, int(fs * WELCH_NOVERLAP_FACTOR))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="density")
    out: dict[str, float] = {}
    for b in range(N_1HZ_BINS):
        lo = float(b + 1)
        hi = lo + 1.0
        out[f"pow1hz_bin{b+1}"] = _integrate_psd_bin(freqs, psd, lo, hi)
    return out


def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if scale < 1:
        return x.copy()
    n = (x.size // scale) * scale
    if n < scale * (MSE_M + 2):
        return np.array([], dtype=float)
    y = x[:n].reshape(-1, scale).mean(axis=1)
    return y


def _sample_entropy_safe(x: np.ndarray, tolerance: float) -> float:
    if not _ANTROPY_AVAILABLE or x.size < (MSE_M + 3):
        return float("nan")
    try:
        return float(ant.sample_entropy(x, order=MSE_M, tolerance=tolerance, metric="chebyshev"))
    except Exception as e:
        logger.warning("sample_entropy failed: %s", e)
        return float("nan")


def multiscale_entropy_profile(
    signal: np.ndarray,
    fs: float,
    scales: tuple[int, ...] = MSE_SCALES,
) -> dict[int, float]:
    """MSE per scale (coarse-grain then sample entropy). tolerance = MSE_R_STD_FACTOR * std(raw)."""
    sig = np.asarray(signal, dtype=float).ravel()
    std0 = float(np.std(sig))
    if std0 < 1e-12:
        return {s: 0.0 for s in scales}
    tol = MSE_R_STD_FACTOR * std0
    out: dict[int, float] = {}
    for s in scales:
        cg = _coarse_grain(sig, s)
        se = _sample_entropy_safe(cg, tol)
        if not np.isfinite(se):
            out[s] = 0.0
            logger.debug("MSE scale %s replaced with 0 (non-finite sample entropy)", s)
        else:
            out[s] = float(se)
    return out


def _mean_scales(profile: dict[int, float], which: tuple[int, ...]) -> float:
    vals = [profile.get(s, 0.0) for s in which]
    v = np.array(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(np.mean(v))


def _aggregate_mse_region(
    profiles_per_ch: list[dict[int, float]],
) -> dict[str, float]:
    """Average channel profiles inside region (caller passes only channels for that region)."""
    if not profiles_per_ch:
        return {"mse_short": 0.0, "mse_medium": 0.0, "mse_long": 0.0}
    shorts, meds, longs = [], [], []
    for prof in profiles_per_ch:
        shorts.append(_mean_scales(prof, MSE_SHORT))
        meds.append(_mean_scales(prof, MSE_MEDIUM))
        longs.append(_mean_scales(prof, MSE_LONG))
    return {
        "mse_short": float(np.mean(shorts)),
        "mse_medium": float(np.mean(meds)),
        "mse_long": float(np.mean(longs)),
    }


def _iter_analysis_segments(x: np.ndarray, fs: float, seg_s: float = MSE_MIN_SEGMENT_S):
    n = x.size
    seg = int(seg_s * fs)
    if n < seg:
        yield x.astype(float)
        return
    for start in range(0, n - seg + 1, seg):
        yield x[start : start + seg].astype(float)


def _mse_profiles_averaged_segments(x: np.ndarray, fs: float) -> dict[int, float]:
    profs: list[dict[int, float]] = []
    for seg in _iter_analysis_segments(x, fs):
        profs.append(multiscale_entropy_profile(seg, fs))
    acc = {s: [] for s in MSE_SCALES}
    for p in profs:
        for s in MSE_SCALES:
            acc[s].append(p.get(s, 0.0))
    return {s: float(np.mean(acc[s])) if acc[s] else 0.0 for s in MSE_SCALES}


def psd_slope(
    signal: np.ndarray,
    fs: float,
    fmin: float = SLOPE_FMIN,
    fmax: float = SLOPE_FMAX,
) -> tuple[float, float]:
    """Huber regression on log10(freq) vs log10(PSD). Returns (slope, r2)."""
    sig = np.asarray(signal, dtype=float).ravel()
    if sig.size < int(fs):
        return 0.0, 0.0
    sig = _bandpass(sig, fs, BP_LIT_LOW, BP_LIT_HIGH)
    nperseg = max(8, min(int(fs * 2), sig.size // 2))
    noverlap = nperseg // 2
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="density")
    mask = (freqs >= fmin) & (freqs <= fmax) & (psd > 0) & np.isfinite(psd)
    if np.sum(mask) < 8:
        return 0.0, 0.0
    xf = np.log10(freqs[mask]).reshape(-1, 1)
    yf = np.log10(psd[mask])
    try:
        reg = HuberRegressor(alpha=0.0, max_iter=200)
        reg.fit(xf, yf)
        slope = float(reg.coef_[0])
        y_pred = reg.predict(xf)
        ss_res = float(np.sum((yf - y_pred) ** 2))
        ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return slope, r2
    except Exception as e:
        logger.warning("psd_slope regression failed: %s", e)
        return 0.0, 0.0


def _phase_shuffle(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = x.size
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    ph = np.angle(spec)
    k = mag.size
    new_ph = np.zeros(k, dtype=float)
    new_ph[0] = ph[0]
    if k > 1:
        new_ph[1:] = rng.uniform(0.0, 2.0 * np.pi, size=k - 1)
    sur = mag * np.exp(1j * new_ph)
    return np.fft.irfft(sur, n=n)


def phase_shuffled_nonlinearity(
    signal: np.ndarray,
    fs: float,
    n_iterations: int = SURROGATE_ITERS_DEFAULT,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Mean over iterations of (mean MSE scales 1–20 on surrogate − same on original), single channel.
    """
    rng = rng or np.random.default_rng(0)
    sig = np.asarray(signal, dtype=float).ravel()
    if sig.size < int(fs * 2):
        return 0.0
    orig_prof = _mse_profiles_averaged_segments(sig, fs)
    orig_mean = float(np.mean([orig_prof[s] for s in MSE_SCALES]))

    diffs: list[float] = []
    for _ in range(max(1, n_iterations)):
        sur = _phase_shuffle(sig, rng)
        sur_prof = _mse_profiles_averaged_segments(sur, fs)
        sur_mean = float(np.mean([sur_prof[s] for s in MSE_SCALES]))
        diffs.append(sur_mean - orig_mean)
    return float(np.mean(diffs))


def literature_feature_count() -> int:
    """Fixed layout size for stacking vectors."""
    return N_CHANNELS * N_1HZ_BINS + 4 * 3 + 4 * 2 + 4


def get_literature_feature_names() -> list[str]:
    names: list[str] = []
    for ch in range(N_CHANNELS):
        for b in range(1, N_1HZ_BINS + 1):
            names.append(f"lit_pow1hz_bin{b}_ch{ch}")
    for reg in ("frontal", "central", "parietal", "occipital"):
        names.append(f"lit_mse_short_{reg}")
        names.append(f"lit_mse_medium_{reg}")
        names.append(f"lit_mse_long_{reg}")
    for reg in ("frontal", "central", "parietal", "occipital"):
        names.append(f"lit_psd_slope_{reg}")
        names.append(f"lit_psd_slope_r2_{reg}")
    for reg in ("frontal", "central", "parietal", "occipital"):
        names.append(f"lit_nonlin_index_{reg}")
    return names


def extract_complexity_features(
    data: np.ndarray,
    fs: float,
    canonical_ch_names: list[str],
    surrogate_iters: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Full literature block for one recording. data shape (n_ch, n_times), picks aligned to canonical.

    Returns (feature_dict, timing_seconds_per_block).
    """
    timings: dict[str, float] = {}
    feats: dict[str, float] = {}
    n_ch = min(N_CHANNELS, data.shape[0])
    surrogate_iters = surrogate_iters if surrogate_iters is not None else SURROGATE_ITERS_DEFAULT
    rng = rng or np.random.default_rng(42)

    regions = _channel_regions(canonical_ch_names)

    t0 = time.perf_counter()
    for ch in range(N_CHANNELS):
        if ch < n_ch:
            pb = power_1hz_bins(data[ch], fs)
            for b in range(1, N_1HZ_BINS + 1):
                feats[f"lit_pow1hz_bin{b}_ch{ch}"] = pb[f"pow1hz_bin{b}"]
        else:
            for b in range(1, N_1HZ_BINS + 1):
                feats[f"lit_pow1hz_bin{b}_ch{ch}"] = 0.0
    timings["power_1hz_bins"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mse_by_reg: dict[str, list[dict[int, float]]] = {r: [] for r in REGION_KEYWORDS}
    for ch in range(n_ch):
        prof = _mse_profiles_averaged_segments(data[ch], fs)
        placed = False
        for reg, idxs in regions.items():
            if ch in idxs:
                mse_by_reg[reg].append(prof)
                placed = True
                break
        if not placed:
            pass
    for reg in REGION_KEYWORDS:
        agg = _aggregate_mse_region(mse_by_reg[reg])
        feats[f"lit_mse_short_{reg}"] = agg["mse_short"]
        feats[f"lit_mse_medium_{reg}"] = agg["mse_medium"]
        feats[f"lit_mse_long_{reg}"] = agg["mse_long"]
    timings["multiscale_entropy"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    slope_ch: dict[str, list[tuple[float, float]]] = {r: [] for r in REGION_KEYWORDS}
    for ch in range(n_ch):
        sl, r2 = psd_slope(data[ch], fs)
        placed = False
        for reg, idxs in regions.items():
            if ch in idxs:
                slope_ch[reg].append((sl, r2))
                placed = True
                break
        if not placed:
            pass
    for reg in REGION_KEYWORDS:
        if slope_ch[reg]:
            slopes = [t[0] for t in slope_ch[reg]]
            r2s = [t[1] for t in slope_ch[reg]]
            feats[f"lit_psd_slope_{reg}"] = float(np.mean(slopes))
            feats[f"lit_psd_slope_r2_{reg}"] = float(np.mean(r2s))
        else:
            feats[f"lit_psd_slope_{reg}"] = 0.0
            feats[f"lit_psd_slope_r2_{reg}"] = 0.0
    timings["psd_slope"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    nonlin_by_ch: dict[str, list[float]] = {r: [] for r in REGION_KEYWORDS}
    for ch in range(n_ch):
        idx = phase_shuffled_nonlinearity(data[ch], fs, n_iterations=surrogate_iters, rng=rng)
        placed = False
        for reg, idxs in regions.items():
            if ch in idxs:
                nonlin_by_ch[reg].append(idx)
                placed = True
                break
        if not placed:
            pass
    for reg in REGION_KEYWORDS:
        if nonlin_by_ch[reg]:
            feats[f"lit_nonlin_index_{reg}"] = float(np.mean(nonlin_by_ch[reg]))
        else:
            feats[f"lit_nonlin_index_{reg}"] = 0.0
    timings["phase_shuffled_surrogate"] = time.perf_counter() - t0

    return feats, timings


def literature_feature_vector_from_raw(
    raw,
    picks: list[int],
    canonical_ch_names: list[str],
    surrogate_iters: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Load picked EEG matrix and return stacked vector + timings."""
    data, _ = raw.get_data(picks=picks, return_times=True)
    fs = float(raw.info["sfreq"])
    d = np.asarray(data, dtype=float)
    feats, timings = extract_complexity_features(
        d, fs, canonical_ch_names, surrogate_iters=surrogate_iters, rng=rng
    )
    if os.environ.get("EEG_LITERATURE_LOG_TIMES", "").strip() == "1":
        for block, dt in timings.items():
            print(f"  [literature timing] {block}: {dt:.3f}s", flush=True)
    order = get_literature_feature_names()
    vec = np.array([feats.get(k, 0.0) for k in order], dtype=float)
    return vec, timings
