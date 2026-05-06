"""
Detrended Fluctuation Analysis (DFA) for EEG amplitude envelopes.

References
----------
- Hardstone et al. (2012), Frontiers in Physiology — DFA / LRTC on amplitude envelopes
  of band-limited neural signals (oscillatory dynamics).
- Peng et al. (1994) — classic DFA scaling; integration + segmented detrending.

We use DFA-1 style: mean-centered integrated profile Y, segment-wise polynomial detrend
(linear for polynomial_order=1), fluctuation F(n) as RMS of the detrended residual
within segments, then estimate scaling exponent alpha as the slope of log F vs log n
over a range of box sizes n (here: scales in samples).

Notes
-----
- Very short signals or scales that leave too few boxes yield NaN; callers should replace.
- Band-limited envelopes reduce filter-induced broadband leakage; still avoid interpreting
  alpha at scales smaller than ~10 cycles of the band center frequency when possible.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal import detrend as signal_detrend


def bandpass_envelope(
    sig: np.ndarray,
    sfreq: float,
    f_lo: float,
    f_hi: float,
    order: int = 4,
) -> np.ndarray:
    """
    Analytic amplitude envelope |Hilbert(bandpass(sig))|.

    Parameters
    ----------
    sig : (n_samples,) float
    sfreq : sampling frequency
    f_lo, f_hi : band edges in Hz
    """
    sig = np.asarray(sig, dtype=float).ravel()
    if len(sig) < 16:
        return np.zeros_like(sig)
    nyq = 0.5 * sfreq
    lo = max(f_lo / nyq, 1e-4)
    hi = min(f_hi / nyq, 0.99)
    if hi <= lo:
        return np.zeros_like(sig)
    b, a = butter(order, [lo, hi], btype="band")
    try:
        xf = filtfilt(b, a, sig)
    except ValueError:
        return np.zeros_like(sig)
    env = np.abs(hilbert(xf))
    return env


def dfa_exponent(
    series: np.ndarray,
    scales: Iterable[int] | np.ndarray,
    *,
    poly_order: int = 1,
    min_segments: int = 4,
) -> float:
    """
    Estimate DFA scaling exponent (alpha) from a 1D real-valued series.

    Steps
    -----
    1. Subtract mean and form cumulative sum (integration).
    2. For each scale n (segment length in samples), split the profile into disjoint
       segments of length n (truncate remainder).
    3. Detrend each segment with a polynomial of degree `poly_order` (Hardstone-style
       linear detrend is poly_order=1).
    4. F(n) = sqrt(mean of squared residuals pooled over all segments).
    5. Fit log10 F vs log10 n with least squares; slope is alpha.

    Returns NaN if insufficient data.
    """
    x = np.asarray(series, dtype=float).ravel()
    if x.size < 32 or not np.all(np.isfinite(x)):
        return float("nan")
    x = x - np.mean(x)
    y = np.cumsum(x)
    scales = np.asarray(list(scales), dtype=int)
    scales = scales[(scales >= 4) & (scales < len(y) // min_segments)]
    if scales.size < 3:
        return float("nan")

    log_n = []
    log_f = []
    for n in scales:
        nseg = len(y) // n
        if nseg < min_segments:
            continue
        segs = y[: nseg * n].reshape(nseg, n)
        # Fast path for Hardstone-style DFA-1 (linear detrending): vectorized detrend.
        # This avoids per-segment np.polyfit calls, which are very slow on long signals.
        if poly_order == 1:
            resid = signal_detrend(segs, axis=1, type="linear")
            mse = np.mean(resid * resid, axis=1)
            f_n = float(np.sqrt(np.mean(mse)))
        else:
            fluct = []
            t = np.arange(n, dtype=float)
            for k in range(nseg):
                seg = segs[k]
                coeffs = np.polyfit(t, seg, deg=poly_order)
                fit = np.polyval(coeffs, t)
                fluct.append(np.mean((seg - fit) ** 2))
            if not fluct:
                continue
            f_n = float(np.sqrt(np.mean(fluct)))
        if f_n <= 0:
            continue
        log_n.append(np.log10(float(n)))
        log_f.append(np.log10(f_n))

    if len(log_n) < 3:
        return float("nan")
    log_n = np.asarray(log_n)
    log_f = np.asarray(log_f)
    # linear regression slope
    slope, _intercept = np.polyfit(log_n, log_f, 1)
    return float(slope)


# Default 19-channel indices (same order as eeg_thesis DEFAULT_19_CH)
REGION_FRONTAL = (0, 1, 2, 3, 4, 5, 6)
REGION_CENTRAL = (7, 8, 9, 10, 11)
REGION_PARIETAL = (12, 13, 14, 15, 16)
REGION_OCCIPITAL = (17, 18)
REGIONS = {
    "frontal": REGION_FRONTAL,
    "central": REGION_CENTRAL,
    "parietal": REGION_PARIETAL,
    "occipital": REGION_OCCIPITAL,
}


def compute_dfa_feature_block(
    data_ct: np.ndarray,
    sfreq: float,
    *,
    scales: Iterable[int] | None = None,
    poly_order: int = 1,
    bands_hz: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    DFA exponents on amplitude envelopes for theta / alpha / beta per channel and per region.

    Parameters
    ----------
    data_ct : (n_channels, n_samples)
    sfreq : float
    scales : iterable of segment lengths in samples; default 1..20 filtered by signal length
    poly_order : detrending polynomial degree (1 = linear, Hardstone-style)
    bands_hz : band definitions; default theta/alpha/beta

    Returns
    -------
    vec : 1D concatenated features
    names : feature names (dfa_{band}_ch{k}, dfa_{band}_region_{name})
    """
    if bands_hz is None:
        bands_hz = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
    if scales is None:
        scales = list(range(1, 21))

    data_ct = np.asarray(data_ct, dtype=float)
    n_ch, n_t = data_ct.shape
    scales_arr = np.array([s for s in scales if isinstance(s, (int, np.integer)) and s >= 4], dtype=int)

    vec: list[float] = []
    names: list[str] = []

    for band_name, (lo, hi) in bands_hz.items():
        # Compute envelopes once per channel, then reuse for both per-channel and per-region DFA.
        # This avoids repeated filtering/Hilbert calls and cuts DFA wall time significantly.
        env_by_ch = [bandpass_envelope(data_ct[ch], sfreq, lo, hi) for ch in range(n_ch)]

        ch_alphas: list[float] = []
        for ch, env in enumerate(env_by_ch):
            a = dfa_exponent(env, scales_arr, poly_order=poly_order)
            ch_alphas.append(float(a) if np.isfinite(a) else 0.0)
            names.append(f"dfa_{band_name}_ch{ch}")
        vec.extend(ch_alphas)

        for reg_name, idxs in REGIONS.items():
            idxs = [i for i in idxs if i < n_ch]
            if not idxs:
                vec.append(0.0)
                names.append(f"dfa_{band_name}_region_{reg_name}")
                continue
            reg_env = np.mean([env_by_ch[i] for i in idxs], axis=0)
            a_r = dfa_exponent(reg_env, scales_arr, poly_order=poly_order)
            vec.append(float(a_r) if np.isfinite(a_r) else 0.0)
            names.append(f"dfa_{band_name}_region_{reg_name}")

    out = np.asarray(vec, dtype=float)
    if np.any(~np.isfinite(out)):
        warnings.warn("DFA block contained non-finite values; replaced with 0.", RuntimeWarning, stacklevel=2)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, names
