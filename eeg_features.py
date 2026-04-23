"""
EEG feature extraction for age-group classification.

Features (per recording, per canonical 19 EEG channels):
- Band powers (76) + band-power ratios (76)
- Spectral centroids (19) + entropies (38)
- Envelope frequencies (76) + Higuchi fractal dimensions (19)
- Hjorth complexities (19) + alpha-variability (mean/std, 38)
- Regional theta1/theta2 ratios (3)
- Alpha-rhythm topography (12, closed-eyes only): alpha (8-13Hz) + predecessor (4-12Hz) - regional powers and ratios for frontal, central, parieto-occipital

Total: 364 scalar features (open-eyes) or 376 (closed-eyes).
If optional libraries are missing (antropy, nolds) the corresponding slots are filled with zeros
so the feature length stays constant.

Why N_CHANNELS = 19: Standard 10-20 EEG montage often has 19 channels (Fp1, Fp2, F7, F3, Fz, ...).
Actual channel count varies by file; we fix to 19 so every sample has the same feature length:
we use the first file's channel order (up to 19) and pad with zeros if a file has fewer channels.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert, welch

try:
    import antropy as ant
    _ANTROPY_AVAILABLE = True
except ImportError:
    _ANTROPY_AVAILABLE = False

try:
    import nolds
    _NOLDS_AVAILABLE = True
except ImportError:
    _NOLDS_AVAILABLE = False

from eeg_literature_features import get_literature_feature_names, literature_feature_count, literature_feature_vector_from_raw

# Band-power features: 19 channels × 4 bands = 76
N_CHANNELS = 19
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

# For spectral centroid
CENTROID_FMIN = 0.5
CENTROID_FMAX = 45.0

# Time-domain segment length to control runtime of entropies / fractal features
TIME_SEGMENT_SAMPLES = 5000

# Envelope frequency search range
ENV_FREQ_MIN = 0.5
ENV_FREQ_MAX = 4.0

# Higuchi settings
HFD_KMAX = 10

# Alpha-variability band (Hz)
ALPHA_VAR_LO = 4.0
ALPHA_VAR_HI = 13.0
ALPHA_VAR_MIN_INT = 0.083  # s (~12 Hz)
ALPHA_VAR_MAX_INT = 0.25   # s (~4 Hz)


def spectral_centroid(freqs: np.ndarray, psd: np.ndarray, fmin: float = 0.5, fmax: float = 45.0) -> float:
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return 0.0
    f = freqs[idx]
    p = np.asarray(psd[idx], dtype=float).flatten()
    return float(np.sum(f * p) / (np.sum(p) + 1e-10))


def _band_power_arrays(raw, picks: list[int], n_ch: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (flat_76, dict of band_name -> array length n_ch)."""
    band_arrays = {}
    flat = []
    for band_name, (fmin, fmax) in BANDS.items():
        psd = raw.compute_psd(picks=picks, fmin=fmin, fmax=fmax, verbose=False)
        data = psd.get_data()
        arr = data.mean(axis=tuple(range(1, data.ndim)))
        if len(arr) < n_ch:
            arr = np.pad(arr, (0, n_ch - len(arr)), constant_values=0.0)
        arr = arr[:n_ch]
        band_arrays[band_name] = arr
        flat.extend(arr.tolist())
    return np.array(flat, dtype=float), band_arrays


def _ratios_from_bands(band_arrays: dict[str, np.ndarray], n_ch: int) -> np.ndarray:
    """Per-channel ratios: theta/alpha, theta/beta, alpha/beta, (delta+theta)/(alpha+beta). 4 * n_ch."""
    delta = band_arrays["delta"]
    theta = band_arrays["theta"]
    alpha = band_arrays["alpha"]
    beta = band_arrays["beta"]
    eps = 1e-6
    out = []
    out.append(theta / (alpha + eps))
    out.append(theta / (beta + eps))
    out.append(alpha / (beta + eps))
    out.append((delta + theta) / (alpha + beta + eps))
    return np.concatenate(out)


def _centroids_from_raw(raw, picks: list[int], n_ch: int) -> np.ndarray:
    """Spectral centroid per channel. Length n_ch."""
    psd = raw.compute_psd(picks=picks, fmin=CENTROID_FMIN, fmax=CENTROID_FMAX, verbose=False)
    freqs = psd.freqs
    data = psd.get_data()
    if data.ndim == 1:
        data = data.reshape(1, -1)
    centroids = []
    for ch in range(min(n_ch, data.shape[0])):
        centroids.append(spectral_centroid(freqs, data[ch], fmin=CENTROID_FMIN, fmax=CENTROID_FMAX))
    while len(centroids) < n_ch:
        centroids.append(0.0)
    return np.array(centroids[:n_ch], dtype=float)


def _entropies_from_raw(raw, picks: list[int], n_ch: int) -> np.ndarray:
    """Sample entropy + approximate entropy per channel. Length n_ch * 2. Returns zeros if antropy missing."""
    if not _ANTROPY_AVAILABLE:
        return np.zeros(n_ch * 2, dtype=float)
    data, _ = raw.get_data(picks=picks, return_times=True)
    n_use = min(data.shape[1], TIME_SEGMENT_SAMPLES)
    seg = data[:, :n_use]
    out = []
    for ch in range(min(n_ch, seg.shape[0])):
        sig = seg[ch]
        try:
            samp = ant.sample_entropy(sig, order=2, metric="chebyshev")
            app = ant.approximate_entropy(sig, order=2, metric="chebyshev")
        except Exception:
            samp = 0.0
            app = 0.0
        out.append(samp)
        out.append(app)
    while len(out) < n_ch * 2:
        out.extend([0.0, 0.0])
    return np.array(out[: n_ch * 2], dtype=float)


def _bandpass(signal: np.ndarray, fs: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    fmin_c = max(0.01, fmin / nyq)
    fmax_c = min(0.99, fmax / nyq)
    if fmin_c >= fmax_c:
        return signal
    b, a = butter(order, [fmin_c, fmax_c], btype="band")
    return filtfilt(b, a, signal)


def _envelope_freqs_from_data(data: np.ndarray, fs: float, n_ch: int) -> np.ndarray:
    """Envelope dominant frequency per band and channel (4 * n_ch)."""
    n_samples = data.shape[1]
    if n_samples < int(2 * fs):
        return np.zeros(N_CHANNELS * len(BANDS), dtype=float)

    out = []
    for band_name, (fmin, fmax) in BANDS.items():
        for ch in range(min(n_ch, data.shape[0])):
            sig = data[ch]
            filt = _bandpass(sig, fs, fmin, fmax)
            analytic = hilbert(filt)
            env = np.abs(analytic)
            freqs, psd = welch(env, fs=fs, nperseg=min(len(env), int(4 * fs)))
            idx = (freqs >= ENV_FREQ_MIN) & (freqs <= ENV_FREQ_MAX)
            if not np.any(idx):
                out.append(0.0)
            else:
                band_freqs = freqs[idx]
                band_psd = psd[idx]
                out.append(float(band_freqs[np.argmax(band_psd)]))
        # pad for missing channels if needed
        while len(out) % n_ch != 0:
            out.append(0.0)
    if len(out) < N_CHANNELS * len(BANDS):
        out.extend([0.0] * (N_CHANNELS * len(BANDS) - len(out)))
    return np.array(out[: N_CHANNELS * len(BANDS)], dtype=float)


def _higuchi_fd_from_data(data: np.ndarray, n_ch: int) -> np.ndarray:
    """Higuchi fractal dimension per channel (n_ch)."""
    if not _NOLDS_AVAILABLE:
        return np.zeros(n_ch, dtype=float)
    n_samples = data.shape[1]
    n_use = min(n_samples, TIME_SEGMENT_SAMPLES)
    out = []
    for ch in range(min(n_ch, data.shape[0])):
        sig = data[ch, :n_use]
        if sig.size < 10:
            out.append(0.0)
            continue
        kmax = min(HFD_KMAX, max(2, sig.size // 5))
        try:
            hfd = float(nolds.hfd(sig, kmax=kmax))
        except Exception:
            hfd = 0.0
        out.append(hfd)
    while len(out) < n_ch:
        out.append(0.0)
    return np.array(out[:n_ch], dtype=float)


def _hjorth_complexity_from_data(data: np.ndarray, fs: float, n_ch: int) -> np.ndarray:
    """Hjorth complexity per channel (n_ch), computed on 4–13 Hz bandpass."""
    out = []
    for ch in range(min(n_ch, data.shape[0])):
        sig = data[ch]
        sig = sig - np.mean(sig)
        sig = _bandpass(sig, fs, ALPHA_VAR_LO, ALPHA_VAR_HI)
        std_sig = float(np.std(sig))
        if std_sig < 1e-8:
            out.append(0.0)
            continue
        diff_sig = np.diff(sig)
        std_diff = float(np.std(diff_sig))
        out.append(std_diff / (std_sig + 1e-10))
    while len(out) < n_ch:
        out.append(0.0)
    return np.array(out[:n_ch], dtype=float)


def _alpha_variability_from_data(data: np.ndarray, fs: float, n_ch: int) -> np.ndarray:
    """Alpha variability (mean and std of instantaneous alpha frequency) per channel: length 2*n_ch."""
    out = []
    for ch in range(min(n_ch, data.shape[0])):
        sig = data[ch]
        filt = _bandpass(sig, fs, ALPHA_VAR_LO, ALPHA_VAR_HI)
        env = np.abs(filt)
        thr = float(np.mean(env))
        peaks, _ = find_peaks(filt)
        if peaks.size < 2:
            out.extend([0.0, 0.0])
            continue
        intervals = np.diff(peaks) / fs
        valid = (intervals > ALPHA_VAR_MIN_INT) & (intervals < ALPHA_VAR_MAX_INT)
        val_int = intervals[valid]
        if val_int.size == 0:
            out.extend([0.0, 0.0])
            continue
        freqs = 1.0 / val_int
        out.append(float(np.mean(freqs)))
        out.append(float(np.std(freqs)))
    while len(out) < 2 * n_ch:
        out.extend([0.0, 0.0])
    return np.array(out[: 2 * n_ch], dtype=float)


def _theta_ratio_topography(raw, canonical_ch_names: list[str]) -> np.ndarray:
    """Theta1/Theta2 ratio averaged over frontal, central, parieto-occipital regions (3 features)."""
    import mne

    if not canonical_ch_names:
        return np.zeros(3, dtype=float)

    picks = []
    idx_map = {}
    for i, name in enumerate(canonical_ch_names):
        if not name:
            continue
        try:
            pick = raw.ch_names.index(name)
        except ValueError:
            continue
        idx_map[len(picks)] = name
        picks.append(pick)
    if not picks:
        return np.zeros(3, dtype=float)

    psd = raw.compute_psd(picks=picks, fmin=4.0, fmax=8.0, verbose=False)
    freqs = psd.freqs
    data = psd.get_data()
    theta1_mask = (freqs >= 4.0) & (freqs < 6.0)
    theta2_mask = (freqs >= 6.0) & (freqs <= 8.0)

    ratios = np.zeros(len(picks), dtype=float)
    eps = 1e-10
    for i in range(len(picks)):
        p = data[i]
        p1 = float(np.sum(p[theta1_mask]))
        p2 = float(np.sum(p[theta2_mask]))
        ratios[i] = p1 / (p2 + eps)

    def _region_indices(keywords: list[str]) -> list[int]:
        idxs = []
        for i, name in idx_map.items():
            if any(kw in name for kw in keywords):
                idxs.append(i)
        return idxs

    frontal_idx = _region_indices(["Fp", "F3", "F4", "F7", "F8", "Fz"])
    central_idx = _region_indices(["C3", "C4", "Cz"])
    parocc_idx = _region_indices(["P3", "P4", "Pz", "O1", "O2"])

    def _avg(idx_list: list[int]) -> float:
        if not idx_list:
            return 0.0
        return float(np.mean(ratios[idx_list]))

    return np.array(
        [
            _avg(frontal_idx),
            _avg(central_idx),
            _avg(parocc_idx),
        ],
        dtype=float,
    )


def _alpha_power_topography(raw, canonical_ch_names: list[str]) -> np.ndarray:
    """
    Alpha-rhythm (8-13 Hz) and predecessor (4-12 Hz) power topography: 12 features total.
    
    For each band (alpha 8-13 Hz, predecessor 4-12 Hz):
    - 3 regional powers (frontal, central, parieto-occipital)
    - 3 power ratios (frontal/central, frontal/parieto-occipital, central/parieto-occipital)
    
    Only meaningful for closed-eyes recordings (alpha is suppressed with open eyes).
    """
    import mne

    if not canonical_ch_names:
        return np.zeros(12, dtype=float)

    picks = []
    idx_map = {}
    for i, name in enumerate(canonical_ch_names):
        if not name:
            continue
        try:
            pick = raw.ch_names.index(name)
        except ValueError:
            continue
        idx_map[len(picks)] = name
        picks.append(pick)
    if not picks:
        return np.zeros(12, dtype=float)

    def _get_regional_powers(fmin: float, fmax: float) -> tuple[float, float, float]:
        """Get average power in frontal, central, parieto-occipital regions for given band."""
        psd = raw.compute_psd(picks=picks, fmin=fmin, fmax=fmax, verbose=False)
        data = psd.get_data()

        channel_power = np.zeros(len(picks), dtype=float)
        for i in range(len(picks)):
            channel_power[i] = float(np.mean(data[i]))

        def _region_indices(keywords: list[str]) -> list[int]:
            idxs = []
            for i, name in idx_map.items():
                if any(kw in name for kw in keywords):
                    idxs.append(i)
            return idxs

        frontal_idx = _region_indices(["Fp", "F3", "F4", "F7", "F8", "Fz"])
        central_idx = _region_indices(["C3", "C4", "Cz"])
        parocc_idx = _region_indices(["P3", "P4", "Pz", "O1", "O2"])

        def _avg_power(idx_list: list[int]) -> float:
            if not idx_list:
                return 0.0
            return float(np.mean(channel_power[idx_list]))

        return _avg_power(frontal_idx), _avg_power(central_idx), _avg_power(parocc_idx)

    # Alpha band (8-13 Hz)
    alpha_frontal, alpha_central, alpha_parocc = _get_regional_powers(8.0, 13.0)
    
    # Predecessor band (4-12 Hz)
    pred_frontal, pred_central, pred_parocc = _get_regional_powers(4.0, 12.0)

    eps = 1e-10
    
    # Alpha features: 3 regional powers + 3 ratios
    alpha_features = [
        alpha_frontal,
        alpha_central,
        alpha_parocc,
        alpha_frontal / (alpha_central + eps),
        alpha_frontal / (alpha_parocc + eps),
        alpha_central / (alpha_parocc + eps),
    ]
    
    # Predecessor features: 3 regional powers + 3 ratios
    pred_features = [
        pred_frontal,
        pred_central,
        pred_parocc,
        pred_frontal / (pred_central + eps),
        pred_frontal / (pred_parocc + eps),
        pred_central / (pred_parocc + eps),
    ]

    return np.array(alpha_features + pred_features, dtype=float)


def extract_all_features(
    raw,
    canonical_ch_names: list[str] | None = None,
    eyes_condition: str = "closed",
    include_literature: bool = False,
    literature_surrogate_iters: int | None = None,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """
    Extract band powers + ratios + spectral centroids + entropies (if antropy installed).
    Same channel alignment as before: first file sets canonical_ch_names; others use it or 0.
    Returns (feat, canonical_ch_names) for first file, feat only otherwise.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    canonical_ch_names : list[str] | None
        Channel names from first file for alignment.
    eyes_condition : str
        "closed" or "open". If "closed", adds alpha-rhythm topography features (3 extra features).
    include_literature : bool
        If True, append Vandenbosch / van Noordt–style features (1 Hz PSD bins, MSE, PSD slope, surrogate).
    literature_surrogate_iters : int | None
        Phase-shuffle iterations per channel (None → eeg_literature_features default / env).
    """
    import mne

    ch_names = raw.ch_names
    n_ch = N_CHANNELS
    is_first_file = canonical_ch_names is None

    if is_first_file:
        picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(picks) == 0:
            picks = list(range(min(n_ch, raw.info["nchan"])))
        picks = picks[:n_ch]
        canonical_ch_names = [ch_names[i] for i in picks]
        while len(canonical_ch_names) < n_ch:
            canonical_ch_names.append("")
        canonical_ch_names = canonical_ch_names[:n_ch]
    else:
        picks = []
        for name in canonical_ch_names:
            if name and name in ch_names:
                picks.append(ch_names.index(name))
            else:
                break
        if len(picks) == 0:
            return np.zeros(total_feature_length(eyes_condition, include_literature), dtype=float)
        picks = picks[:n_ch]

    # Raw data segment for time-domain features
    data, _ = raw.get_data(picks=picks, return_times=True)
    fs = float(raw.info["sfreq"])

    # Band powers (76)
    flat_bp, band_arrays = _band_power_arrays(raw, picks, n_ch)
    if len(flat_bp) < n_ch * len(BANDS):
        flat_bp = np.pad(flat_bp, (0, n_ch * len(BANDS) - len(flat_bp)), constant_values=0.0)
    flat_bp = flat_bp[: n_ch * len(BANDS)]

    # Ratios (76)
    ratios = _ratios_from_bands(band_arrays, n_ch)

    # Centroids (19)
    centroids = _centroids_from_raw(raw, picks, n_ch)

    # Entropies (38) or zeros
    entropies = _entropies_from_raw(raw, picks, n_ch)

    # Envelope frequencies (76)
    env_freqs = _envelope_freqs_from_data(data, fs, n_ch)

    # Higuchi FD (19)
    hfd = _higuchi_fd_from_data(data, n_ch)

    # Hjorth complexity (19)
    hjorth = _hjorth_complexity_from_data(data, fs, n_ch)

    # Alpha variability (38)
    alpha_var = _alpha_variability_from_data(data, fs, n_ch)

    # Theta1/Theta2 regional ratios (3)
    theta_topo = _theta_ratio_topography(raw, canonical_ch_names)

    # Alpha-rhythm topography (3, closed-eyes only)
    use_alpha_topo = (eyes_condition == "closed")
    if use_alpha_topo:
        alpha_topo = _alpha_power_topography(raw, canonical_ch_names)
    else:
        alpha_topo = np.array([], dtype=float)

    feat = np.concatenate(
        [flat_bp, ratios, centroids, entropies, env_freqs, hfd, hjorth, alpha_var, theta_topo, alpha_topo]
    )
    target_len = legacy_feature_length(eyes_condition)
    if len(feat) < target_len:
        feat = np.pad(feat, (0, target_len - len(feat)), constant_values=0.0)
    feat = feat[:target_len].astype(float)

    if include_literature:
        lit_vec, _ = literature_feature_vector_from_raw(
            raw,
            picks,
            canonical_ch_names,
            surrogate_iters=literature_surrogate_iters,
        )
        feat = np.concatenate([feat, lit_vec.astype(float)])

    full_len = total_feature_length(eyes_condition, include_literature)
    if len(feat) < full_len:
        feat = np.pad(feat, (0, full_len - len(feat)), constant_values=0.0)
    feat = feat[:full_len].astype(float)

    if is_first_file:
        return feat, canonical_ch_names
    return feat


def legacy_feature_length(eyes_condition: str = "closed") -> int:
    # 76 band power + 76 ratios + 19 centroids + 38 entropies +
    # 76 envelope freqs + 19 HFD + 19 Hjorth + 38 alpha-var + 3 theta-topography = 364
    # + 12 alpha-rhythm topography (closed-eyes only) = 376
    base = 76 + 76 + 19 + 38 + 76 + 19 + 19 + 38 + 3
    if eyes_condition == "closed":
        return base + 12
    return base


def total_feature_length(eyes_condition: str = "closed", include_literature: bool = False) -> int:
    n = legacy_feature_length(eyes_condition)
    if include_literature:
        n += literature_feature_count()
    return n


def _total_feature_length(eyes_condition: str = "closed") -> int:
    """Backward-compatible alias: legacy-only length."""
    return legacy_feature_length(eyes_condition)


def get_feature_names(eyes_condition: str = "closed", include_literature: bool = False) -> list[str]:
    """Return names for each feature in the same order as extract_all_features output (for importance analysis)."""
    names = []
    # Band powers
    for band in BANDS:
        for ch in range(N_CHANNELS):
            names.append(f"band_{band}_ch{ch}")
    # Ratios
    for ratio_name in ("theta_alpha", "theta_beta", "alpha_beta", "slow_fast"):
        for ch in range(N_CHANNELS):
            names.append(f"ratio_{ratio_name}_ch{ch}")
    # Spectral centroids
    for ch in range(N_CHANNELS):
        names.append(f"centroid_ch{ch}")
    # Entropies
    for ch in range(N_CHANNELS):
        names.append(f"samp_ent_ch{ch}")
        names.append(f"app_ent_ch{ch}")
    # Envelope frequencies
    for band in BANDS:
        for ch in range(N_CHANNELS):
            names.append(f"envfreq_{band}_ch{ch}")
    # Higuchi FD
    for ch in range(N_CHANNELS):
        names.append(f"hfd_ch{ch}")
    # Hjorth complexity
    for ch in range(N_CHANNELS):
        names.append(f"hjorth_complexity_ch{ch}")
    # Alpha variability (mean / std)
    for ch in range(N_CHANNELS):
        names.append(f"alpha_var_mean_ch{ch}")
        names.append(f"alpha_var_std_ch{ch}")
    # Theta topography
    names.append("theta_ratio_frontal")
    names.append("theta_ratio_central")
    names.append("theta_ratio_paroccipital")
    # Alpha-rhythm topography (closed-eyes only, 12 features)
    if eyes_condition == "closed":
        # Alpha band (8-13 Hz): 3 regional powers + 3 ratios
        names.append("alpha_power_frontal")
        names.append("alpha_power_central")
        names.append("alpha_power_paroccipital")
        names.append("alpha_ratio_frontal_central")
        names.append("alpha_ratio_frontal_paroccipital")
        names.append("alpha_ratio_central_paroccipital")
        # Predecessor band (4-12 Hz): 3 regional powers + 3 ratios
        names.append("pred_power_frontal")
        names.append("pred_power_central")
        names.append("pred_power_paroccipital")
        names.append("pred_ratio_frontal_central")
        names.append("pred_ratio_frontal_paroccipital")
        names.append("pred_ratio_central_paroccipital")
    if include_literature:
        names.extend(get_literature_feature_names())
    return names


def feature_description(eyes_condition: str = "closed", include_literature: bool = False) -> str:
    n = total_feature_length(eyes_condition, include_literature)
    parts = [
        "19 EEG ch × 4 bands (delta, theta, alpha, beta) = 76",
        "19 ch × 4 ratios (theta/alpha, theta/beta, alpha/beta, slow/fast) = 76",
        "19 spectral centroids (0.5–45 Hz) = 19",
        "19 ch × 2 entropies (sample + approximate) = 38"
        + ("" if _ANTROPY_AVAILABLE else " (zeros if antropy not installed)"),
        "Envelope dominant freqs per band & ch = 76",
        "Higuchi fractal dimension per ch = 19"
        + ("" if _NOLDS_AVAILABLE else " (zeros if nolds not installed)"),
        "Hjorth complexity per ch (4–13 Hz) = 19",
        "Alpha variability per ch (mean & std of alpha freq) = 38",
        "Theta1/theta2 regional ratios (frontal, central, parieto-occipital) = 3",
    ]
    if eyes_condition == "closed":
        parts.append("Alpha-rhythm topography (alpha 8-13Hz + predecessor 4-12Hz: regional powers + ratios) = 12")
    if include_literature:
        parts.append(
            f"Literature block (1 Hz bins 1–24 Hz × {N_CHANNELS} ch, regional MSE short/med/long, "
            "PSD slope+R², phase-shuffle nonlinearity) = "
            f"{literature_feature_count()}"
        )
    return f"  Features ({n}): " + "; ".join(parts) + "."


def get_feature_indices_by_category(eyes_condition: str = "closed", include_literature: bool = False) -> dict[str, list[int]]:
    """
    Return indices of features by category for selective training.
    
    Categories:
    - 'band_power': 76 features (19 ch × 4 bands)
    - 'band_ratio': 76 features (19 ch × 4 ratios)
    - 'centroid': 19 features (spectral centroids)
    - 'entropy': 38 features (sample + approximate entropy)
    - 'envelope_freq': 76 features (envelope dominant frequencies)
    - 'hfd': 19 features (Higuchi fractal dimension)
    - 'hjorth': 19 features (Hjorth complexity)
    - 'alpha_var': 38 features (alpha variability mean/std)
    - 'theta_topo': 3 features (theta1/theta2 regional ratios)
    - 'alpha_topo': 12 features (alpha-rhythm topography, closed-eyes only)
    - 'all': all features
    - 'alpha': alpha-related features (alpha_var + alpha_topo + centroids)
    - 'spectral': band_power + centroid + envelope_freq
    - 'time_domain': entropy + hfd + hjorth
    """
    indices = {}
    idx = 0
    
    # Band powers: 76 (19 × 4)
    band_power_end = idx + 76
    indices['band_power'] = list(range(idx, band_power_end))
    idx = band_power_end
    
    # Band ratios: 76 (19 × 4)
    band_ratio_end = idx + 76
    indices['band_ratio'] = list(range(idx, band_ratio_end))
    idx = band_ratio_end
    
    # Spectral centroids: 19
    centroid_end = idx + 19
    indices['centroid'] = list(range(idx, centroid_end))
    idx = centroid_end
    
    # Entropies: 38 (19 × 2)
    entropy_end = idx + 38
    indices['entropy'] = list(range(idx, entropy_end))
    idx = entropy_end
    
    # Envelope frequencies: 76 (19 × 4)
    envelope_end = idx + 76
    indices['envelope_freq'] = list(range(idx, envelope_end))
    idx = envelope_end
    
    # Higuchi FD: 19
    hfd_end = idx + 19
    indices['hfd'] = list(range(idx, hfd_end))
    idx = hfd_end
    
    # Hjorth complexity: 19
    hjorth_end = idx + 19
    indices['hjorth'] = list(range(idx, hjorth_end))
    idx = hjorth_end
    
    # Alpha variability: 38 (19 × 2)
    alpha_var_end = idx + 38
    indices['alpha_var'] = list(range(idx, alpha_var_end))
    idx = alpha_var_end
    
    # Theta topography: 3
    theta_topo_end = idx + 3
    indices['theta_topo'] = list(range(idx, theta_topo_end))
    idx = theta_topo_end
    
    # Alpha-rhythm topography: 12 (closed-eyes only)
    if eyes_condition == "closed":
        alpha_topo_end = idx + 12
        indices['alpha_topo'] = list(range(idx, alpha_topo_end))
        idx = alpha_topo_end

    if include_literature:
        lit_n = literature_feature_count()
        indices['literature'] = list(range(idx, idx + lit_n))
        idx += lit_n

    # All features
    indices['all'] = list(range(idx))
    
    # Combined categories
    # Alpha-related: alpha_var + alpha_topo + centroids
    alpha_indices = indices['alpha_var'].copy()
    if 'alpha_topo' in indices:
        alpha_indices.extend(indices['alpha_topo'])
    alpha_indices.extend(indices['centroid'])
    indices['alpha'] = alpha_indices
    
    # Spectral: band_power + centroid + envelope_freq
    indices['spectral'] = indices['band_power'] + indices['centroid'] + indices['envelope_freq']
    
    # Time domain: entropy + hfd + hjorth
    indices['time_domain'] = indices['entropy'] + indices['hfd'] + indices['hjorth']
    
    return indices
