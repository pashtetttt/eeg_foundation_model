"""
Quality control: compare raw EDF vs processed FIF for the example file.

Generates comparison plots and prints basic stats. Run from thesis folder:
  python qc_processing.py

Output: plots saved under qc_output/ (raw_vs_clean.png, psd_before_after.png)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

BASE = Path(".")
PROCESSED_DIR = BASE / "processed"
QC_OUTPUT_DIR = BASE / "qc_output"

# Example file (same as process_edf.py)
EXAMPLE_EDF = BASE / "preschooler" / "6(38)" / "настя_6_zg.EDF"
EXAMPLE_FIF = PROCESSED_DIR / "preschooler" / "6(38)" / "настя_6_zg_processed.fif"

DURATION_PLOT = 10  # seconds for time-series comparison
FMIN, FMAX = 1.0, 40.0


def main():
    if not EXAMPLE_EDF.exists():
        raise FileNotFoundError(f"Raw EDF not found: {EXAMPLE_EDF}")
    if not EXAMPLE_FIF.exists():
        raise FileNotFoundError(f"Processed FIF not found: {EXAMPLE_FIF}. Run process_edf.py first.")

    QC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw EDF and processed FIF...")
    raw = mne.io.read_raw_edf(EXAMPLE_EDF, preload=True, verbose=False)
    clean = mne.io.read_raw_fif(EXAMPLE_FIF, preload=True, verbose=False)

    picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, exclude="bads")
    if len(picks) == 0:
        picks = list(range(min(19, raw.info["nchan"])))
    n_plot = min(10, len(picks))
    picks_plot = picks[:n_plot]

    # --- 1) Time series: raw vs clean with SHARED y-scale so amplitude difference is visible ---
    d_raw, times = raw.get_data(picks=picks_plot, return_times=True)
    d_clean, _ = clean.get_data(picks=picks_plot, return_times=True)
    mask = times <= DURATION_PLOT
    t_plot = times[mask]
    d_raw_plot = d_raw[:, mask]
    d_clean_plot = d_clean[:, mask]

    # Use same channel spacing for both panels (based on raw amplitude) so scale is comparable
    ch_spacing = np.nanmax(np.abs(d_raw_plot)) * 2.2 or 1.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for i, (ch_name, sig_r, sig_c) in enumerate(zip(
        [raw.ch_names[j] for j in picks_plot], d_raw_plot, d_clean_plot
    )):
        off = i * ch_spacing
        axes[0].plot(t_plot, sig_r + off, linewidth=0.7, label=ch_name)
        axes[1].plot(t_plot, sig_c + off, linewidth=0.7, label=ch_name)
    axes[0].set_ylabel("amplitude (offset)")
    axes[0].set_title("Raw (EDF) — larger amplitude")
    axes[0].legend(loc="upper right", fontsize=6, ncol=2)
    axes[0].set_ylim(-0.3 * ch_spacing, n_plot * ch_spacing + 0.3 * ch_spacing)

    axes[1].set_ylabel("amplitude (offset)")
    axes[1].set_title("Processed (FIF) — reduced amplitude after filtering + ICA")
    axes[1].legend(loc="upper right", fontsize=6, ncol=2)
    axes[1].set_ylim(-0.3 * ch_spacing, n_plot * ch_spacing + 0.3 * ch_spacing)

    # Third panel: difference (raw − processed) to show what was removed
    for i, (ch_name, sig_r, sig_c) in enumerate(zip(
        [raw.ch_names[j] for j in picks_plot], d_raw_plot, d_clean_plot
    )):
        diff = sig_r - sig_c
        off = i * (np.nanmax(np.abs(d_raw_plot - d_clean_plot)) * 2.2 or 1.0)
        axes[2].plot(t_plot, diff + off, linewidth=0.7, label=ch_name)
    axes[2].set_ylabel("amplitude (offset)")
    axes[2].set_title("Difference (raw − processed) — removed noise/artifacts")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right", fontsize=6, ncol=2)

    fig.suptitle(f"Example: {EXAMPLE_EDF.name} — first {DURATION_PLOT} s (same scale: raw vs processed)")
    plt.tight_layout()
    out_ts = QC_OUTPUT_DIR / "raw_vs_clean.png"
    plt.savefig(out_ts, dpi=120)
    plt.close()
    print(f"Saved: {out_ts}")

    # --- 2) PSD before/after (one channel): plot manually so both lines are visible ---
    ch_idx = picks_plot[0]
    ch_name = raw.ch_names[ch_idx]

    raw_psd = raw.compute_psd(picks=[ch_idx], fmin=FMIN, fmax=FMAX, verbose=False)
    clean_psd = clean.compute_psd(picks=[ch_idx], fmin=FMIN, fmax=FMAX, verbose=False)

    freqs = raw_psd.freqs
    raw_power = raw_psd.get_data().squeeze()
    clean_power = clean_psd.get_data().squeeze()
    # Convert to dB if not already (MNE often returns linear; check and use 10*log10 for dB)
    if raw_power.max() > 100:
        raw_power_db = 10 * np.log10(raw_power + 1e-20)
        clean_power_db = 10 * np.log10(clean_power + 1e-20)
        ylabel = "Power (dB)"
    else:
        raw_power_db = raw_power
        clean_power_db = clean_power
        ylabel = "Power (dB/Hz re 1 µV²)" if raw_power.max() < 50 else "Power"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, raw_power_db, color="C0", linewidth=2, label="Raw (EDF)", alpha=0.9)
    ax.plot(freqs, clean_power_db, color="C1", linewidth=2, linestyle="--", label="Processed (FIF)", alpha=0.9)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"PSD — {ch_name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_psd = QC_OUTPUT_DIR / "psd_before_after.png"
    plt.savefig(out_psd, dpi=120)
    plt.close()
    print(f"Saved: {out_psd}")

    # --- 3) Quantitative stats ---
    raw_data, times = raw.get_data(picks=picks, return_times=True)
    clean_data, _ = clean.get_data(picks=picks, return_times=True)

    # Align lengths (in case of slight difference)
    n_min = min(raw_data.shape[1], clean_data.shape[1])
    raw_data = raw_data[:, :n_min]
    clean_data = clean_data[:, :n_min]

    mean_amp_raw = np.mean(np.abs(raw_data))
    mean_amp_clean = np.mean(np.abs(clean_data))
    corrs = []
    for r, c in zip(raw_data, clean_data):
        cc = np.corrcoef(r, c)[0, 1]
        corrs.append(cc if not np.isnan(cc) else 0.0)

    print("\n--- QC stats ---")
    print(f"Mean |amplitude| raw:     {mean_amp_raw:.4f}")
    print(f"Mean |amplitude| processed: {mean_amp_clean:.4f}")
    print(f"Channel-wise correlation (raw vs processed): mean = {np.mean(corrs):.3f}, min = {np.min(corrs):.3f}")

    print("\nDone. Check plots in qc_output/.")


if __name__ == "__main__":
    main()
