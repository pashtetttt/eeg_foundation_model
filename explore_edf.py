"""
EEG EDF Data Exploration

Explores one sample EDF file from the thesis dataset using MNE-Python.

Dataset folders: дошкольники (124), младшие школьники (177), подростки (160), юность (273)

Run from the thesis folder: python explore_edf.py
"""

import mne
from pathlib import Path
import matplotlib.pyplot as plt

# Base path (thesis folder — run script from there, or set absolute path)
BASE = Path(".")

# Example file: one EDF from дошкольники (preschoolers)
EXAMPLE_EDF = BASE / "дошкольники (124)" / "6(38)" / "настя_6_zg.EDF"

# Alternatively, pick from other groups:
# EXAMPLE_EDF = BASE / "младшие школьники (177)" / "..." / "file.edf"
# EXAMPLE_EDF = BASE / "подростки (160)" / "..." / "file.edf"
# EXAMPLE_EDF = BASE / "юность(273)" / "23(42)" / "male_23_og.edf"

if not EXAMPLE_EDF.exists():
    raise FileNotFoundError(f"File not found: {EXAMPLE_EDF}")

print(f"Loading: {EXAMPLE_EDF.name}")

# Read EDF with MNE (preload=True for easier plotting)
raw = mne.io.read_raw_edf(EXAMPLE_EDF, preload=True, verbose=False)
print(raw)

# --- Basic metadata ---
print("\n--- Metadata ---")
print("Sampling frequency (Hz):", raw.info["sfreq"])
print("Number of channels:", raw.info["nchan"])
print("Duration (seconds):", raw.times[-1] - raw.times[0])
print("Channel names:", raw.ch_names)
print("Channel types:", raw.get_channel_types())

# --- Plot 1: Raw data overview (first 10 s) ---
raw.plot(duration=10, n_channels=min(20, raw.info["nchan"]), scalings="auto")  # blocks until window closed

# --- Plot 2: A few EEG channels (time series) ---
eeg_chs = [ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == "eeg"]
picks = eeg_chs[:6] if len(eeg_chs) >= 6 else raw.ch_names[:6]

fig, axes = plt.subplots(len(picks), 1, figsize=(12, 2 * len(picks)), sharex=True)
if len(picks) == 1:
    axes = [axes]

t_sec = 5
data, times = raw.get_data(picks=picks, return_times=True)
mask = times <= t_sec
times_plot = times[mask]
data_plot = data[:, mask]

for ax, ch_name, sig in zip(axes, picks, data_plot):
    ax.plot(times_plot, sig, linewidth=0.8)
    ax.set_ylabel(ch_name, fontsize=9)
    ax.set_ylim(sig.min() - 5, sig.max() + 5)

axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"First {t_sec} s — selected channels")
plt.tight_layout()
plt.show()

# --- Plot 3: Power spectrum (one channel) ---
pick = picks[0]
raw.compute_psd(picks=[pick], fmax=60).plot(average=True)
plt.title(f"PSD — {pick}")
plt.show()
