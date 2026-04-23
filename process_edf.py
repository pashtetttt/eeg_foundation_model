"""
EEG data processing: bandpass filter, notch filter, ICA artifact removal.

Processes EDF files and saves cleaned data as MNE .fif in processed/
One class per group (4 groups); age subfolders inside each group are just paths.

Run from the thesis folder: python process_edf.py

Usage:
  Single file (example):  python process_edf.py
  All files (batch):      python process_edf.py --batch
  Limit per group:        python process_edf.py --batch --max 10
"""

import argparse
from pathlib import Path

import mne
from mne.preprocessing import ICA

# --- Config (edit if needed) ---
BASE = Path(".")
PROCESSED_DIR = BASE / "processed"

# Group folders and class names
GROUPS = {
    "дошкольники (124)": "preschooler",
    "младшие школьники (177)": "primary",
    "подростки (160)": "teenager",
    "юность(273)": "adolescence",
}

# Filtering
BANDPASS_LOW = 1.0   # Hz
BANDPASS_HIGH = 40.0 # Hz
NOTCH_FREQ = 50.0    # Hz (use 60.0 for US powerline)

# ICA
# Use an integer number of components to avoid issues when the data is nearly rank-1.
# You can lower this (e.g. 10) to speed up, or increase if you have many channels.
ICA_N_COMPONENTS = 20
ICA_RANDOM_STATE = 42


def find_edf_files(base: Path, group_folder: str, max_per_group: int | None) -> list[Path]:
    """List EDF files under group folder, optionally limited."""
    folder = base / group_folder
    if not folder.exists():
        return []
    paths = sorted(folder.rglob("*.edf")) + sorted(folder.rglob("*.EDF"))
    # dedupe (same file might match both)
    seen = set()
    unique = []
    for p in paths:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if max_per_group is not None:
        unique = unique[:max_per_group]
    return unique


def process_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply bandpass, notch, and ICA artifact removal. Modifies raw in place."""
    # 1) Bandpass
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, verbose=False)

    # 2) Notch (powerline)
    raw.notch_filter(NOTCH_FREQ, verbose=False)

    # 3) ICA
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, stim=False, exclude="bads")
    if len(picks_eeg) < 2:
        return raw

    try:
        ica = ICA(n_components=ICA_N_COMPONENTS, random_state=ICA_RANDOM_STATE, verbose=False)
        ica.fit(raw, picks=picks_eeg, verbose=False)
    except Exception as e:
        # If ICA cannot be fit (e.g. rank-1 data, missing sklearn), skip ICA but keep filtered data.
        print(f"  ICA skipped: {e}")
        return raw

    # Find artifact components (ECG/EOG); use synthetic if no dedicated channels
    bads = []
    try:
        bads_ecg = ica.find_bads_ecg(raw, method="correlation", verbose=False)
        bads.extend(bads_ecg)
    except Exception:
        pass
    try:
        bads_eog = ica.find_bads_eog(raw, verbose=False)
        bads.extend(bads_eog)
    except Exception:
        pass
    bads = list(dict.fromkeys(bads))  # unique, preserve order

    if bads:
        ica.exclude = bads
        ica.apply(raw, verbose=False)

    return raw


def process_one_file(edf_path: Path, out_path: Path) -> bool:
    """Load one EDF, process, save as .fif. Returns True on success."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  Skip (read error): {edf_path.name} — {e}")
        return False

    try:
        process_raw(raw)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(out_path, overwrite=True, verbose=False)
        return True
    except Exception as e:
        print(f"  Skip (process error): {edf_path.name} — {e}")
        return False


def run_example():
    """Process single example file and save to processed/."""
    example = BASE / "дошкольники (124)" / "6(38)" / "настя_6_zg.EDF"
    if not example.exists():
        raise FileNotFoundError(f"Example file not found: {example}")

    out = PROCESSED_DIR / "дошкольники (124)" / "6(38)" / "настя_6_zg_processed.fif"
    print(f"Processing example: {example.name}")
    ok = process_one_file(example, out)
    if ok:
        print(f"Saved: {out}")
    return ok


def run_batch(max_per_group: int | None = None):
    """Process all EDF files in each group, mirroring folder structure under processed/."""
    for group_folder in GROUPS:
        edfs = find_edf_files(BASE, group_folder, max_per_group)
        print(f"{group_folder}: {len(edfs)} files")
        for edf_path in edfs:
            # Mirror path under processed/ and add _processed.fif
            rel = edf_path.relative_to(BASE / group_folder)
            out_path = PROCESSED_DIR / group_folder / rel.parent / (edf_path.stem + "_processed.fif")
            process_one_file(edf_path, out_path)


def main():
    parser = argparse.ArgumentParser(description="Process EEG EDF: bandpass, notch, ICA.")
    parser.add_argument("--batch", action="store_true", help="Process all files in all groups")
    parser.add_argument("--max", type=int, default=None, help="Max files per group (for quick runs)")
    args = parser.parse_args()

    if args.batch:
        run_batch(max_per_group=args.max)
    else:
        run_example()


if __name__ == "__main__":
    main()
