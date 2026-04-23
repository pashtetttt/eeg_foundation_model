"""
Resilient EDF reader for files with malformed numeric header fields.

Some EDF files contain blank numeric slots (e.g. "        ") in per-channel
header fields. MNE then raises ValueError during float conversion even though
the signal data itself is readable. This module retries by sanitizing numeric
header fields in a temporary copy.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import mne


def _read_ascii_field(header: bytearray, start: int, width: int) -> str:
    return bytes(header[start : start + width]).decode("ascii", errors="ignore")


def _write_ascii_field(header: bytearray, start: int, width: int, value: str) -> None:
    s = value[:width].rjust(width)
    header[start : start + width] = s.encode("ascii", errors="ignore")


def _parse_int(text: str, default: int) -> int:
    t = text.replace("\x00", "").replace(",", ".").strip()
    if not t:
        return default
    try:
        return int(float(t))
    except Exception:
        return default


def _parse_float(text: str, default: float) -> float:
    t = text.replace("\x00", "").replace(",", ".").strip()
    if not t:
        return default
    try:
        return float(t)
    except Exception:
        return default


def _format_int(value: int, width: int) -> str:
    return str(int(value)).rjust(width)[:width]


def _format_float(value: float, width: int) -> str:
    raw = f"{float(value):.5g}"
    if len(raw) > width:
        raw = f"{float(value):.1f}"
    if len(raw) > width:
        raw = str(float(value))
    return raw.rjust(width)[:width]


def _sanitize_edf_numeric_header(path: Path) -> Path:
    raw_bytes = bytearray(path.read_bytes())
    if len(raw_bytes) < 256:
        raise ValueError("EDF file too small for header")

    header = raw_bytes
    header_nbytes = _parse_int(_read_ascii_field(header, 184, 8), 0)
    n_signals = _parse_int(_read_ascii_field(header, 252, 4), 0)
    if n_signals <= 0:
        if header_nbytes >= 256 and (header_nbytes - 256) % 256 == 0:
            n_signals = (header_nbytes - 256) // 256
        else:
            raise ValueError("Cannot infer number of signals from EDF header")

    expected_header = 256 + 256 * n_signals
    if header_nbytes <= 0 or header_nbytes > len(header):
        header_nbytes = expected_header
    if header_nbytes < expected_header:
        raise ValueError("EDF header shorter than expected")

    _write_ascii_field(header, 184, 8, _format_int(header_nbytes, 8))
    num_records = _parse_int(_read_ascii_field(header, 236, 8), -1)
    duration = _parse_float(_read_ascii_field(header, 244, 8), 1.0)
    if duration <= 0:
        duration = 1.0
    _write_ascii_field(header, 236, 8, _format_int(num_records, 8))
    _write_ascii_field(header, 244, 8, _format_float(duration, 8))
    _write_ascii_field(header, 252, 4, _format_int(n_signals, 4))

    base = 256
    transducer_block = base + 16 * n_signals
    phys_dim_block = transducer_block + 80 * n_signals
    phys_min_block = phys_dim_block + 8 * n_signals
    phys_max_block = phys_min_block + 8 * n_signals
    dig_min_block = phys_max_block + 8 * n_signals
    dig_max_block = dig_min_block + 8 * n_signals
    prefilter_block = dig_max_block + 8 * n_signals
    samples_block = prefilter_block + 80 * n_signals

    for i in range(n_signals):
        phys_min = _parse_float(_read_ascii_field(header, phys_min_block + 8 * i, 8), -1.0)
        phys_max = _parse_float(_read_ascii_field(header, phys_max_block + 8 * i, 8), 1.0)
        dig_min = _parse_int(_read_ascii_field(header, dig_min_block + 8 * i, 8), -32768)
        dig_max = _parse_int(_read_ascii_field(header, dig_max_block + 8 * i, 8), 32767)
        n_samp = _parse_int(_read_ascii_field(header, samples_block + 8 * i, 8), 1)

        if phys_max <= phys_min:
            phys_min, phys_max = -1.0, 1.0
        if dig_max <= dig_min:
            dig_min, dig_max = -32768, 32767
        if n_samp <= 0:
            n_samp = 1

        _write_ascii_field(header, phys_min_block + 8 * i, 8, _format_float(phys_min, 8))
        _write_ascii_field(header, phys_max_block + 8 * i, 8, _format_float(phys_max, 8))
        _write_ascii_field(header, dig_min_block + 8 * i, 8, _format_int(dig_min, 8))
        _write_ascii_field(header, dig_max_block + 8 * i, 8, _format_int(dig_max, 8))
        _write_ascii_field(header, samples_block + 8 * i, 8, _format_int(n_samp, 8))

    fd, temp_path = tempfile.mkstemp(prefix="edf_sanitized_", suffix=".edf")
    os.close(fd)
    tmp = Path(temp_path)
    tmp.write_bytes(bytes(header))
    return tmp


def _read_raw_edf_quiet(path: Path, preload: bool, verbose: bool):
    """read_raw_edf with noisy RuntimeWarnings suppressed (per-channel EDF metadata)."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Physical range is not defined.*",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Channels contain different (high|low)pass filters.*",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Channel names are not unique.*",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Invalid measurement date encountered.*",
            category=RuntimeWarning,
        )
        return mne.io.read_raw_edf(path, preload=preload, verbose=verbose)


def load_raw_edf_resilient(path: Path, preload: bool = True, verbose: bool = False):
    """
    Read EDF via MNE. If numeric header parsing fails, sanitize and retry once.
    """
    try:
        return _read_raw_edf_quiet(path, preload=preload, verbose=verbose)
    except Exception as e:
        msg = str(e)
        if "could not convert string to float" not in msg:
            raise
        temp_path = _sanitize_edf_numeric_header(path)
        try:
            return _read_raw_edf_quiet(temp_path, preload=preload, verbose=verbose)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
