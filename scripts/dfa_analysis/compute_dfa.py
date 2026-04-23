from __future__ import annotations

"""
Шаг 1: расчёт DFA-альфы по субъектам/каналам и агрегация по регионам.

Результат: таблица
Subject_ID | Group | Age_Class | Region | DFA_Alpha

Комментарии: на русском. Имена переменных/функций: английские.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from edf_loader import load_raw_edf_resilient
from eeg_experiment_shared import GROUPS, GROUPS_DATA_KIDS, find_edf_files
from scripts.error_analysis.subject_ids import subject_id_from_path
from .dfa_core import dfa_alpha, dfa_backend_name


REGION_MAP = {
    "frontal": ("Fp", "F"),
    "central": ("C",),
    "parietal": ("P",),
    "occipital": ("O",),
}


def channel_region(ch_name: str) -> str:
    ch = ch_name.strip()
    # Простая эвристика по 10-20
    if ch.startswith("Fp") or ch.startswith("F"):
        return "frontal"
    if ch.startswith("C"):
        return "central"
    if ch.startswith("P"):
        return "parietal"
    if ch.startswith("O"):
        return "occipital"
    if ch.startswith("T") or ch.startswith("FT") or ch.startswith("TP"):
        return "temporal"
    return "other"


def bandpass_0p5_45(x: np.ndarray, fs: float) -> np.ndarray:
    # 0.5–45 Hz Butterworth
    nyq = fs / 2.0
    lo = 0.5 / nyq
    hi = min(0.99, 45.0 / nyq)
    if lo >= hi:
        return x
    b, a = butter(4, [lo, hi], btype="band")
    return filtfilt(b, a, x)


def is_artifact_channel(x: np.ndarray) -> bool:
    """
    Простая проверка артефактов/выбросов:
    - слишком маленькая дисперсия (плоский канал)
    - слишком много экстремальных значений по MAD
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 200:
        return True
    sd = float(np.std(x))
    if sd < 1e-10:
        return True
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-12
    z = np.abs((x - med) / (1.4826 * mad))
    frac_extreme = float(np.mean(z > 8.0))
    return frac_extreme > 0.02


def _resolve_groups(data_dir: Path) -> dict[str, str]:
    try:
        children = {p.name for p in data_dir.iterdir() if p.is_dir()}
    except Exception:
        children = set()
    if any(n in children for n in GROUPS_DATA_KIDS.keys()):
        return GROUPS_DATA_KIDS
    return GROUPS


@dataclass(frozen=True)
class DfaRow:
    subject_id: str
    group: str  # healthy|patients
    age_class: str
    region: str
    dfa_alpha: float


def compute_dfa_table(
    *,
    data_dir: Path,
    group_label: str,
    eyes: str,
    max_per_class: int | None,
    min_scale: int = 4,
    order: int = 1,
) -> tuple[list[DfaRow], dict]:
    """
    Возвращает (rows, meta).
    eyes: closed|open
    """
    groups = _resolve_groups(data_dir)
    class_names = list(groups.values())
    rows: list[DfaRow] = []
    per_file_debug = 0

    for folder, label in groups.items():
        paths = find_edf_files(data_dir, folder, max_per_class, eyes_condition=eyes)
        if not paths:
            continue
        for p in paths:
            sid = subject_id_from_path(p)
            try:
                raw = load_raw_edf_resilient(p, preload=True, verbose=False)
                fs = float(raw.info["sfreq"])
                import mne

                picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
                if len(picks) == 0:
                    continue
                data = raw.get_data(picks=picks)
                ch_names = [raw.ch_names[i] for i in picks]
                # bandpass
                data_f = np.zeros_like(data)
                for i in range(data.shape[0]):
                    sig = data[i]
                    sig = sig - float(np.mean(sig))
                    sig = bandpass_0p5_45(sig, fs)
                    data_f[i] = sig

                # compute DFA per channel
                region_vals: dict[str, list[float]] = {}
                for i, ch in enumerate(ch_names):
                    sig = data_f[i]
                    if is_artifact_channel(sig):
                        continue
                    alpha = dfa_alpha(sig, min_scale=min_scale, max_scale=len(sig) // 10, order=order)
                    if not np.isfinite(alpha):
                        continue
                    reg = channel_region(ch)
                    if reg not in ("frontal", "central", "parietal", "occipital"):
                        continue
                    region_vals.setdefault(reg, []).append(float(alpha))

                # aggregate per region (median)
                for reg, vals in region_vals.items():
                    if len(vals) == 0:
                        continue
                    rows.append(
                        DfaRow(
                            subject_id=sid,
                            group=group_label,
                            age_class=label,
                            region=reg,
                            dfa_alpha=float(np.median(np.asarray(vals, dtype=float))),
                        )
                    )
            except Exception:
                continue

    meta = {
        "data_dir": str(data_dir),
        "group_label": group_label,
        "eyes": eyes,
        "max_per_class": max_per_class,
        "min_scale": min_scale,
        "order": order,
        "dfa_backend": dfa_backend_name(),
    }
    return rows, meta


def save_dfa_rows(rows: list[DfaRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "group", "age_class", "region", "dfa_alpha"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "subject_id": r.subject_id,
                    "group": r.group,
                    "age_class": r.age_class,
                    "region": r.region,
                    "dfa_alpha": f"{r.dfa_alpha:.6g}",
                }
            )


def save_meta(meta: dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

