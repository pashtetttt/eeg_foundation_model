from __future__ import annotations

"""
Шаг 2: статистический анализ DFA таблицы.

Тесты:
- Kruskal-Wallis (возраст внутри группы) по регионам
- Пост-хок: попарные сравнения (MWU) с FDR (как практический аналог Dunn + FDR)
- Healthy vs Patients внутри возраста: MWU + FDR

Эффект: Cliff's delta (непараметрический) + направление (у кого выше).
"""

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, mannwhitneyu

from scripts.error_analysis.stats_utils import benjamini_hochberg


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    # O(n*m) — ок для агрегированных регионов; если будет медленно, оптимизируем.
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    denom = a.size * b.size
    return (gt - lt) / denom


def load_dfa_csv(path: Path) -> list[dict]:
    import csv

    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["dfa_alpha"] = float(row["dfa_alpha"])
            rows.append(row)
    return rows


def _subset(rows: list[dict], **conds) -> np.ndarray:
    vals = []
    for r in rows:
        ok = True
        for k, v in conds.items():
            if r.get(k) != v:
                ok = False
                break
        if ok and np.isfinite(r["dfa_alpha"]):
            vals.append(r["dfa_alpha"])
    return np.asarray(vals, dtype=float)


def analyze_stats(rows: list[dict]) -> list[dict]:
    age_classes = sorted({r["age_class"] for r in rows})
    regions = sorted({r["region"] for r in rows})
    groups = sorted({r["group"] for r in rows})

    out: list[dict] = []

    # A/B: age effect within each group
    for g in groups:
        for reg in regions:
            samples = []
            for age in age_classes:
                x = _subset(rows, group=g, region=reg, age_class=age)
                samples.append(x)
            if any(s.size < 5 for s in samples):
                continue
            stat, p = kruskal(*samples)
            out.append(
                {
                    "comparison_type": f"age_within_{g}",
                    "group": g,
                    "region": reg,
                    "age_class": "ALL",
                    "test": "kruskal_wallis",
                    "statistic": float(stat),
                    "p_value": float(p),
                }
            )

            # posthoc pairwise MWU (Dunn-like) with FDR within this reg+group
            pairs = list(itertools.combinations(age_classes, 2))
            pvals = []
            pair_rows = []
            for a1, a2 in pairs:
                x1 = _subset(rows, group=g, region=reg, age_class=a1)
                x2 = _subset(rows, group=g, region=reg, age_class=a2)
                if x1.size < 5 or x2.size < 5:
                    continue
                pr = mannwhitneyu(x1, x2, alternative="two-sided").pvalue
                pvals.append(float(pr))
                delta = cliffs_delta(x1, x2)
                direction = f"{a1}>{a2}" if float(np.median(x1)) > float(np.median(x2)) else f"{a1}<{a2}"
                pair_rows.append(
                    {
                        "comparison_type": f"age_within_{g}_posthoc",
                        "group": g,
                        "region": reg,
                        "age_class": f"{a1} vs {a2}",
                        "test": "mwu_pairwise",
                        "statistic": None,
                        "p_value": float(pr),
                        "effect_size": float(delta),
                        "direction": direction,
                    }
                )
            if pvals:
                padj = benjamini_hochberg(np.asarray(pvals, dtype=float))
                for i, r in enumerate(pair_rows):
                    r["p_adj"] = float(padj[i])
                    out.append(r)

    # C: healthy vs patients within each age class
    if "healthy" in groups and "patients" in groups:
        for age in age_classes:
            for reg in regions:
                xh = _subset(rows, group="healthy", region=reg, age_class=age)
                xp = _subset(rows, group="patients", region=reg, age_class=age)
                if xh.size < 5 or xp.size < 5:
                    continue
                p = float(mannwhitneyu(xh, xp, alternative="two-sided").pvalue)
                delta = cliffs_delta(xh, xp)
                direction = "patients>healthy" if float(np.median(xp)) > float(np.median(xh)) else "patients<healthy"
                out.append(
                    {
                        "comparison_type": "health_within_age",
                        "group": "healthy_vs_patients",
                        "region": reg,
                        "age_class": age,
                        "test": "mann_whitney_u",
                        "statistic": None,
                        "p_value": p,
                        "effect_size": float(delta),
                        "direction": direction,
                    }
                )

        # FDR for these tests across all age×region
        idx = [i for i, r in enumerate(out) if r["comparison_type"] == "health_within_age"]
        if idx:
            pvals = np.asarray([out[i]["p_value"] for i in idx], dtype=float)
            padj = benjamini_hochberg(pvals)
            for j, i in enumerate(idx):
                out[i]["p_adj"] = float(padj[j])

    return out


def save_stats(results: list[dict], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    import csv

    fields = sorted({k for r in results for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)

