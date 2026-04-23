from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _sig_kids(run_dir: Path, eyes: str, exp: str) -> set[str]:
    tag = f"kids_{eyes}_{'all_features' if exp == 'all' else 'corr_filter'}"
    tables = run_dir / tag / "tables"
    if not tables.is_dir():
        return set()
    sig: set[str] = set()
    for p in tables.glob(f"{tag}_*_cluster_stats.json"):
        try:
            for row in json.loads(p.read_text(encoding="utf-8")):
                if row.get("significant"):
                    sig.add(str(row["cluster"]))
        except Exception:
            continue
    return sig


def build_stability_table(run_dir: Path) -> list[dict[str, Any]]:
    """
    Step 8: Cluster | closed+all | closed+corr | open+all | open+corr | Stable (all four)
    """
    s_closed_all = _sig_kids(run_dir, "closed", "all")
    s_closed_corr = _sig_kids(run_dir, "closed", "corr")
    s_open_all = _sig_kids(run_dir, "open", "all")
    s_open_corr = _sig_kids(run_dir, "open", "corr")

    all_c = s_closed_all | s_closed_corr | s_open_all | s_open_corr
    rows = []
    for c in sorted(all_c):
        e1 = c in s_closed_all and c in s_open_all  # «все признаки» в обоих условиях глаз
        e2 = c in s_closed_corr and c in s_open_corr  # «фильтр» в обоих условиях
        c1 = c in s_closed_all and c in s_closed_corr  # оба эксперимента, closed
        c2 = c in s_open_all and c in s_open_corr  # оба эксперимента, open
        stable = c in s_closed_all and c in s_closed_corr and c in s_open_all and c in s_open_corr
        rows.append(
            {
                "cluster": c,
                "experiment_all_features_both_eyes": e1,
                "experiment_corr_filter_both_eyes": e2,
                "condition_closed_both_experiments": c1,
                "condition_open_both_experiments": c2,
                "stable_significant_all_4_cells": stable,
            }
        )
    return rows
