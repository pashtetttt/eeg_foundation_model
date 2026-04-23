from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _interpret_cluster_ru(cluster: str) -> str:
    parts = cluster.split("_")
    metric = parts[0] if parts else cluster
    region = parts[-1] if len(parts) > 1 else ""
    if metric == "power":
        return f"Гипотеза: сдвиг спектральной мощности ({cluster}) может менять разделимость возрастных классов на фоне изменений корковых ритмов в регионе {region}."
    if metric == "ratio":
        return f"Гипотеза: сдвиг соотношений полос ({cluster}) отражает перекос медленных/быстрых компонентов, важный для возрастной классификации."
    if metric in ("hjorth", "higuchi", "entropy"):
        return f"Гипотеза: изменение показателей сложности/нерегулярности ({cluster}) может быть связано с различиями в организации ЭЭГ между группами."
    return f"Гипотеза: кластер {cluster} объединяет схожие по смыслу признаки; совместное смещение и ошибки модели указывают на уязвимость модели к этому типу сигнала."


def collect_significant_error_clusters(kids_tables_dir: Path, tag_prefix: str) -> set[str]:
    """Union of clusters with significant==True from fn_true_class and fp_pred_class JSON files."""
    sig: set[str] = set()
    if not kids_tables_dir.is_dir():
        return sig
    for p in kids_tables_dir.glob(f"{tag_prefix}_*_cluster_stats.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for row in data:
                if row.get("significant"):
                    sig.add(str(row["cluster"]))
        except Exception:
            continue
    return sig


def load_significant_domain_shift(domain_shift_dir: Path, tag: str) -> set[str]:
    """Clusters with KS p < 0.01 from Step 6 table."""
    p = domain_shift_dir / f"{tag}_domain_shift_ks_table.json"
    if not p.is_file():
        return set()
    rows = json.loads(p.read_text(encoding="utf-8"))
    return {r["cluster"] for r in rows if r.get("significant_ks_p_lt_0.01")}


def compute_error_domain_intersection(
    *,
    kids_out_dir: Path,
    domain_shift_dir: Path,
    tag_kids: str,
    tag_domain: str,
    out_path: Path,
) -> list[dict[str, Any]]:
    """
    Step 7: intersection of error-associated clusters (kids, Step 3) and domain shift (Step 6).
    """
    tables = kids_out_dir / "tables"
    err_clusters = collect_significant_error_clusters(tables, tag_kids)
    dom_clusters = load_significant_domain_shift(domain_shift_dir, tag_domain)

    inter = sorted(err_clusters & dom_clusters)
    out_rows = []
    for c in inter:
        out_rows.append(
            {
                "cluster": c,
                "interpretation_ru": _interpret_cluster_ru(c),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_rows
