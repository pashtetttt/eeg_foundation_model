from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .compute_dfa import compute_dfa_table, save_dfa_rows, save_meta
from .plots import boxplots_by_age, boxplots_health_within_age, heatmap_significance, trend_plot
from .stats import analyze_stats, load_dfa_csv, save_stats


def _comment_ru_for_effect(row: dict) -> str:
    # Небольшая авто-интерпретация; для отчёта достаточно как гипотеза.
    reg = row.get("region", "")
    age = row.get("age_class", "")
    direction = row.get("direction", "")
    return (
        f"Гипотеза: различия DFA-альфы в регионе {reg} (возраст: {age}) "
        f"могут отражать изменения долгосрочных корреляций в нейронной активности; "
        f"направление эффекта: {direction}."
    )


def generate_report(*, out_dir: Path, meta: dict, stats_rows: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.md"
    lines: list[str] = []
    lines.append("# DFA analysis report\n\n")
    lines.append("## Метод DFA и интерпретация\n")
    lines.append(
        "- DFA \\(\\alpha\\)-экспонента: \\(0.5\\) ~ белый шум, \\(\\approx 1.0\\) ~ 1/f, "
        "> 1.0 может указывать на более сильную нестационарность/долгосрочные корреляции.\n\n"
    )
    lines.append("## Параметры расчёта\n\n")
    lines.append("```json\n")
    lines.append(json.dumps(meta, ensure_ascii=False, indent=2))
    lines.append("\n```\n\n")

    sig = [r for r in stats_rows if r.get("p_adj") is not None and float(r["p_adj"]) < 0.05]
    lines.append(f"## Значимые эффекты (p_adj < 0.05): {len(sig)}\n\n")
    for r in sig[:80]:
        lines.append(
            f"- **{r.get('comparison_type')}** | region={r.get('region')} | age={r.get('age_class')} | "
            f"test={r.get('test')} | p_adj={r.get('p_adj')} | effect={r.get('effect_size')} | {r.get('direction','')}\n"
        )
        lines.append(f"  - { _comment_ru_for_effect(r) }\n")

    lines.append("\n## Ограничения\n")
    lines.append("- Длина сигнала и артефакты могут влиять на оценку DFA.\n")
    lines.append("- Множественные сравнения контролируются FDR (Benjamini–Hochberg).\n")
    lines.append("- Выбросы: выполнена грубая фильтрация каналов по MAD-эвристике.\n")
    report.write_text("".join(lines), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="DFA analysis pipeline (healthy vs patients, age, regions).")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--kids-dir", type=Path, default=Path("data_kids"))
    ap.add_argument("--eyes", type=str, default="open", choices=["open", "closed"])
    ap.add_argument("--max", type=int, default=None, help="Debug cap per class")
    ap.add_argument("--order", type=int, default=1, choices=[1, 2], help="Polynomial detrending order")
    ap.add_argument("--min-scale", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=Path("results/dfa_analysis"))
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"run_{args.eyes}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dfa] compute healthy ({args.data_dir})…")
    rows_h, meta_h = compute_dfa_table(
        data_dir=args.data_dir,
        group_label="healthy",
        eyes=args.eyes,
        max_per_class=args.max,
        min_scale=args.min_scale,
        order=args.order,
    )
    print(f"[dfa] compute patients ({args.kids_dir})…")
    rows_k, meta_k = compute_dfa_table(
        data_dir=args.kids_dir,
        group_label="patients",
        eyes=args.eyes,
        max_per_class=args.max,
        min_scale=args.min_scale,
        order=args.order,
    )
    rows = rows_h + rows_k

    out_csv = out_dir / "dfa_table.csv"
    save_dfa_rows(rows, out_csv)
    meta = {"healthy": meta_h, "patients": meta_k}
    save_meta(meta, out_dir / "meta.json")

    print("[dfa] stats…")
    rows_loaded = load_dfa_csv(out_csv)
    stats_rows = analyze_stats(rows_loaded)
    save_stats(stats_rows, out_dir / "stats.json", out_dir / "stats.csv")

    print("[dfa] plots…")
    boxplots_by_age(rows_loaded, group_label="healthy", out_path=out_dir / "plots" / "box_healthy_by_age.png")
    boxplots_by_age(rows_loaded, group_label="patients", out_path=out_dir / "plots" / "box_patients_by_age.png")
    boxplots_health_within_age(rows_loaded, out_path=out_dir / "plots" / "box_health_within_age.png")
    heatmap_significance(stats_rows, out_path=out_dir / "plots" / "heatmap_health_significance.png")
    trend_plot(rows_loaded, out_path=out_dir / "plots" / "trend_age_health.png")

    print("[dfa] report…")
    generate_report(out_dir=out_dir, meta=meta, stats_rows=stats_rows)
    print(f"[dfa] done: {out_dir}")


if __name__ == "__main__":
    main()

