from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml

from .compute_dfa import compute_dfa_table, save_dfa_rows, save_meta
from .plots import boxplots_by_age, boxplots_health_within_age, heatmap_significance, trend_plot
from .stats import analyze_stats, load_dfa_csv, save_stats


def _load_dfa_yaml(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"DFA config must be a YAML mapping: {path}")
    return data


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
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None, help="YAML (see configs/dfa/default.yml).")
    pre_args, rest = pre.parse_known_args()
    ycfg = _load_dfa_yaml(pre_args.config)

    def _i(key: str, fallback: int) -> int:
        v = ycfg.get(key, fallback)
        return int(v) if v is not None else fallback

    max_yaml = ycfg.get("max_per_class")
    max_default = int(max_yaml) if max_yaml is not None else None

    ap = argparse.ArgumentParser(description="DFA analysis pipeline (healthy vs patients, age, regions).")
    ap.add_argument("--config", type=Path, default=pre_args.config, help="Optional YAML; CLI overrides YAML when passed.")
    ap.add_argument("--data-dir", type=Path, default=Path(str(ycfg.get("healthy_data_dir", "data"))))
    ap.add_argument("--kids-dir", type=Path, default=Path(str(ycfg.get("patients_data_dir", "data_kids"))))
    ap.add_argument("--eyes", type=str, default=str(ycfg.get("eyes", "open")), choices=["open", "closed"])
    ap.add_argument("--max", type=int, default=max_default, help="Debug cap per class (YAML: max_per_class).")
    ap.add_argument("--order", type=int, default=_i("order", 1), choices=[1, 2], help="Polynomial detrending order")
    ap.add_argument("--min-scale", type=int, default=_i("min_scale", 4))
    ap.add_argument("--out-dir", type=Path, default=Path(str(ycfg.get("out_dir", "results/dfa_analysis"))))
    ap.add_argument(
        "--run-name",
        type=str,
        default=ycfg.get("run_name") if ycfg.get("run_name") is not None else None,
        help="Output folder suffix: run_<eyes>_<run_name>. Default: timestamp.",
    )
    args = ap.parse_args(rest)

    ts = args.run_name if args.run_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"run_{args.eyes}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.config is not None and args.config.is_file():
        shutil.copy2(args.config, out_dir / "config_used.yml")

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
    print(f"DFA_OUTPUT_DIR={out_dir.resolve()}", flush=True)
    print(f"[dfa] done: {out_dir}")


if __name__ == "__main__":
    main()

