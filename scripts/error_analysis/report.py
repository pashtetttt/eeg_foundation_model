from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_error_report(
    *,
    run_dir: Path,
    reports_dir: Path,
    filename: str = "full_error_analysis.md",
) -> Path:
    """
    Step 10: aggregate JSON outputs into one markdown report (Russian headings).
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / filename

    lines: list[str] = []
    lines.append("# ОТЧЁТ ПО ОШИБКАМ МОДЕЛИ (XGBoost regularized)\n")
    lines.append(f"Папка запуска: `{run_dir}`\n")

    meta_p = run_dir / "meta.json"
    if meta_p.is_file():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        lines.append("## Параметры запуска\n")
        lines.append(f"- Данные здоровых: `{meta.get('data_dir')}`\n")
        lines.append(f"- Данные больных: `{meta.get('kids_dir')}`\n")
        lines.append(f"- Бутстрап (внутри кластерного анализа ошибок): {meta.get('bootstrap_iters')} итераций\n\n")

    for eyes in ("closed", "open"):
        for exp in ("all", "corr"):
            exp_label = "все признаки" if exp == "all" else "после корреляционного фильтра"
            tag_h = f"healthy_{eyes}_{'all_features' if exp == 'all' else 'corr_filter'}"
            tag_k = f"kids_{eyes}_{'all_features' if exp == 'all' else 'corr_filter'}"
            tag_dom = f"domain_shift_{eyes}_{'all_features' if exp == 'all' else 'corr_filter'}"

            lines.append(f"---\n\n## Группа: здоровые / больные | Условие: **{eyes}** | Эксперимент: **{exp_label}**\n")

            cmp_p = run_dir / f"step5_{eyes}_{exp}_healthy_vs_patients.json"
            if cmp_p.is_file():
                cmp_data = json.loads(cmp_p.read_text(encoding="utf-8"))
                h, p = cmp_data["healthy"], cmp_data["patients"]
                lines.append("### Сводка метрик (здоровые OOF vs больные тест)\n")
                lines.append(
                    f"- Всего сэмплов: здоровые {h['n_samples']}, больные {p['n_samples']}\n"
                )
                lines.append(
                    f"- Ошибок: здоровые {h['n_errors']} ({h['error_rate_pct']:.1f}%), "
                    f"больные {p['n_errors']} ({p['error_rate_pct']:.1f}%)\n"
                )
                lines.append(
                    f"- Accuracy: здоровые {h['accuracy']:.4f}, больные {p['accuracy']:.4f} "
                    f"(Δ%: {cmp_data['relative_change_pct']['accuracy']:.1f}%)\n"
                )
                lines.append(
                    f"- F1-macro: здоровые {h['macro_f1']:.4f}, больные {p['macro_f1']:.4f} "
                    f"(Δ%: {cmp_data['relative_change_pct']['macro_f1']:.1f}%)\n"
                )
                lines.append("\n**Классы с наибольшим падением (по F1, Δ%):**\n")
                for w in cmp_data.get("worst_degraded_classes", [])[:4]:
                    lines.append(
                        f"- `{w['class']}`: ΔF1% = {w['f1_delta_pct']:.1f}, ΔRecall% = {w['recall_delta_pct']:.1f}\n"
                    )
                lines.append("\n")

            # Problem pairs (kids)
            pp_k = run_dir / tag_k / "tables" / f"{tag_k}_problem_pairs.json"
            if pp_k.is_file():
                pairs = json.loads(pp_k.read_text(encoding="utf-8"))
                lines.append("### Путающиеся классы (больные, частые ошибки)\n")
                for row in pairs[:12]:
                    lines.append(
                        f"- {row.get('true_class')} → {row.get('pred_class')}: n={row.get('n')}, "
                        f"доля строки ≈ {row.get('row_rate', 0)*100:.1f}%\n"
                    )
                lines.append("\n")

            # Intersection step 7
            inter_p = run_dir / f"step7_intersection_{eyes}_{exp}.json"
            if inter_p.is_file():
                inter = json.loads(inter_p.read_text(encoding="utf-8"))
                lines.append("### Кандидаты: пересечение «ошибки на больных» и domain shift (KS p<0.01)\n")
                for row in inter[:30]:
                    lines.append(f"- **{row['cluster']}**: {row.get('interpretation_ru', '')}\n")
                lines.append("\n")

            # Domain shift summary
            dom_json = run_dir / tag_dom / f"{tag_dom}_domain_shift_ks_table.json"
            if dom_json.is_file():
                dom = json.loads(dom_json.read_text(encoding="utf-8"))
                sig = [r for r in dom if r.get("significant_ks_p_lt_0.01")]
                lines.append(f"### Domain shift (KS): значимых кластеров (p<0.01): **{len(sig)}**\n")
                for r in sig[:15]:
                    lines.append(
                        f"- `{r['cluster']}`: D={r['ks_statistic']}, p={r['p_value']}, "
                        f"Δmean={r['mean_diff_patients_minus_healthy']:.4g}\n"
                    )
                lines.append("\n")

            # Bootstrap stability file
            boot_p = run_dir / tag_h / "tables" / f"{tag_h}_bootstrap_subject_stability.json"
            if boot_p.is_file():
                boot = json.loads(boot_p.read_text(encoding="utf-8"))
                high = [r for r in boot if r.get("passes_0.70")]
                lines.append("### Бутстрап по субъектам (стабильность кластеров Step 3): ≥70%\n")
                for r in high[:25]:
                    lines.append(f"- `{r['cluster']}`: стабильность = {r['stability']:.2f}\n")
                lines.append("\n")

    # Step 8 table
    stab_p = run_dir / "step8_experiment_condition_stability.json"
    if stab_p.is_file():
        stab = json.loads(stab_p.read_text(encoding="utf-8"))
        lines.append("## Сравнение экспериментов и условий (кластеры значимы на больных)\n")
        stable = [r for r in stab if r.get("stable_significant_all_4_cells")]
        lines.append(
            f"*Устойчивые (значимы во всех 4 ячейках: closed/open × all/corr): **{len(stable)}***\n\n"
        )
        for r in stable[:50]:
            lines.append(f"- `{r['cluster']}`\n")
        lines.append("\n")

    lines.append("\n---\n*Отчёт сгенерирован автоматически. Интерпретации — гипотезы для проверки, не клинический диагноз.*\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def write_step8_table(run_dir: Path) -> Path:
    from .experiment_compare import build_stability_table

    rows = build_stability_table(run_dir)
    p = run_dir / "step8_experiment_condition_stability.json"
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
