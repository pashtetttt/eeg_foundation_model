from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from .bootstrap_subjects import bootstrap_cluster_stability
from .data_loading import load_dataset_with_subjects
from .feature_clusters import aggregate_cluster_matrix, build_clusters, clusters_table
from .intersections import compute_error_domain_intersection
from .metrics_comparison import compare_healthy_vs_patients, confusion_counts, metrics_from_arrays
from .modeling import build_xgb_regularized_pipeline, fit_xgb_balanced
from .report import generate_error_report, write_step8_table
from .stats_utils import benjamini_hochberg, run_tests
from .viz import save_confusion_matrix, save_confusion_pair_side_by_side, save_kde_overlay
from .shap_errors import shap_on_error_subset
from .domain_shift import analyze_domain_shift


def _safe_log10(x: float) -> float:
    return -math.log10(max(1e-300, float(x)))


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _comment_ru(metric: str, band: str | None, region: str) -> str:
    # lightweight interpretation stub; richer text can be added later without changing analytics.
    if metric == "power" and band:
        return f"Возможная интерпретация: различия мощности в диапазоне {band} в регионе {region} могут отражать особенности корковой активности и зрелости ритмов."
    if metric == "ratio":
        return f"Возможная интерпретация: соотношения полос в регионе {region} часто отражают баланс медленных/быстрых ритмов и уровень активации."
    if metric in ("hjorth_complexity", "higuchi_fd", "entropy"):
        return f"Возможная интерпретация: меры сложности/энтропии в регионе {region} могут быть связаны со степенью нерегулярности сигнала и организацией нейронной динамики."
    return "Возможная интерпретация: кластер отражает совокупность признаков, связанных с региональной/спектральной организацией ЭЭГ."


def _split_correct_error_true_class(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> tuple[np.ndarray, np.ndarray]:
    mask = y_true == cls
    idx = np.where(mask)[0]
    correct = idx[y_pred[idx] == cls]
    error = idx[y_pred[idx] != cls]
    return correct, error


def _split_correct_error_pred_class(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> tuple[np.ndarray, np.ndarray]:
    mask = y_pred == cls
    idx = np.where(mask)[0]
    correct = idx[y_true[idx] == cls]
    error = idx[y_true[idx] != cls]
    return correct, error


def _bootstrap_stability(
    *,
    Xc: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    n_iter: int,
    p_adj_thr: float,
    d_thr: float,
) -> np.ndarray:
    """
    Bootstrap stability per cluster:
    in each iteration, resample within each group and recompute MWU p_adj (BH across clusters) + Cohen d threshold.
    Returns stability fraction for each cluster column.
    """
    n_clusters = Xc.shape[1]
    hits = np.zeros(n_clusters, dtype=int)
    if idx_a.size < 5 or idx_b.size < 5:
        return np.zeros(n_clusters, dtype=float)

    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        a = rng.choice(idx_a, size=idx_a.size, replace=True)
        b = rng.choice(idx_b, size=idx_b.size, replace=True)
        pvals = np.ones(n_clusters, dtype=float)
        ds = np.zeros(n_clusters, dtype=float)
        for j in range(n_clusters):
            tr = run_tests(Xc[a, j], Xc[b, j])
            pvals[j] = 1.0 if tr.p_mwu is None else tr.p_mwu
            ds[j] = tr.cohens_d
        padj = benjamini_hochberg(pvals)
        good = (padj < p_adj_thr) & (np.abs(ds) > d_thr)
        hits += good.astype(int)
    return hits / float(n_iter)


def analyze_one_split(
    *,
    tag: str,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    feature_names: list[str],
    canonical_ch_names: list[str] | None,
    out_dir: Path,
    n_bootstrap: int,
    p_adj_thr: float,
    d_thr: float,
) -> None:
    """
    Step 1-3 of the spec on an already evaluated split (OOF for healthy or test for kids).
    Produces confusion matrix + pair index dumps + cluster grouping + per-class FN/FP stats with FDR and bootstrap stability.
    """
    out_tables = _mkdir(out_dir / "tables")
    out_plots = _mkdir(out_dir / "plots")
    out_pairs = _mkdir(out_dir / "pairs")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    save_confusion_matrix(
        cm=cm,
        class_names=class_names,
        out_path=out_plots / f"{tag}_confusion_row_norm.png",
        title=f"{tag}: confusion matrix (row-normalized, %)",
        normalize_rows=True,
        dpi=300,
    )
    np.savetxt(out_tables / f"{tag}_confusion_counts.csv", cm, delimiter=",", fmt="%d")

    # Step 1: problematic pairs (adaptive: rate>=10% AND n>=5)
    pair_rows = []
    row_sums = np.maximum(1, cm.sum(axis=1))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            rate = float(cm[i, j]) / float(row_sums[i])
            if cm[i, j] >= 5 and rate >= 0.10:
                idx = np.where((y_true == i) & (y_pred == j))[0]
                pair_name = f"true_{class_names[i]}__pred_{class_names[j]}"
                np.savetxt(out_pairs / f"{tag}_{pair_name}_indices.csv", idx, delimiter=",", fmt="%d")
                pair_rows.append(
                    {
                        "pair": pair_name,
                        "n": int(cm[i, j]),
                        "row_rate": rate,
                        "true_class": class_names[i],
                        "pred_class": class_names[j],
                        "index_file": f"pairs/{tag}_{pair_name}_indices.csv",
                    }
                )

    (out_tables / f"{tag}_problem_pairs.json").write_text(json.dumps(pair_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # Step 2: clustering
    clusters = build_clusters(feature_names, canonical_ch_names)
    rows = clusters_table(clusters, feature_names)
    (out_tables / f"{tag}_cluster_table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    Xc, cluster_keys, cluster_sizes = aggregate_cluster_matrix(X, clusters, agg="median")
    # nan/inf guard at cluster level
    bad = ~np.isfinite(Xc)
    if np.any(bad):
        Xc = np.where(np.isfinite(Xc), Xc, np.nan)

    # Step 3: per-class stats (FN and FP views)
    for view in ("fn_true_class", "fp_pred_class"):
        for cls, cls_name in enumerate(class_names):
            if view == "fn_true_class":
                idx_ok, idx_err = _split_correct_error_true_class(y_true, y_pred, cls)
                a_label = f"{cls_name}: correct (y_true={cls_name}, y_pred={cls_name})"
                b_label = f"{cls_name}: error (y_true={cls_name}, y_pred!= {cls_name})"
                prefix = f"{tag}_{view}_{cls_name}"
            else:
                idx_ok, idx_err = _split_correct_error_pred_class(y_true, y_pred, cls)
                a_label = f"{cls_name}: correct (y_pred={cls_name}, y_true={cls_name})"
                b_label = f"{cls_name}: error (y_pred={cls_name}, y_true!= {cls_name})"
                prefix = f"{tag}_{view}_{cls_name}"

            if idx_ok.size < 5 or idx_err.size < 5:
                warn = {
                    "tag": tag,
                    "view": view,
                    "class": cls_name,
                    "n_correct": int(idx_ok.size),
                    "n_error": int(idx_err.size),
                    "warning": "Skipped stats: need >=5 samples in each group",
                }
                (out_tables / f"{prefix}_skipped.json").write_text(json.dumps(warn, ensure_ascii=False, indent=2), encoding="utf-8")
                continue

            pvals = np.ones(len(cluster_keys), dtype=float)
            pks = np.ones(len(cluster_keys), dtype=float)
            ds = np.zeros(len(cluster_keys), dtype=float)
            mean_a = np.zeros(len(cluster_keys), dtype=float)
            mean_b = np.zeros(len(cluster_keys), dtype=float)
            na = np.zeros(len(cluster_keys), dtype=int)
            nb = np.zeros(len(cluster_keys), dtype=int)
            dropped = np.zeros(len(cluster_keys), dtype=int)

            for j in range(len(cluster_keys)):
                tr = run_tests(Xc[idx_ok, j], Xc[idx_err, j])
                pvals[j] = 1.0 if tr.p_mwu is None else tr.p_mwu
                pks[j] = 1.0 if tr.p_ks is None else tr.p_ks
                ds[j] = tr.cohens_d
                mean_a[j] = tr.mean_a
                mean_b[j] = tr.mean_b
                na[j] = tr.n_a
                nb[j] = tr.n_b
                dropped[j] = tr.n_drop_naninf

            padj = benjamini_hochberg(pvals)
            stability = _bootstrap_stability(
                Xc=Xc,
                idx_a=idx_ok,
                idx_b=idx_err,
                n_iter=n_bootstrap,
                p_adj_thr=p_adj_thr,
                d_thr=d_thr,
            )

            score = np.array([_safe_log10(p) for p in padj]) * np.abs(ds)
            significant = (padj < p_adj_thr) & (np.abs(ds) > d_thr) & (stability >= 0.70)

            out_rows = []
            for j, k in enumerate(cluster_keys):
                metric = k.split("_", 1)[0]
                band = None
                parts = k.split("_")
                if len(parts) >= 3:
                    band = parts[1]
                region = parts[-1]
                out_rows.append(
                    {
                        "cluster": k,
                        "n_features": int(cluster_sizes[j]),
                        "p_mwu": float(pvals[j]),
                        "p_adj": float(padj[j]),
                        "p_ks": float(pks[j]),
                        "cohens_d": float(ds[j]),
                        "mean_correct": float(mean_a[j]),
                        "mean_error": float(mean_b[j]),
                        "direction": "error>correct" if mean_b[j] > mean_a[j] else "error<correct",
                        "n_correct": int(na[j]),
                        "n_error": int(nb[j]),
                        "n_drop_naninf": int(dropped[j]),
                        "score": float(score[j]),
                        "stability": float(stability[j]),
                        "significant": bool(significant[j]),
                        "comment_ru": _comment_ru(metric=metric, band=band, region=region),
                    }
                )

            out_rows.sort(key=lambda r: (-r["significant"], -r["score"]))
            (out_tables / f"{prefix}_cluster_stats.json").write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")

            # KDE plots for top-5 significant clusters (adaptive: only those passing thresholds)
            top_sig = [r for r in out_rows if r["significant"]][:5]
            for r in top_sig:
                j = cluster_keys.index(r["cluster"])
                a = Xc[idx_ok, j]
                b = Xc[idx_err, j]
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if a.size < 5 or b.size < 5:
                    continue
                save_kde_overlay(
                    a=a,
                    b=b,
                    label_a="Correct",
                    label_b="Error",
                    title=f"{tag} | {view} | {cls_name}\n{r['cluster']} (p_adj={r['p_adj']:.2e}, d={r['cohens_d']:.2f}, stab={r['stability']:.2f})",
                    xlabel="Cluster aggregated value (median)",
                    out_path=out_plots / f"{prefix}_kde_{r['cluster']}.png",
                    dpi=300,
                )


def main() -> None:
    # reduce noisy warnings while keeping actual exceptions visible
    warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description="False-positive / false-negative error analysis (no leakage, GroupKFold, clusters, FDR, bootstrap).")
    ap.add_argument("--data-dir", type=Path, default=Path("data"), help="Healthy dataset root (train/validation)")
    ap.add_argument("--kids-dir", type=Path, default=Path("data_kids"), help="Kids/sick dataset root (test only)")
    ap.add_argument("--out-root", type=Path, default=Path("results/error_analysis"), help="Root output directory under results/")
    ap.add_argument("--max", type=int, default=None, help="Debug cap per class")
    ap.add_argument("--eyes", type=str, default="both", choices=["closed", "open", "both"], help="Which eyes condition(s) to analyze")
    ap.add_argument("--experiment", type=str, default="both", choices=["all", "corr", "both"], help="Which feature experiment(s) to run")
    ap.add_argument("--corr-threshold", type=float, default=0.95)
    ap.add_argument("--bootstrap-iters", type=int, default=50)
    ap.add_argument("--p-adj-thr", type=float, default=0.01)
    ap.add_argument("--d-thr", type=float, default=0.5)
    ap.add_argument("--shap", action="store_true", help="Compute SHAP on errors (no leakage: per-fold on healthy OOF, full-fit on kids)")
    ap.add_argument("--shap-max", type=int, default=300, help="Max error samples per analysis unit for SHAP")
    ap.add_argument(
        "--bootstrap-subjects",
        action="store_true",
        help="Step 9: bootstrap by subjects (50 iters) for cluster stability — can be slow",
    )
    ap.add_argument(
        "--report-path",
        type=Path,
        default=Path("results/reports/full_error_analysis.md"),
        help="Step 10: path for combined markdown report",
    )
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _mkdir(args.out_root / f"fp_fn_{ts}")
    t_global = time.perf_counter()
    print(f"[error_analysis] started at {ts}")
    print(f"[error_analysis] out_dir={run_dir}")
    print(f"[error_analysis] data={args.data_dir} kids={args.kids_dir}")
    print(f"[error_analysis] eyes={args.eyes} experiment={args.experiment} corr_thr={args.corr_threshold}")
    print(f"[error_analysis] bootstrap={args.bootstrap_iters} p_adj<{args.p_adj_thr} |d|>{args.d_thr} shap={args.shap}")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "started_at": ts,
                "data_dir": str(args.data_dir),
                "kids_dir": str(args.kids_dir),
                "model": "xgb_regularized",
                "experiments": ["all_features", "corr_filter_features"],
                "eyes_conditions": ["closed", "open"],
                "corr_threshold": args.corr_threshold,
                "bootstrap_iters": args.bootstrap_iters,
                "p_adj_thr": args.p_adj_thr,
                "d_thr": args.d_thr,
                "no_leakage": True,
                "validation": "GroupKFold by subject_id (healthy)",
                "test": "kids_dir (not used in training)",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    eyes_list = ("closed", "open") if args.eyes == "both" else (args.eyes,)
    exp_list = ("all", "corr") if args.experiment == "both" else (args.experiment,)

    for eyes in eyes_list:
        t0 = time.perf_counter()
        print(f"\n[stage] load healthy ({eyes}) …")
        ds = load_dataset_with_subjects(data_dir=args.data_dir, eyes=eyes, max_per_group=args.max)
        print(f"[ok] healthy loaded: n={ds.X.shape[0]} p={ds.X.shape[1]} unique_subjects={len(set(ds.subject_ids))} in {time.perf_counter()-t0:.1f}s")
        groups = np.asarray(ds.subject_ids)

        # group CV predictions for healthy
        gkf = GroupKFold(n_splits=5)

        for exp in exp_list:
            use_corr = exp == "corr"
            tag = f"healthy_{eyes}_{'all_features' if exp == 'all' else 'corr_filter'}"
            out_dir = _mkdir(run_dir / tag)
            print(f"\n[stage] healthy OOF GroupKFold: {tag} (use_corr={use_corr})")

            y_pred_oof = np.empty_like(ds.y)
            X_used = ds.X  # raw features for analysis; model may apply corr filter inside pipeline
            shap_dir = _mkdir(out_dir / "shap") if args.shap else None

            for fold, (tr, va) in enumerate(gkf.split(ds.X, ds.y, groups=groups), start=1):
                t_fold = time.perf_counter()
                print(f"[fold {fold}/5] fit: train_n={tr.size} val_n={va.size} …", flush=True)
                pipe = build_xgb_regularized_pipeline(use_corr_filter=use_corr, corr_threshold=args.corr_threshold)
                fit_xgb_balanced(pipe, ds.X[tr], ds.y[tr])
                y_pred_oof[va] = pipe.predict(ds.X[va])
                n_err = int((y_pred_oof[va] != ds.y[va]).sum())
                print(f"[fold {fold}/5] done: val_errors={n_err} in {time.perf_counter()-t_fold:.1f}s", flush=True)

                # Save fold indices (for reproducibility / later SHAP extension)
                np.savetxt(out_dir / f"fold_{fold}_val_indices.csv", va, delimiter=",", fmt="%d")

                # Step 4: SHAP only on fold errors (OOF, no leakage)
                if args.shap and shap_dir is not None:
                    err_idx = va[(y_pred_oof[va] != ds.y[va])]
                    if err_idx.size >= 5:
                        print(f"[fold {fold}/5] SHAP on errors: n={err_idx.size} …", flush=True)
                        shap_on_error_subset(
                            fitted_pipeline=pipe,
                            X_raw=ds.X,
                            feature_names=ds.feature_names,
                            canonical_ch_names=ds.canonical_ch_names,
                            error_indices=err_idx,
                            out_dir=shap_dir,
                            tag=f"{tag}_fold{fold}",
                            max_samples=args.shap_max,
                            dpi=300,
                        )

            print(f"[stage] stats+plots for healthy: {tag} …", flush=True)
            analyze_one_split(
                tag=tag,
                X=X_used,
                y_true=ds.y,
                y_pred=y_pred_oof,
                class_names=ds.class_names,
                feature_names=ds.feature_names,
                canonical_ch_names=ds.canonical_ch_names,
                out_dir=out_dir,
                n_bootstrap=args.bootstrap_iters,
                p_adj_thr=args.p_adj_thr,
                d_thr=args.d_thr,
            )
            print(f"[ok] healthy analysis saved: {out_dir}", flush=True)

            # Step 9: bootstrap by subjects (healthy OOF)
            if args.bootstrap_subjects:
                print(f"[stage] bootstrap by subjects (Step 9): {tag} …", flush=True)
                clusters_h = build_clusters(ds.feature_names, ds.canonical_ch_names)
                Xc_h, keys_h, _sizes_h = aggregate_cluster_matrix(ds.X, clusters_h, agg="median")
                bad = ~np.isfinite(Xc_h)
                if np.any(bad):
                    Xc_h = np.where(np.isfinite(Xc_h), Xc_h, np.nan)
                boot_t = _mkdir(out_dir / "tables")
                bootstrap_cluster_stability(
                    Xc=Xc_h,
                    cluster_keys=keys_h,
                    y_true=ds.y,
                    y_pred=y_pred_oof,
                    subject_ids=ds.subject_ids,
                    n_iter=50,
                    p_adj_thr=args.p_adj_thr,
                    d_thr=args.d_thr,
                    out_path=boot_t / f"{tag}_bootstrap_subject_stability.json",
                )
                print(f"[ok] bootstrap stability saved: {boot_t / f'{tag}_bootstrap_subject_stability.json'}", flush=True)

            # kids test (train on all healthy, test on kids)
            print(f"\n[stage] load kids ({eyes}) …", flush=True)
            t_k = time.perf_counter()
            kids = load_dataset_with_subjects(data_dir=args.kids_dir, eyes=eyes, max_per_group=args.max)
            print(f"[ok] kids loaded: n={kids.X.shape[0]} p={kids.X.shape[1]} unique_subjects={len(set(kids.subject_ids))} in {time.perf_counter()-t_k:.1f}s")
            tag_k = f"kids_{eyes}_{'all_features' if exp == 'all' else 'corr_filter'}"
            out_dir_k = _mkdir(run_dir / tag_k)

            print(f"[stage] train on healthy-all, test on kids: {tag_k} …", flush=True)
            pipe = build_xgb_regularized_pipeline(use_corr_filter=use_corr, corr_threshold=args.corr_threshold)
            fit_xgb_balanced(pipe, ds.X, ds.y)
            y_pred_k = pipe.predict(kids.X)

            analyze_one_split(
                tag=tag_k,
                X=kids.X,
                y_true=kids.y,
                y_pred=y_pred_k,
                class_names=kids.class_names,
                feature_names=kids.feature_names,
                canonical_ch_names=kids.canonical_ch_names,
                out_dir=out_dir_k,
                n_bootstrap=args.bootstrap_iters,
                p_adj_thr=args.p_adj_thr,
                d_thr=args.d_thr,
            )
            print(f"[ok] kids analysis saved: {out_dir_k}", flush=True)

            # Step 5: metrics + side-by-side confusion matrices
            print(f"[stage] Step 5: metrics healthy vs patients ({eyes}, {exp}) …", flush=True)
            m_h = metrics_from_arrays(ds.y, y_pred_oof, ds.class_names)
            m_p = metrics_from_arrays(kids.y, y_pred_k, kids.class_names)
            cmp_path = run_dir / f"step5_{eyes}_{exp}_healthy_vs_patients.json"
            compare_healthy_vs_patients(
                metrics_healthy=m_h,
                metrics_patients=m_p,
                class_names=ds.class_names,
                out_path=cmp_path,
            )
            cm_h = confusion_counts(ds.y, y_pred_oof, len(ds.class_names))
            cm_k = confusion_counts(kids.y, y_pred_k, len(kids.class_names))
            save_confusion_pair_side_by_side(
                cm_left=cm_h,
                cm_right=cm_k,
                class_names=ds.class_names,
                out_path=run_dir / f"step5_{eyes}_{exp}_confusion_healthy_vs_patients.png",
                title_left=f"Healthy OOF ({tag})",
                title_right=f"Patients ({tag_k})",
                normalize_rows=True,
                dpi=300,
            )
            print(f"[ok] Step 5 saved: {cmp_path}", flush=True)

            # SHAP on kids errors (trained on all healthy; ok as test-only)
            if args.shap:
                shap_dir_k = _mkdir(out_dir_k / "shap")
                err_k = np.where(y_pred_k != kids.y)[0]
                if err_k.size >= 5:
                    print(f"[stage] kids SHAP on errors: n={err_k.size} …", flush=True)
                    shap_on_error_subset(
                        fitted_pipeline=pipe,
                        X_raw=kids.X,
                        feature_names=kids.feature_names,
                        canonical_ch_names=kids.canonical_ch_names,
                        error_indices=err_k,
                        out_dir=shap_dir_k,
                        tag=f"{tag_k}_all_errors",
                        max_samples=args.shap_max,
                        dpi=300,
                    )

            # Step 6: domain shift (KS on clusters): X_train healthy vs X_patients
            print(f"[stage] Step 6 domain shift (KS): eyes={eyes} exp={exp} …", flush=True)
            shift_dir = _mkdir(run_dir / f"domain_shift_{eyes}_{'all_features' if exp == 'all' else 'corr_filter'}")
            dom_tag = f"domain_shift_{eyes}_{'all_features' if exp == 'all' else 'corr_filter'}"
            analyze_domain_shift(
                tag=dom_tag,
                X_train=ds.X,
                X_patients=kids.X,
                feature_names=ds.feature_names,
                canonical_ch_names_healthy=ds.canonical_ch_names,
                out_dir=shift_dir,
                ks_p_thr=0.01,
                dpi=300,
            )
            print(f"[ok] domain shift (KS) saved: {shift_dir}", flush=True)

            # Step 7: intersection error clusters (kids) ∩ domain shift
            print(f"[stage] Step 7 intersection …", flush=True)
            inter_p = run_dir / f"step7_intersection_{eyes}_{exp}.json"
            compute_error_domain_intersection(
                kids_out_dir=out_dir_k,
                domain_shift_dir=shift_dir,
                tag_kids=tag_k,
                tag_domain=dom_tag,
                out_path=inter_p,
            )
            print(f"[ok] intersection saved: {inter_p}", flush=True)

    print(f"\n[stage] Step 8: cross-experiment stability table …", flush=True)
    write_step8_table(run_dir)
    print(f"[ok] Step 8: {run_dir / 'step8_experiment_condition_stability.json'}", flush=True)

    print(f"\n[stage] Step 10: generate report → {args.report_path} …", flush=True)
    generate_error_report(run_dir=run_dir, reports_dir=args.report_path.parent, filename=args.report_path.name)
    print(f"[ok] report: {args.report_path}", flush=True)

    print(f"\n[error_analysis] finished in {time.perf_counter()-t_global:.1f}s")


if __name__ == "__main__":
    main()

