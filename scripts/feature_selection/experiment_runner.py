from __future__ import annotations

"""
Feature Selection Experiments: сравнение стратегий работы с коррелированными кластерами признаков.

Требования:
- модульность: класс FeatureSelectionExperiments
- утечка: стратегии Exp3 и Exp4 обучаются/выбирают признаки ТОЛЬКО на train-fold
- валидация: GroupKFold по субъектам
- логирование: CSV + текстовый лог; фиксация random_state
"""

import csv
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from scripts.error_analysis.modeling import XGB_REGULARIZED_PARAMS
from .selection_strategies import (
    baseline_all_features,
    cluster_aggregation_median,
    cluster_representative_one_feature,
    topn_by_xgb_importance,
)


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int = 42
    n_splits: int = 5
    rep_criterion: str = "variance"  # variance|corr_y
    topn_keep: int = 100


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    n_features: int
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    train_time_s_mean: float
    train_time_s_std: float
    fold_metrics: list[dict[str, Any]]
    feature_names: list[str]
    # optional per-fold indices for subset-based experiments
    selected_indices_by_fold: list[list[int]] | None


class FeatureSelectionExperiments:
    def __init__(self, cfg: ExperimentConfig | None = None):
        self.cfg = cfg or ExperimentConfig()
        self.results: dict[str, ExperimentResult] = {}
        self.meta: dict[str, Any] = {}

    def _build_xgb(self) -> XGBClassifier:
        return XGBClassifier(
            random_state=self.cfg.random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
            **XGB_REGULARIZED_PARAMS,
        )

    def _fit_predict_one_fold(self, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", self._build_xgb())])
        sw = compute_sample_weight("balanced", y_tr)
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
        train_s = time.perf_counter() - t0
        y_pred = pipe.predict(X_va)
        importances = pipe.named_steps["clf"].feature_importances_
        if importances is None:
            importances = np.zeros(X_tr.shape[1], dtype=float)
        return y_pred, float(train_s), np.asarray(importances, dtype=float)

    def run_all_experiments(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        clusters_dict: dict[str, list[int]],
        feature_names: list[str],
    ) -> dict[str, ExperimentResult]:
        """
        Run 4 experiments under GroupKFold CV.
        Returns dict of results (also stored in self.results).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        groups = np.asarray(groups)
        if X.shape[0] != y.shape[0] or X.shape[0] != groups.shape[0]:
            raise ValueError("X, y, groups must have same n_samples")

        self.meta = {
            "random_state": self.cfg.random_state,
            "n_splits": self.cfg.n_splits,
            "rep_criterion": self.cfg.rep_criterion,
            "topn_keep": self.cfg.topn_keep,
            "n_samples": int(X.shape[0]),
            "n_features_in": int(X.shape[1]),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "xgb_params": dict(XGB_REGULARIZED_PARAMS),
        }

        gkf = GroupKFold(n_splits=self.cfg.n_splits)
        folds = list(gkf.split(X, y, groups=groups))

        def _run(name: str, selector_fn) -> ExperimentResult:
            accs, f1s, times = [], [], []
            fold_rows: list[dict[str, Any]] = []
            selected_by_fold: list[list[int]] | None = []
            feat_names_out: list[str] | None = None
            fold_importances: list[list[tuple[str, float]]] = []

            for fold_idx, (tr, va) in enumerate(folds, start=1):
                X_tr0, y_tr = X[tr], y[tr]
                X_va0, y_va = X[va], y[va]

                sel = selector_fn(X_tr0, y_tr, X_va0)
                y_pred, train_s, imp = self._fit_predict_one_fold(sel.X_train, y_tr, sel.X_val)
                # сохраняем топ важностей для анализа стабильности (адаптивно: top 20% по важности)
                if imp.size == len(sel.feature_names):
                    pairs = list(zip(sel.feature_names, imp.tolist()))
                else:
                    pairs = list(zip(sel.feature_names[: imp.size], imp[: len(sel.feature_names)].tolist()))
                pairs.sort(key=lambda t: -t[1])
                fold_importances.append(pairs)

                acc = float(accuracy_score(y_va, y_pred))
                mf1 = float(f1_score(y_va, y_pred, average="macro", zero_division=0))
                accs.append(acc)
                f1s.append(mf1)
                times.append(train_s)
                fold_rows.append(
                    {
                        "fold": fold_idx,
                        "train_n": int(tr.size),
                        "val_n": int(va.size),
                        "n_features": int(sel.X_train.shape[1]),
                        "accuracy": acc,
                        "macro_f1": mf1,
                        "train_time_s": train_s,
                    }
                )

                if feat_names_out is None:
                    feat_names_out = sel.feature_names
                # if selection differs per fold, we still store fold-wise indices when available
                if sel.selected_indices is None:
                    selected_by_fold = None
                elif selected_by_fold is not None:
                    selected_by_fold.append(sel.selected_indices.tolist())

            assert feat_names_out is not None
            return ExperimentResult(
                name=name,
                n_features=int(len(feat_names_out)),
                accuracy_mean=float(np.mean(accs)),
                accuracy_std=float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                macro_f1_mean=float(np.mean(f1s)),
                macro_f1_std=float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                train_time_s_mean=float(np.mean(times)),
                train_time_s_std=float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
                fold_metrics=fold_rows,
                feature_names=feat_names_out,
                selected_indices_by_fold=selected_by_fold,
            )

        # Exp1: baseline
        def sel1(X_tr0, y_tr, X_va0):
            return baseline_all_features(X_tr0, X_va0, feature_names)

        # Exp2: cluster aggregation
        def sel2(X_tr0, y_tr, X_va0):
            return cluster_aggregation_median(X_tr0, X_va0, clusters_dict, feature_names=feature_names)

        # Exp3: representative per cluster (fit on train only)
        def sel3(X_tr0, y_tr, X_va0):
            return cluster_representative_one_feature(
                X_train=X_tr0,
                y_train=y_tr,
                X_val=X_va0,
                clusters_dict=clusters_dict,
                feature_names=feature_names,
                criterion=self.cfg.rep_criterion,
            )

        # Exp4: top-N by importance (fit on train only)
        def sel4(X_tr0, y_tr, X_va0):
            return topn_by_xgb_importance(
                X_train=X_tr0,
                y_train=y_tr,
                X_val=X_va0,
                feature_names=feature_names,
                n_keep=self.cfg.topn_keep,
                random_state=self.cfg.random_state,
            )

        self.results = {
            "exp1_baseline_all": _run("exp1_baseline_all", sel1),
            "exp2_cluster_agg_median": _run("exp2_cluster_agg_median", sel2),
            f"exp3_cluster_rep_{self.cfg.rep_criterion}": _run(f"exp3_cluster_rep_{self.cfg.rep_criterion}", sel3),
            f"exp4_topn_importance_{self.cfg.topn_keep}": _run(f"exp4_topn_importance_{self.cfg.topn_keep}", sel4),
        }
        return self.results

    def save_results(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # summary csv
        summary_path = output_dir / "experiment_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "experiment",
                    "n_features",
                    "accuracy_mean",
                    "accuracy_std",
                    "macro_f1_mean",
                    "macro_f1_std",
                    "train_time_s_mean",
                    "train_time_s_std",
                ],
            )
            w.writeheader()
            for k, r in self.results.items():
                w.writerow(
                    {
                        "experiment": r.name,
                        "n_features": r.n_features,
                        "accuracy_mean": f"{r.accuracy_mean:.6f}",
                        "accuracy_std": f"{r.accuracy_std:.6f}",
                        "macro_f1_mean": f"{r.macro_f1_mean:.6f}",
                        "macro_f1_std": f"{r.macro_f1_std:.6f}",
                        "train_time_s_mean": f"{r.train_time_s_mean:.4f}",
                        "train_time_s_std": f"{r.train_time_s_std:.4f}",
                    }
                )

        # feature lists
        feat_lists = {k: {"feature_names": r.feature_names, "selected_indices_by_fold": r.selected_indices_by_fold} for k, r in self.results.items()}
        (output_dir / "feature_lists.json").write_text(json.dumps(feat_lists, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "run_meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # text log (human-friendly)
        log_lines = []
        log_lines.append("Feature Selection Experiments (XGBoost regularized, GroupKFold)\n")
        log_lines.append(json.dumps(self.meta, ensure_ascii=False, indent=2))
        log_lines.append("\n\nSummary:\n")
        for k, r in self.results.items():
            log_lines.append(
                f"- {r.name}: n_features={r.n_features} "
                f"acc={r.accuracy_mean:.4f}±{r.accuracy_std:.4f} "
                f"macro_f1={r.macro_f1_mean:.4f}±{r.macro_f1_std:.4f} "
                f"train_s={r.train_time_s_mean:.2f}±{r.train_time_s_std:.2f}\n"
            )
        (output_dir / "experiment_log.txt").write_text("".join(log_lines), encoding="utf-8")

    @staticmethod
    def save_cluster_membership(
        *,
        output_dir: Path,
        clusters_dict: dict[str, list[int]],
        original_feature_names: list[str],
    ) -> Path:
        """
        Важно для интерпретации: сохраняем, какие именно исходные признаки входят в каждый кластер.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        rows: dict[str, list[str]] = {}
        for k in sorted(clusters_dict.keys()):
            idxs = clusters_dict[k]
            rows[k] = [original_feature_names[i] for i in idxs]
        out = output_dir / "clusters_membership.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

