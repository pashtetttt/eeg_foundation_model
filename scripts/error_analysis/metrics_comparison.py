from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def compute_metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    labels = np.arange(len(class_names))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = []
    for i, name in enumerate(class_names):
        per_class.append(
            {
                "class": name,
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(sup[i]),
            }
        )
    n_err = int(np.sum(y_true != y_pred))
    n = int(y_true.size)
    return {
        "n_samples": n,
        "n_errors": n_err,
        "error_rate_pct": 100.0 * n_err / max(1, n),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def relative_change_pct(patients_val: float, healthy_val: float) -> float:
    if healthy_val == 0.0:
        return float("nan") if patients_val != 0 else 0.0
    return 100.0 * (patients_val - healthy_val) / healthy_val


def compare_healthy_vs_patients(
    *,
    metrics_healthy: dict[str, Any],
    metrics_patients: dict[str, Any],
    class_names: list[str],
    out_path: Path,
) -> dict[str, Any]:
    """Step 5: relative metric changes and classes that degraded most (by recall F1)."""
    out: dict[str, Any] = {
        "healthy": metrics_healthy,
        "patients": metrics_patients,
        "relative_change_pct": {
            "accuracy": relative_change_pct(metrics_patients["accuracy"], metrics_healthy["accuracy"]),
            "macro_f1": relative_change_pct(metrics_patients["macro_f1"], metrics_healthy["macro_f1"]),
        },
        "per_class_relative_change": [],
        "worst_degraded_classes": [],
    }

    # by recall (sensitivity per class) and F1
    deg_scores: list[tuple[str, float, float]] = []
    for ph, pp in zip(metrics_healthy["per_class"], metrics_patients["per_class"]):
        assert ph["class"] == pp["class"]
        name = ph["class"]
        r_h, r_p = ph["recall"], pp["recall"]
        f_h, f_p = ph["f1"], pp["f1"]
        d_r = relative_change_pct(r_p, r_h)
        d_f = relative_change_pct(f_p, f_h)
        out["per_class_relative_change"].append(
            {
                "class": name,
                "precision_delta_pct": relative_change_pct(pp["precision"], ph["precision"]),
                "recall_delta_pct": d_r,
                "f1_delta_pct": d_f,
            }
        )
        # "сильнее всего упало" — по отрицательному изменению F1 (или recall)
        deg_scores.append((name, float(d_f) if d_f == d_f else -1e9, float(d_r) if d_r == d_r else -1e9))

    deg_scores.sort(key=lambda t: t[1])  # most negative F1 change first
    out["worst_degraded_classes"] = [
        {"class": t[0], "f1_delta_pct": t[1], "recall_delta_pct": t[2]} for t in deg_scores[:4]
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    return compute_metrics_dict(y_true, y_pred, class_names)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
