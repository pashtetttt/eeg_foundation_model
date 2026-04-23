from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .stats_utils import benjamini_hochberg, run_tests


def _split_fn(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> tuple[np.ndarray, np.ndarray]:
    mask = y_true == cls
    idx = np.where(mask)[0]
    correct = idx[y_pred[idx] == cls]
    error = idx[y_pred[idx] != cls]
    return correct, error


def _split_fp(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> tuple[np.ndarray, np.ndarray]:
    mask = y_pred == cls
    idx = np.where(mask)[0]
    correct = idx[y_true[idx] == cls]
    error = idx[y_true[idx] != cls]
    return correct, error


def bootstrap_cluster_stability(
    *,
    Xc: np.ndarray,
    cluster_keys: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: list[str] | np.ndarray,
    n_iter: int = 50,
    p_adj_thr: float = 0.01,
    d_thr: float = 0.5,
    random_state: int = 42,
    out_path: Path | None = None,
) -> dict[str, float]:
    """
    Step 9: bootstrap resampling by subjects; repeat Step 3 (FN/FP cluster tests);
    record frequency each cluster is significant (p_adj & d & any view/class).
    """
    subject_ids = np.asarray(subject_ids)
    unique_subj = np.unique(subject_ids)
    rng = np.random.default_rng(random_state)
    n_classes = int(y_true.max()) + 1 if y_true.size else 0
    n_clusters = Xc.shape[1]
    hits = np.zeros(n_clusters, dtype=int)

    for _ in range(n_iter):
        boot_subj = rng.choice(unique_subj, size=len(unique_subj), replace=True)
        idx_list: list[int] = []
        for s in boot_subj:
            idx_list.extend(np.where(subject_ids == s)[0].tolist())
        idx = np.asarray(idx_list, dtype=int)
        if idx.size == 0:
            continue
        yt = y_true[idx]
        yp = y_pred[idx]
        Xb = Xc[idx]

        sig_clusters: set[int] = set()
        for view in ("fn", "fp"):
            for cls in range(n_classes):
                if view == "fn":
                    ok, err = _split_fn(yt, yp, cls)
                else:
                    ok, err = _split_fp(yt, yp, cls)
                if ok.size < 5 or err.size < 5:
                    continue
                pvals = np.ones(n_clusters, dtype=float)
                ds = np.zeros(n_clusters, dtype=float)
                for j in range(n_clusters):
                    tr = run_tests(Xb[ok, j], Xb[err, j])
                    pvals[j] = 1.0 if tr.p_mwu is None else tr.p_mwu
                    ds[j] = tr.cohens_d
                padj = benjamini_hochberg(pvals)
                for j in range(n_clusters):
                    if padj[j] < p_adj_thr and abs(ds[j]) > d_thr:
                        sig_clusters.add(j)
        for j in sig_clusters:
            hits[j] += 1

    freq = hits / float(n_iter)
    out = {cluster_keys[j]: float(freq[j]) for j in range(n_clusters)}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"cluster": cluster_keys[j], "stability": float(freq[j]), "passes_0.70": bool(freq[j] >= 0.70)}
            for j in range(n_clusters)
        ]
        rows.sort(key=lambda r: -r["stability"])
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
