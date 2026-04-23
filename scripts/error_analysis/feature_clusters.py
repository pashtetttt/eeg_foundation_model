from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ParsedFeature:
    name: str
    metric: str
    band: str | None
    channel_index: int | None
    channel_name: str | None
    region: str


FRONTAL_KW = ("Fp", "F3", "F4", "F7", "F8", "Fz")
CENTRAL_KW = ("C3", "C4", "Cz")
PAROCC_KW = ("P3", "P4", "Pz", "O1", "O2")


def _region_from_channel_name(ch: str | None) -> str:
    if not ch:
        return "unknown"
    if any(k in ch for k in FRONTAL_KW):
        return "frontal"
    if any(k in ch for k in CENTRAL_KW):
        return "central"
    if any(k in ch for k in PAROCC_KW):
        return "paroccipital"
    # keep it coarse but explicit
    if ch.startswith("T") or ch.startswith("FT") or ch.startswith("TP"):
        return "temporal"
    if ch.startswith("P"):
        return "parietal"
    if ch.startswith("O"):
        return "occipital"
    if ch.startswith("F"):
        return "frontal"
    if ch.startswith("C"):
        return "central"
    return "other"


def _parse_channel_index(name: str) -> int | None:
    # expected "..._ch0" .. "..._ch18"
    if "_ch" not in name:
        return None
    try:
        tail = name.rsplit("_ch", 1)[1]
        return int(tail)
    except Exception:
        return None


def parse_feature_name(name: str, canonical_ch_names: list[str] | None) -> ParsedFeature:
    """
    Parse a feature name into {metric, band/range, channel, region}.

    The project uses names from `eeg_features.get_feature_names`, e.g.:
    - band_alpha_ch3
    - ratio_theta_alpha_ch10
    - centroid_ch5
    - samp_ent_ch0 / app_ent_ch0
    - envfreq_beta_ch2
    - hfd_ch7
    - hjorth_complexity_ch12
    - alpha_var_mean_ch1 / alpha_var_std_ch1
    - theta_ratio_frontal
    - alpha_power_frontal / pred_ratio_central_paroccipital (closed-eyes topo)
    """
    band = None
    ch_idx = _parse_channel_index(name)
    ch_name = None
    if ch_idx is not None and canonical_ch_names and 0 <= ch_idx < len(canonical_ch_names):
        ch_name = canonical_ch_names[ch_idx] or None

    if name.startswith("band_"):
        # band_{band}_chN
        metric = "power"
        band = name.split("_", 2)[1]
        region = _region_from_channel_name(ch_name)
        return ParsedFeature(name, metric, band, ch_idx, ch_name, region)

    if name.startswith("ratio_"):
        # ratio_theta_alpha_chN
        metric = "ratio"
        band = name.split("_", 2)[1]  # theta_alpha etc.
        region = _region_from_channel_name(ch_name)
        return ParsedFeature(name, metric, band, ch_idx, ch_name, region)

    if name.startswith("envfreq_"):
        metric = "envelope_freq"
        band = name.split("_", 2)[1]
        region = _region_from_channel_name(ch_name)
        return ParsedFeature(name, metric, band, ch_idx, ch_name, region)

    if name.startswith("centroid_"):
        metric = "spectral_centroid"
        region = _region_from_channel_name(ch_name)
        return ParsedFeature(name, metric, None, ch_idx, ch_name, region)

    if name.startswith("samp_ent_") or name.startswith("app_ent_"):
        metric = "entropy"
        region = _region_from_channel_name(ch_name)
        return ParsedFeature(name, metric, None, ch_idx, ch_name, region)

    if name.startswith("hfd_"):
        metric = "higuchi_fd"
        region = _region_from_channel_name(ch_name)
        return ParsedFeature(name, metric, None, ch_idx, ch_name, region)

    if name.startswith("hjorth_complexity_"):
        metric = "hjorth_complexity"
        region = _region_from_channel_name(ch_name)
        return ParsedFeature(name, metric, None, ch_idx, ch_name, region)

    if name.startswith("alpha_var_mean_") or name.startswith("alpha_var_std_"):
        metric = "alpha_var"
        band = "mean" if "alpha_var_mean_" in name else "std"
        region = _region_from_channel_name(ch_name)
        return ParsedFeature(name, metric, band, ch_idx, ch_name, region)

    if name.startswith("theta_ratio_"):
        # theta_ratio_frontal / central / paroccipital
        metric = "theta_ratio"
        region = name.split("_", 2)[2]
        return ParsedFeature(name, metric, "4-8", None, None, region)

    if name.startswith("alpha_power_") or name.startswith("pred_power_") or name.startswith("alpha_ratio_") or name.startswith("pred_ratio_"):
        # closed-eyes topo already encodes region
        metric = "alpha_topo" if name.startswith("alpha_") else "pred_topo"
        # alpha_power_frontal, pred_ratio_frontal_central etc.
        parts = name.split("_")
        region = parts[-1] if parts[-1] in ("frontal", "central", "paroccipital") else parts[-2] if parts[-2] in ("frontal", "central", "paroccipital") else "mixed"
        band = "alpha" if name.startswith("alpha_") else "pred"
        return ParsedFeature(name, metric, band, None, None, region)

    return ParsedFeature(name, "other", None, ch_idx, ch_name, _region_from_channel_name(ch_name))


def cluster_key(p: ParsedFeature) -> str:
    """
    Cluster pattern: {metric}{band_or_range}{region}.

    Examples:
    - power_alpha_frontal
    - ratio_theta_alpha_paroccipital
    - hjorth_complexity_central
    """
    metric = p.metric
    band = p.band
    region = p.region
    if band:
        return f"{metric}_{band}_{region}"
    return f"{metric}_{region}"


def build_clusters(feature_names: list[str], canonical_ch_names: list[str] | None) -> dict[str, list[int]]:
    clusters: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        pf = parse_feature_name(name, canonical_ch_names)
        k = cluster_key(pf)
        clusters.setdefault(k, []).append(i)
    return clusters


def aggregate_cluster_matrix(X: np.ndarray, clusters: dict[str, list[int]], agg: str = "median") -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Aggregate original feature matrix X into cluster-level matrix.

    Returns:
    - Xc: shape (n_samples, n_clusters)
    - keys: cluster names aligned with columns
    - sizes: number of raw features inside each cluster
    """
    keys = sorted(clusters.keys())
    sizes = np.array([len(clusters[k]) for k in keys], dtype=int)
    Xc = np.zeros((X.shape[0], len(keys)), dtype=float)
    for j, k in enumerate(keys):
        cols = clusters[k]
        block = X[:, cols]
        if agg == "median":
            Xc[:, j] = np.nanmedian(block, axis=1)
        elif agg == "mean":
            Xc[:, j] = np.nanmean(block, axis=1)
        else:
            raise ValueError(f"Unknown agg='{agg}'")
    return Xc, keys, sizes


def clusters_table(clusters: dict[str, list[int]], feature_names: list[str], max_examples: int = 12) -> list[dict]:
    rows = []
    for k, idxs in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        ex = [feature_names[i] for i in idxs[:max_examples]]
        rows.append(
            {
                "cluster": k,
                "n_features": len(idxs),
                "examples": ", ".join(ex) + (" ..." if len(idxs) > max_examples else ""),
            }
        )
    return rows

