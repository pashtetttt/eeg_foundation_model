# Report: feature importance logs, feature lists, and visualization outputs

This document summarizes **non-experiment** artifacts in `results/` related to feature ranking, selection, and figures produced by `visualize_features.py`.

---

## 1. Feature importance text files (`feature_importance_*.txt`)

These list **Random Forest Gini importances** (trained on a **class-balanced** training subset, consistent with `train_rf.py` / `feature_importance.py` style runs).

| File | # features | Top-ranked themes (see file for full table) |
|------|--------------|---------------------------------------------|
| `feature_importance_20260312_224707.txt` | 209 | Strong **delta** band powers, **ratio_slow_fast** and **ratio_theta_*** across channels, **spectral centroids** (e.g. `centroid_ch14`). |
| `feature_importance_20260313_120226.txt` | 364 (open-eyes layout) | **ratio_slow_fast**, **hjorth_complexity**, **alpha_var_mean**, **band_delta** — similar to 122619 with small ordering differences. |
| `feature_importance_20260313_122619.txt` | 364 | #1 **`theta_ratio_paroccipital`** (theta1/theta2 regional ratio), then **ratio_slow_fast**, **band_delta**, **alpha_var_mean**, **hjorth_complexity**, **ratio_theta_alpha** / **ratio_theta_beta**. |
| `feature_importance_20260320_111845.txt` | 12 | **Closed-eyes alpha topography** block only: `pred_ratio_*`, `alpha_power_*`, `alpha_ratio_*` dominate — posterior/central alpha structure. Includes a small **SelectKBest-style** table: best **K=10**, weighted F1 **0.3673** (from that script’s evaluation). |

**Interpretation (thesis angle):** across the full 364-feature runs, importance is spread across **spectral bands (especially delta)**, **band ratios (slow/fast, theta/alpha, theta/beta)**, **alpha variability**, **Hjorth complexity**, and **regional theta ratio** (`theta_ratio_paroccipital`). The **12-feature** closed-eyes run isolates **alpha-rhythm topography** as highly informative when that block is used alone.

---

## 2. Selected-feature lists (`selected_features_*.txt`)

These are **names only** (one per line), aligned with `eeg_features.get_feature_names(...)`, typically produced when selecting top-K by mutual information or by importance:

| File | Role (from naming) |
|------|---------------------|
| `selected_features_top209_20260312_224707.txt` | Large set (209 names) aligned with the 209-feature importance run. |
| `selected_features_top120_20260313_120226.txt`, `selected_features_top120_20260313_122619.txt` | Top 120 features (two runs). |
| `selected_features_top10_20260320_111845.txt` | Top 10 from alpha-topography / small feature set experiment. |
| `selected_features_k50_mutual_info_20260319_205831.txt` | k=50 features by mutual information. |
| `selected_features_k150_mutual_info_20260319_205929.txt` | k=150 features by mutual information (used later in grid experiments as `--features selected`). |

Use these files to **reproduce** which columns were used in any pipeline that reads them by name.

---

## 3. Visualization outputs (`results/visualizations/`)

All files below share timestamp **`20260319_194345`** and **closed-eyes** data (`*_closed_*` in filenames), from a single `visualize_features.py` run.

| File | Type | Purpose |
|------|------|--------|
| `correlation_matrix_closed_20260319_194345.png` | Heatmap | Pairwise **feature correlation** (multicollinearity / redundancy). |
| `correlation_clustermap_closed_20260319_194345.png` | Heatmap + clustering | Same matrix with **hierarchical clustering** to group similar features. |
| `correlation_matrix_closed_20260319_194345.npy` | NumPy array | Saved numeric correlation matrix for reuse. |
| `mutual_information_closed_20260319_194345.png` | Bar / plot | **Mutual information** between each feature and class label (univariate relevance). |
| `mi_scores_closed_20260319_194345.npy` | NumPy array | Raw MI scores. |
| `feature_clustering_closed_20260319_194345.png` | Dendrogram | **Hierarchical clustering** of features (distance-based). |
| `shap_summary_closed_20260319_194345.png` | SHAP summary | **SHAP** global feature impact for the fitted RF (direction and magnitude). |
| `shap_values_closed_20260319_194345.png` | SHAP plot | Additional SHAP visualization (e.g. detailed view). |

**Note:** Figures are **not** embedded here; open the PNGs in `results/visualizations/` (or regenerate with `python visualize_features.py --eyes closed --all`).

---

## 4. How this ties to the best single RF run

The best **single-run** CV macro F1 in `results_rf_*.txt` (see `report_metrics_single_runs_and_baselines.md`) is **0.48** (`results_rf_20260313_121519.txt`, 364 features). The importance and MI/SHAP artifacts above support interpreting **which** signal types drive age-group separation (ratios, delta, alpha variability, topography, etc.) — **not** to replace the numerical scores in the experiment logs, which are the authoritative evaluation.

---

*Generated as a static report; update when new `feature_importance_*`, `selected_features_*`, or `visualizations/*` are added.*
