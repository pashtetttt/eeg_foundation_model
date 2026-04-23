# Results

This chapter consolidates outcomes across all result streams: multiclass and binary classification, feature-set ablations, feature selection, feature importance and SHAP outputs, cluster-level analyses, error analysis under cohort shift, and DFA statistical findings.

---

## 1. Main multiclass classification outcomes

### 1.1 Overall ranking of model families (5-fold CV, macro-F1)

Across the experiment grids, performance clustered in a narrow but consistent band. The strongest cross-validated macro-F1 values were:

- **XGBoost:** up to **0.4937** (closed-eyes, full feature set + complexity extension).
- **Balanced Random Forest:** up to **0.4876** (closed-eyes, full feature set).
- **Random Forest:** up to **0.4867** (open-eyes, full feature set).

This indicates that boosting had a small but reproducible edge, while all three families produced broadly similar class-separation quality under the same data protocol.

### 1.2 Hold-out behavior and class pattern

Best hold-out macro-F1 values were slightly higher than CV in some runs (typical sampling variability), with peaks around **0.50–0.53** for multiclass settings. A stable confusion pattern appears across RF/BRF/XGB:

- **Adolescence** has high recall and tends to absorb misclassifications from other classes.
- **Primary** and especially **teenager** are harder to separate from adolescence.
- **Preschooler** varies by condition and feature mode, but is also affected by the adolescence attractor effect.

In practical terms, the models are learning a strong gradient toward “older-like” EEG profiles, and class boundaries among middle groups remain the main bottleneck.

### 1.3 Effect of feature-mode ablations

Across RF/BRF/XGB, the same ordering repeats:

- **`all`** (full engineered set) is generally strongest and most stable.
- **`selected`** (preselected ~150 features) is close but usually slightly lower.
- **`alpha`-only** degrades most in multiclass performance.

This supports that age-group discrimination is not purely alpha-driven; it relies on mixed contributions from spectral ratios, centroids, envelope dynamics, and complexity descriptors.

### 1.4 Added complexity block (`all_plus_complexity`)

When the literature-inspired complexity block is appended (856 total features in closed eyes), the best CV macro-F1 reaches **0.4937**, i.e., the top observed multiclass CV score. The gain over strong baseline XGB runs is modest rather than dramatic, suggesting:

- the baseline handcrafted set already captures substantial signal,
- the extra complexity features provide incremental improvement, and
- complexity terms likely help specific class boundaries rather than changing the full error structure.

---

## 2. Binary task (adolescence vs rest)

The binary formulation clearly outperforms multiclass in aggregate metrics:

- **Hold-out:** macro-F1 **0.6378**, balanced accuracy **0.7073**, adolescence F1 **0.6624**.
- **5-fold CV:** macro-F1 **0.6271**, balanced accuracy **0.6969**, adolescence F1 **0.6557**.

Interpretation:

- The target class is highly recoverable in a one-vs-rest setting.
- Recall for adolescence is very high (about 0.95-0.96), but precision is moderate (~0.50), meaning many non-adolescence cases are pulled into the positive class.
- This is consistent with multiclass confusion behavior: adolescence acts as a dominant attractor when boundaries are uncertain.

---

## 3. Baselines and sanity checks

From baseline reports:

- Historical single-run RF best CV macro-F1 is **0.48** (same scale as experiment-grid RF).
- Dummy and random baselines remain far below tuned ensembles (macro-F1 roughly in the low **0.1-0.2** range for many baseline runs).

So the ensemble models are learning nontrivial structure beyond class priors or random allocation, even though absolute multiclass performance remains moderate.

---

## 4. Feature selection and dimensionality reduction results

Feature-selection experiments with grouped CV show an important trade-off between compression and accuracy.

### 4.1 Closed-eyes feature selection

- **Baseline all (376 features):** macro-F1 **0.4654 ± 0.0307**.
- **Cluster-aggregated median (77 features):** macro-F1 **0.4693 ± 0.0480**.
- **Cluster representative (77 features):** macro-F1 **0.4616 ± 0.0557**.
- **Top-100 by importance (100 features):** macro-F1 **0.4607 ± 0.0306**.

Key point: reducing from 376 to ~77 features can preserve performance within variance, especially for cluster aggregation.

### 4.2 Open-eyes feature selection

- **Baseline all (364 features):** macro-F1 **0.4884 ± 0.0544**.
- **Cluster-aggregated median (71 features):** macro-F1 **0.4657 ± 0.0427**.
- **Cluster representative (71 features):** macro-F1 **0.4453 ± 0.0320**.
- **Top-100 by importance (100 features):** macro-F1 **0.4820 ± 0.0443**.

Open-eyes results suggest that top-100 retains most of the baseline signal, while aggressive cluster representation can over-compress discriminative information.

### 4.3 Practical conclusion from selection studies

- For interpretability and runtime, **top-100** or **cluster-aggregated** variants are reasonable.
- For peak multiclass performance, full features still provide a small advantage.
- Cluster-based compression is especially useful when one wants stable regional constructs instead of single-channel coefficients.

---

## 5. Feature importance, SHAP, and feature clusters

Feature-importance and visualization outputs converge on a consistent signal hierarchy:

1. **Band ratios** (especially slow/fast, theta-linked ratios),
2. **Delta-band and broad spectral power structure**,
3. **Spectral centroid patterns**,
4. **Alpha variability descriptors**,
5. **Hjorth complexity**,
6. **Regional topographic ratios (including parieto-occipital theta/alpha structures)**.

The SHAP and clustering outputs support the same interpretation: predictive signal is distributed across multiple physiological families, not concentrated in one metric type.

Closed-eyes alpha-topography-only analyses show that this block is informative, but not sufficient to replace the broader multi-family feature stack for multiclass discrimination.

---

## 6. Correlation-filter and eyes-difference auxiliary analyses

### 6.1 Correlation filter (open-eyes all-features setting)

A strict redundancy filter reduced dimensionality strongly (e.g., **364 → 138** features at |r| > 0.95). Out-of-fold performance after filtering remained around macro-F1 **0.4545** in the inspected run.

Interpretation: many engineered features are highly collinear, and pruning can simplify the feature space substantially, but aggressive filtering may also remove weakly unique information useful for class boundaries.

### 6.2 Eyes-difference feature experiments

In eyes-difference hold-out runs, macro-F1 was lower (best observed around **0.4071**). This indicates that the derived difference representation, while conceptually appealing, did not outperform direct condition-specific feature modeling in current settings.

---

## 7. Error analysis under healthy-to-patient shift

The most important findings come from comparing in-cohort predictions (healthy) to patient-cohort testing.

### 7.1 Metric collapse in cross-cohort transfer

For both open and closed conditions, and both with/without correlation filtering:

- **Healthy macro-F1** is around **0.45-0.49**,
- **Patient macro-F1** drops to roughly **0.18-0.21**,
- Relative macro-F1 loss is severe (about **-55% to -62%**).

Accuracy drops are smaller than macro-F1 drops, showing that class imbalance can mask true degradation if only accuracy is considered.

### 7.2 Error structure on patient cohort

Misclassifications are strongly directional:

- `preschooler`, `primary`, and `teenager` are very frequently predicted as `adolescence`.
- In some settings, teenager→adolescence dominates nearly the entire row.

This is not random error; it is a systematic shift toward one class prototype.

### 7.3 Domain-shift statistics and clusters

Kolmogorov-Smirnov cluster screening reports many significant shifts:

- about **69 significant clusters** in closed-eye analyses,
- about **63 significant clusters** in open-eye analyses.

Most shifted clusters repeatedly involve:

- **Hjorth complexity** (frontal/central/temporal/parieto-occipital),
- **Spectral centroid** clusters,
- **Theta-ratio** topography,
- **Alpha variability** and envelope-frequency clusters.

These are exactly the families highlighted by feature-importance analyses, which strengthens the interpretation: the model is relying on biologically meaningful constructs that themselves move between cohorts.

### 7.4 Robustness of “stable vulnerable clusters”

The strict intersection criterion for clusters significant in all condition/experiment cells yielded **0 universally stable clusters**, indicating that vulnerable constructs depend on recording condition and preprocessing variant. In other words, shift sensitivity is broad but not perfectly invariant.

---

## 8. DFA results

DFA adds an independent, statistical view of temporal scaling differences.

### 8.1 Closed-eyes DFA

The closed-eyes DFA run reports **32 FDR-significant effects**. Main pattern:

- Strong age-stratified differences within the healthy cohort across frontal/central/parietal/occipital regions.
- Adolescence tends to differ in direction from younger groups in multiple regions.
- Cohort differences (healthy vs patients) are age-dependent: for adolescence, patients often show higher DFA alpha; for other ages, direction can reverse.

This suggests that long-range temporal organization is development-sensitive and interacts with cohort status in a nonuniform way.

### 8.2 Open-eyes DFA

The open-eyes DFA run shows **0 FDR-significant effects** in the sampled configuration. This aligns with a weaker separability signal for open-eye DFA in that specific run and emphasizes sensitivity to sample size/setting.

---

## 9. Integrated interpretation across all result streams

Across metrics, importance maps, feature-selection studies, error decomposition, cluster tests, and DFA:

- The project consistently detects age-related EEG structure, but multiclass separation remains moderate.
- The strongest recurrent failure mode is over-assignment to adolescence under uncertainty.
- Feature relevance and domain-shift relevance point to the same families (ratios, centroids, alpha variability, Hjorth/complexity), which is methodologically coherent.
- Generalization to the external patient cohort is the critical challenge: distribution shift is broad and materially harms macro-level discrimination.
- Dimensionality reduction can preserve much of the signal, but does not by itself solve cross-cohort shift.

In short, the results support a robust internal signal and a clear next-stage agenda: shift-aware modeling and calibration for external cohorts.
