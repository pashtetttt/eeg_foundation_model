# Methods

The following sections describe the **data**, **signal handling**, **feature construction**, **predictive models**, **study design** (including analyses of generalization and distributional shift), and **evaluation criteria** used in this work. The focus is on methodological choices and their rationale, not on software usage.

---

## 1. Data

### 1.1 Recording format and cohorts

EEG was stored in **EDF** format. Two cohorts were distinguished analytically: a **main (development) cohort** used to fit and cross-validate age-group classifiers, and a **separate patient cohort** reserved for analyses of **out-of-distribution** performance and **domain shift** (comparison of feature distributions between cohorts). Class labels for supervised learning came from **recording-level group membership** (four age bands), not from fine-grained ages encoded in nested folder names.

### 1.2 Age classes (multiclass task)

Four mutually exclusive classes were used, corresponding to broad developmental stages (preschool, early school age, adolescence in a narrower sense, and late adolescence / “youth”). The exact mapping from archive structure to class identifiers is fixed for the whole study so that every recording in a given top-level group shares one label.

### 1.3 Resting-state condition: eyes open vs eyes closed

Open- and closed-eyes segments were treated as **separate modelling problems**: each recording was assigned to one condition using **conventions in the file name** (distinct markers for closed vs open). This design **avoids leakage** where the same participant could otherwise appear in both training and test sets under different vigilance and occipital α conditions, which would inflate apparent performance.

### 1.4 Class balance and sample sizes

The usable corpus was **moderately imbalanced**: one age band (the largest) contained on the order of **three dozen percent** of closed-eyes recordings, while the smallest band was near **one sixth** of the data. Exact counts depended on how many files passed quality checks after loading, but the imbalance was **preserved** in the primary protocol (see Section 5) rather than equalizing classes by subsampling, so that metrics reflect realistic prevalence.

### 1.5 Binary formulation (secondary task)

In addition to four-way classification, a **binary** task was defined by grouping three classes into a “non-target” set and treating the remaining band (late adolescence / youth) as the positive class. This supports questions where separation of that band from all others is of primary interest.

### 1.6 Data integrity at load time

Some EDF headers in real-world archives contain invalid numeric fields; readers may fail even when the signal block is intact. The ingestion path therefore allowed **header repair** when necessary so that the same decoding strategy could be applied across files. Recordings whose time series contained **non-finite values** after load were excluded from feature computation.

---

## 2. Preprocessing and channel handling

### 2.1 Minimal path used for end-to-end classification

The main experiments followed a **lightweight pipeline**: continuous EEG was read into memory with **MNE**-style objects, and **no global bandpass, notch, or ICA** was required before feature extraction. Instead, **band-specific operations** (filtering, PSD, bandpassed complexity measures) were applied **inside** each feature definition. This keeps a single transparent path from raw archive to feature vector while still respecting frequency content relevant to each descriptor.

### 2.2 Optional offline cleaning (alternative workflows)

For analyses that required **denoised** continuous data saved to disk, an optional pipeline applied a **1–40 Hz bandpass**, **50 Hz notch** (line noise), and **independent component analysis** with exclusion of components associated with typical cardiac and ocular artifacts when such structure was identifiable. If ICA was ill-conditioned, filtered data were retained without component rejection. That path was conceptually separate from the minimal feature-extraction route above.

### 2.3 Canonical montage and missing channels

Features were computed in a **fixed 19-channel layout** aligned with a common clinical montage. **Channel order** was established once from an initial recording and held fixed for all files so that each dimension of the feature vector had a consistent neuroanalytic meaning. Recordings with fewer than 19 channels were **padded with zeros** in missing slots so that vector length stayed constant across the dataset.

---

## 3. Feature extraction

Features were **per recording** (one vector per file). They combined **classical spectral summaries**, **time–frequency and envelope descriptors**, **nonlinear / complexity measures**, **regional aggregates** that do not decompose channel-by-channel, and an optional **literature-motivated** block. Optional dependencies for entropy and fractal measures were handled so that vector dimensionality stayed fixed if a library was unavailable (missing components set to zero rather than dropping columns).

### 3.1 Conventional spectral and regional summaries

- **Band powers:** mean power in δ (1–4 Hz), θ (4–8 Hz), α (8–13 Hz), and β (13–30 Hz) via **Welch-type PSD**, per channel (19 × 4 scalars).
- **Band-power ratios** per channel: ratios between θ, α, and β, plus a “slow vs fast” index combining δ+θ vs α+β, capturing relative prominence of slow vs fast oscillations.
- **Spectral centroid** per channel over a broad band (e.g. 0.5–45 Hz), summarizing where mass of the spectrum lies.
- **Envelope-related features:** dominant frequency of the **analytic envelope** per frequency band and channel, targeting low-rate modulation of band-limited activity.
- **Regional θ topography:** a small set of **scalar ratios** comparing θ sub-bands aggregated over **frontal, central, and parieto-occipital** regions (not 19 separate regional channels).

For **closed eyes only**, an additional **α-rhythm topography** block compared **true α (8–13 Hz)** and a **wider “predecessor” band (4–12 Hz)** via regional powers and pairwise regional ratios (frontal, central, parieto-occipital). This block was omitted for open eyes so that feature dimension matched the physiology of the condition.

### 3.2 Complexity, irregularity, and temporal dynamics

- **Sample entropy and approximate entropy** per channel on a time-domain segment, quantifying unpredictability of the signal at short lags.
- **Higuchi fractal dimension** per channel, summarizing geometric complexity of the time series.
- **Hjorth complexity** on an α-relevant bandpassed signal (e.g. 4–13 Hz), sensitive to the shape of the power spectrum relative to a simple oscillation.
- **α variability:** variability of **instantaneous α frequency** (mean and standard deviation per channel) within an α band, reflecting stability vs wandering of the dominant rhythm.

Together, these go beyond linear spectral power toward **nonlinear and nonstationary** aspects of the EEG that are often discussed in developmental and clinical contexts.

### 3.3 Optional literature-based block

An extended feature set appended **narrowband PSD sampling** (e.g. **1 Hz-wide bins** from 1–24 Hz across channels), **multiscale entropy** summarized over short, medium, and long temporal scales **by scalp region**, **log–log PSD slope and goodness-of-fit** in a low-to-mid frequency range **by region**, and a **surrogate-based nonlinearity index** comparing multiscale entropy on the signal to entropy on **phase-randomized surrogates** (iterations chosen to balance stability and cost). This block was inspired by published EEG complexity and scaling literature and was used to test whether **explicit complexity and 1/f-style** descriptors add information beyond the hand-crafted spectral stack.

### 3.4 Feature subsets for ablation

Experiments could use the **full vector** for the given eye condition, the full vector **plus** the literature block, **α-related** dimensions only, **everything except α-related** dimensions, or a **data-driven subset** (e.g. features ranked by univariate association with the label). The latter supports questions about redundancy and overfitting when dimensionality is high.

---

## 4. Predictive models

### 4.1 Primary learners: ensembles and gradient boosting

**Random forests** were used with **class-weighting** strategies so that minority classes were not ignored during tree splits. **Balanced random forests** (undersampling inside each bootstrap replicate) offered an alternative imbalance treatment complementary to reweighting.

**Gradient-boosted decision trees (XGBoost)** were the main high-capacity learners. **Multiclass** runs optimized multinomial log-loss; **binary** runs used logistic loss for the adolescence-vs-rest task. In all cases, features were **standardized** (zero mean, unit variance) **within the training fold** so that scale differences between power, entropy, and slope features did not dominate the fit.

**Class imbalance** on the training objective was addressed by **balanced per-sample weights** in boosting fits, so that each class contributed equally in expectation to the loss despite unequal counts.

Several **hyperparameter configurations** (tree depth, learning rate, subsampling of rows and columns, L1/L2-style regularization) were evaluated under the same data protocol to assess sensitivity of conclusions to model capacity and stochastic structure.

### 4.2 Baselines and auxiliary models

**Majority-class** and **stratified-random** baselines established that scores above chance were not artifacts of a fixed class distribution. A **kernel SVM** on heavily preprocessed saved data was explored in earlier workflows as a linear/nonlinear alternative in feature space defined by offline cleaning.

**Correlation filtering:** before fitting, highly redundant features could be pruned by detecting **large absolute Pearson correlations** on the training set and **dropping one member** of each pair (typically the lower-variance variable), reducing collinearity without using test labels.

### 4.3 Interpretation

**Feature importance** from tree models and **SHAP-style attributions** were used to summarize which constructed descriptors drove decisions, optionally **stratified by predicted class** to describe decision boundaries in more detail.

---

## 5. Experimental design and domain-shift analysis

### 5.1 Within-cohort supervised protocol

The primary evaluation used **stratified splitting** so that class proportions matched in train and evaluation subsets. A **single hold-out split** (e.g. 80% / 20%) gave unbiased point estimates on held-out data, while **k-fold cross-validation** (typically five folds) produced **out-of-fold predictions** on the full cohort for model comparison and for analyses that require a prediction for every recording.

Training did **not** artificially balance classes by throwing away data in the main protocol; instead, **sample weighting** (and balanced forest internals) encoded equal importance of classes in the loss.

### 5.2 Subject-aware validation when identifiers exist

Where a **subject identifier** could be attached to files, **grouped cross-validation** ensured that all segments from one participant stayed in either training or validation, mitigating **pseudo-replication** from multiple files per person.

### 5.3 Generalization to a patient cohort and domain shift

A separate workflow compared **in-cohort** performance (e.g. out-of-fold on the development cohort) to **performance on the patient cohort**, using the same fitted model or the same modelling family. This quantifies **performance drop** under **covariate and label-scheme shift**.

To interpret *why* distributions differ, features were **clustered** into interpretable groups (e.g. regional band power, regional complexity). Within each cluster, values were **aggregated** (e.g. median across member features) to obtain a stable scalar per recording. **Two-sample Kolmogorov–Smirnov tests** compared the **healthy training distribution** to the **patient distribution** on these aggregates; **small p-values** indicated **distributional shift** on that construct. **Intersection** of “model error” subsets with “shifted cluster” subsets highlighted constructs that were both **unstable across cohorts** and **associated with mistakes**. **Bootstrap resampling** assessed **stability** of rank-based statements about cluster differences under sampling noise.

**Correlation filtering** fit only on healthy training data was also evaluated as a way to reduce **redundant** sensitivity before patient testing.

### 5.4 Complementary scaling analysis (DFA)

**Detrended fluctuation analysis (DFA)** was applied in parallel on regional summaries to study **long-range temporal correlations** and age or cohort effects with **multiple-comparison control** (false discovery rate). This line is **orthogonal** to the main feature-based classifiers but addresses similar scientific questions about maturation and pathology at the level of **temporal scaling** rather than hand-engineered PSD features.

---

## 6. Statistical analysis and evaluation metrics

### 6.1 Classification performance

Standard **multiclass** reports included **accuracy**; **balanced accuracy** (average of per-class recall, useful under imbalance); **macro-averaged F1** (equal weight per class); **per-class precision, recall, and F1**; and **confusion matrices** for error structure. For the **binary** adolescence task, **F1 for the positive class** was reported alongside macro measures.

### 6.2 Comparing cohorts on scalar metrics

When contrasting healthy vs patient predictions, **relative change** in global metrics (e.g. accuracy, macro-F1) and **per-class** degradation highlighted which age bands were most affected by domain shift.

### 6.3 Tests on feature-level and cluster-level comparisons

For comparing two groups on continuous features (e.g. correct vs incorrect subsets, or healthy vs patient on cluster aggregates), **Mann–Whitney U** tests provided a nonparametric location shift; **Cohen’s d** summarized standardized effect size. **Benjamini–Hochberg adjustment** controlled **false discovery rate** across many simultaneous tests. **Kolmogorov–Smirnov** tests assessed **full distribution shape** differences for domain-shift screening. **Bootstrap** over subjects or recordings quantified **stability** of conclusions when sample sizes were moderate.

### 6.4 Role of naive baselines

Comparisons against **dummy** (no signal) and **random** stratified baselines supported the claim that ensemble scores reflected **learned structure** in the feature space rather than only **prior class frequencies**.

---

Together, these methods define a pipeline from raw resting EEG archives to **interpretable multivariate features**, **regularized nonlinear classifiers**, and **explicit tests of robustness** when models trained on one population are applied to another.
