# Metrics report: single `train_rf` runs and baselines (no `*_experiments*` files)

Generated from text logs under `results/` whose names do **not** contain `_experiments`.

## 1. Best Random Forest run (by stratified 5-fold CV macro F1)

Two historical `results_rf_*.txt` runs tie for **highest CV macro F1 = 0.48**:

| File | N samples | Features | CV accuracy | CV macro F1 | CV weighted F1 | Notes |
|------|-----------|----------|-------------|-------------|----------------|--------|
| `results_rf_20260313_121519.txt` | 1373 | 364 (full open-eyes vector) | 0.55 | **0.48** | 0.51 | Default RF; open-eyes feature layout (no alpha-topography block). |
| `results_rf_20260312_165938.txt` | 683 | 76 (band powers only) | 0.55 | **0.48** | 0.51 | Older/smaller feature set (band power only). |

**Recommended “best” row for reporting:** `results_rf_20260313_121519.txt` — same peak macro F1, **larger dataset** (1373 vs 683) and **full** feature vector for that eyes setting.

### Best-run details (`results_rf_20260313_121519.txt`)

- **Model:** Random Forest, `n_estimators=200`, `max_depth=20`, `class_weight=balanced`, `random_state=42`
- **Experiment window:** 2026-03-13 12:15–12:20
- **Balanced hold-out (stratified 80/20 on balanced subset):** accuracy **0.46**, macro F1 **0.43**, weighted F1 **0.43** (support 193 in that split)
- **Stratified 5-fold CV (full imbalanced dataset):** accuracy **0.55**, macro precision / recall / F1 **0.65 / 0.47 / 0.48**, weighted F1 **0.51** (support 1373)

### All non-experiment `results_rf_*.txt` runs (CV macro F1, descending)

| File | CV acc | CV macro F1 | CV weighted F1 |
|------|--------|-------------|------------------|
| results_rf_20260313_121519.txt | 0.55 | 0.48 | 0.51 |
| results_rf_20260312_165938.txt | 0.55 | 0.48 | 0.51 |
| results_rf_20260312_192116.txt | 0.55 | 0.45 | 0.51 |
| results_rf_20260319_173307.txt | 0.54 | 0.45 | 0.51 |
| results_rf_20260319_174935.txt | 0.54 | 0.45 | 0.51 |
| results_rf_20260313_120314.txt | 0.54 | 0.46 | 0.50 |
| results_rf_20260320_111620.txt | 0.54 | 0.44 | 0.51 |
| results_rf_20260320_113011.txt | 0.41 | 0.40 | 0.41 |
| results_rf_20260324_164527.txt | 0.48 | 0.40 | 0.41 |
| results_rf_20260319_161404.txt | 0.52 | 0.43 | 0.49 |
| results_rf_20260319_162141.txt | 0.52 | 0.43 | 0.49 |
| results_rf_20260319_204224.txt | 0.51 | 0.43 | 0.48 |
| results_rf_selectkbest_20260319_205831.txt | 0.51 | 0.41 | 0.48 |
| results_rf_selectkbest_20260319_205929.txt | 0.51 | 0.40 | 0.48 |

*(SelectKBest runs use feature selection; see those files for K and eyes settings.)*

---

## 2. Dummy classifier (`train_dummy.py` — most frequent class)

| File | Eyes | Features | N | Hold-out acc | CV acc | CV macro F1 | CV weighted F1 |
|------|------|----------|---|--------------|--------|-------------|----------------|
| `results_dummy_20260324_171517.txt` | closed | all | 116 (quick `--max` run) | 0.22 | 0.24 | 0.19 | 0.19 |
| `results_dummy_20260324_171902.txt` | open | alpha | 792 | 0.25 | 0.38 | 0.14 | 0.21 |

Interpretation: the dummy ignores inputs; metrics depend heavily on **class frequencies** and **train/test split**. The 116-sample run is a small subsample; the 792-sample run is closer to full data but still a weak baseline.

---

## 3. Random baseline (`train_random_baseline.py` — stratified random)

| File | Eyes | N | CV acc | CV macro F1 | CV weighted F1 |
|------|------|---|--------|-------------|----------------|
| `results_random_baseline_20260319_201328.txt` | closed | 328 | 0.27 | 0.23 | 0.27 |
| `results_random_baseline_20260319_204749.txt` | open | 328 | 0.27 | 0.23 | 0.27 |

---

## 4. Takeaway

- The strongest **single** `train_rf` log in this folder (by CV macro F1) is **0.48**, from `results_rf_20260313_121519.txt` (and a tied older run on 76 features only).
- **Dummy** and **random** baselines stay well **below** that CV macro F1, which supports that the forest is learning signal beyond chance and majority-class guessing (under each file’s own protocol and sample size).

---

*This file is descriptive text only; regenerate or extend if you add new `results_rf_*.txt` / dummy / random logs.*
