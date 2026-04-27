# EEG Thesis Repository

This repository contains EEG feature engineering, classification experiments, domain-shift/error analysis, and DFA-based statistical analysis used for thesis work.

## Repository Layout

- `src/eeg_thesis/`: reusable package entrypoints and wrappers.
- `scripts/`: top-level runnable commands for main workflows.
- `configs/`: example YAML configs for reproducible runs.
- `reports/`: curated methods/results markdown for GitHub reading.
- `results/reports/`: generated long-form reports kept in repo.
- `tests/`: test scaffolding.

Legacy experiment scripts are still present in the repository root for backward compatibility.

## Quickstart

1. Create environment and install deps:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Run one workflow:
   - `make train`
   - `make feature-selection`
   - `make error-analysis`
   - `make dfa`

If cloning from GitHub, include submodules:

- `git clone --recurse-submodules <repo-url>`
- or after clone: `git submodule update --init --recursive`

## HEEGNet Training

- Install runtime deps: `python3 -m pip install -r requirements-heegnet.txt`
- Config run: `python3 scripts/train_heegnet.py --config configs/heegnet_age4.yaml`
- Smoke test override: `python3 scripts/train_heegnet.py --config configs/heegnet_age4.yaml --limit-total 64 --epochs 3 --device cpu`
- Full-run guide: `docs/heegnet_remote_training.md`

## Data

Raw EEG data is not versioned in GitHub. See `docs/data.md` for expected local directory structure and naming assumptions.

## Reproducibility

- Use configs under `configs/`.
- Prefer top-level script wrappers in `scripts/`.
- Save run outputs into `results/` (ignored by default except curated report files).
