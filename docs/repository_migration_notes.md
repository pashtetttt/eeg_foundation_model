# Repository Migration Notes

This repository now includes a GitHub-friendly scaffold:

- package root: `src/eeg_thesis/`
- workflow wrappers: `scripts/run_*.py`
- config templates: `configs/**/default.yml`
- curated documentation: `docs/` and `reports/`
- reproducibility helpers: `Makefile`, `pyproject.toml`, `.gitignore`

## Legacy compatibility

Existing top-level experiment scripts are intentionally kept unchanged for now.
The new wrappers call those scripts to avoid breaking current pipelines.

## Recommended next incremental step

Gradually move core logic from legacy scripts into `src/eeg_thesis/` modules and keep script files as thin CLI adapters.
