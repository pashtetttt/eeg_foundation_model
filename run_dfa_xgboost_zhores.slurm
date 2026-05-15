#!/bin/bash
#SBATCH --job-name=dfa_xgb
#SBATCH --partition=ais-gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/dfa_xgb_%j.log
#SBATCH --error=logs/dfa_xgb_%j.err

set -euo pipefail

PROJECT_DIR=/trinity/home/pavel.borisenko/lab/gpfs/neuro_ml/eeg_predict/eeg_foundation_model
GPFS_BIND=/trinity/home/pavel.borisenko/lab/gpfs
CONTAINER=docker://nvcr.io/nvidia/pytorch:24.04-py3

CFG_HEALTHY=configs/feature_extraction_zhores_healthy.yaml
CFG_PATIENTS=configs/feature_extraction_zhores_patients.yaml
CFG_XGB=configs/xgboost_dfa_only.yaml

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

singularity exec \
  --pwd /gpfs/neuro_ml/eeg_predict/eeg_foundation_model \
  -B "${GPFS_BIND}:/gpfs" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    export PYTHONNOUSERSITE=1

    export TMPDIR=\$(pwd)/.pip_tmp
    DEPS_DIR=\$(pwd)/.slurm_job_deps
    rm -rf \"\$TMPDIR\" \"\$DEPS_DIR\"
    mkdir -p \"\$TMPDIR\" \"\$DEPS_DIR\"
    python -m pip install --target \"\$DEPS_DIR\" --no-deps mne
    python -m pip install --target \"\$DEPS_DIR\" --no-deps PyYAML
    python -m pip install --target \"\$DEPS_DIR\" --no-deps joblib
    export PYTHONPATH=\"\$DEPS_DIR:\$(pwd)/src:\${PYTHONPATH:-}\"

    LOG=\"\$(pwd)/logs/dfa_xgb_\${SLURM_JOB_ID:-local}_console.txt\"
    mkdir -p \"\$(dirname \"\$LOG\")\"

    run_cohort() {
      local FEAT_CFG=\"\$1\"
      local COHORT=\"\$2\"
      echo \"========== cohort=\$COHORT ==========\" | tee -a \"\$LOG\"
      python scripts/features/compute_all_features.py --config \"\$FEAT_CFG\" 2>&1 | tee -a \"\$LOG\"
      python scripts/features/prepare_dfa_cache.py --config \"\$FEAT_CFG\" 2>&1 | tee -a \"\$LOG\"
      python scripts/classification/train_xgboost.py \\
        --config ${CFG_XGB} \\
        --cohort-name \"\$COHORT\" \\
        --condition closed \\
        --experiment dfa_only \\
        2>&1 | tee -a \"\$LOG\"
    }

    run_cohort ${CFG_HEALTHY} healthy
    run_cohort ${CFG_PATIENTS} patients

    echo 'Done. Results under results/classification/xgb_dfa_only/ and cached features under results/features/'
  "

echo "Job finished. See logs/dfa_xgb_${SLURM_JOB_ID}.log"
