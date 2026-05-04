#!/bin/bash
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/feature_pipeline_%j.log
#SBATCH --error=logs/feature_pipeline_%j.err

set -euo pipefail

# Project path on Zhores-2 (edit if your clone path differs)
PROJECT_DIR=/trinity/home/pavel.borisenko/lab/gpfs/neuro_ml/eeg_predict/eeg_foundation_model
GPFS_BIND=/trinity/home/pavel.borisenko/lab/gpfs
CONTAINER=docker://nvcr.io/nvidia/pytorch:24.04-py3
DATA_DIR_HOST=/trinity/home/pavel.borisenko/lab/gpfs/neuro_ml/eeg_predict/eeg_foundation_model/data
DATA_DIR_CONT=/gpfs/neuro_ml/eeg_predict/eeg_foundation_model/data

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

if [ ! -d "${DATA_DIR_HOST}" ]; then
  echo "ERROR: data directory not found on host: ${DATA_DIR_HOST}"
  exit 1
fi

if [ ! -d "${PROJECT_DIR}/eegpt/EEGPT/downstream" ]; then
  echo "ERROR: eegpt/EEGPT submodule missing. Run: git submodule update --init --recursive"
  exit 1
fi

# Config paths (edit cohort_name / data_dir inside these YAMLs on the cluster before sbatch)
CFG_FEATURES=configs/feature_extraction.yaml
CFG_EMBED=configs/zhores_pipeline_embedding.yaml
CFG_MERGE=configs/merge_features.yaml
CFG_XGB=configs/xgboost_training.yaml

singularity exec --nv \
  --pwd /gpfs/neuro_ml/eeg_predict/eeg_foundation_model \
  -B "${GPFS_BIND}:/gpfs" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    export PYTHONNOUSERSITE=1

    echo 'Container cwd:' \$(pwd)
    echo 'data_dir (host check was): ${DATA_DIR_HOST}'
    ls -la ${DATA_DIR_CONT} | head -n 15 || true
    python - <<'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
PY

    export TMPDIR=\$(pwd)/.pip_tmp
    DEPS_DIR=\$(pwd)/.slurm_job_deps
    rm -rf \"\$TMPDIR\" \"\$DEPS_DIR\"
    mkdir -p \"\$TMPDIR\" \"\$DEPS_DIR\"
    python -m pip install --target \"\$DEPS_DIR\" --no-deps geoopt
    python -m pip install --target \"\$DEPS_DIR\" --no-deps mne
    python -m pip install --target \"\$DEPS_DIR\" --no-deps imbalanced-learn==0.11.0
    python -m pip install --target \"\$DEPS_DIR\" --no-deps joblib
    export PYTHONPATH=\"\$DEPS_DIR:\${PYTHONPATH:-}\"

    OUT=\$(pwd)/results/pipeline_slurm_\${SLURM_JOB_ID:-local}
    mkdir -p \"\$OUT\"
    echo \"Pipeline log dir: \$OUT\"
    echo \"Started: \$(date -Is)\" | tee \"\$OUT/pipeline.log\"

    echo '=== 1/4 compute_all_features ===' | tee -a \"\$OUT/pipeline.log\"
    python scripts/features/compute_all_features.py --config ${CFG_FEATURES} | tee -a \"\$OUT/pipeline.log\"

    echo '=== 2/4 extract_embeddings (EEGPT; see configs/zhores_pipeline_embedding.yaml) ===' | tee -a \"\$OUT/pipeline.log\"
    python scripts/embeddings/extract_embeddings.py --config ${CFG_EMBED} | tee -a \"\$OUT/pipeline.log\"

    echo '=== 3/4 merge_features ===' | tee -a \"\$OUT/pipeline.log\"
    python scripts/features/merge_features.py --config ${CFG_MERGE} | tee -a \"\$OUT/pipeline.log\"

    echo '=== 4/4 train_xgboost ===' | tee -a \"\$OUT/pipeline.log\"
    python scripts/classification/train_xgboost.py --config ${CFG_XGB} | tee -a \"\$OUT/pipeline.log\"

    echo \"Finished: \$(date -Is)\" | tee -a \"\$OUT/pipeline.log\"
  "
