#!/bin/bash
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/reve_pipeline_%j.log
#SBATCH --error=logs/reve_pipeline_%j.err

set -euo pipefail

# Project path on Zhores-2 (edit if needed)
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

if [ ! -d "${PROJECT_DIR}/reve/reve_eeg/src" ]; then
  echo "ERROR: REVE repo missing at reve/reve_eeg. Clone it first."
  exit 1
fi

CFG_FEATURES=configs/feature_extraction.yaml
CFG_REVE_EMB=configs/reve_embedding_extraction.yaml
CFG_REVE_XGB=configs/reve_xgboost.yaml

singularity exec --nv \
  --pwd /gpfs/neuro_ml/eeg_predict/eeg_foundation_model \
  -B "${GPFS_BIND}:/gpfs" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    export PYTHONNOUSERSITE=1

    echo 'Container cwd:' \$(pwd)
    ls -la ${DATA_DIR_CONT} | head -n 15 || true
    python - <<'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
PY

    export TMPDIR=\$(pwd)/.pip_tmp
    DEPS_DIR=\$(pwd)/.slurm_job_deps
    rm -rf \"\$TMPDIR\" \"\$DEPS_DIR\"
    mkdir -p \"\$TMPDIR\" \"\$DEPS_DIR\"

    # Runtime deps for this REVE pipeline.
    python -m pip install --target \"\$DEPS_DIR\" --no-deps mne
    python -m pip install --target \"\$DEPS_DIR\" --no-deps joblib
    # For REVE/HF stack, install with dependencies to avoid missing transitive imports
    # (e.g. httpcore/httpx required by huggingface_hub).
    python -m pip install --target \"\$DEPS_DIR\" --upgrade \
      einops \
      transformers==4.56.2 \
      huggingface_hub \
      safetensors
    export PYTHONPATH=\"\$DEPS_DIR:\${PYTHONPATH:-}\"

    OUT=\$(pwd)/results/reve_pipeline_slurm_\${SLURM_JOB_ID:-local}
    mkdir -p \"\$OUT\"
    echo \"Pipeline log dir: \$OUT\"
    echo \"Started: \$(date -Is)\" | tee \"\$OUT/pipeline.log\"

    echo '=== 1/3 compute_all_features (for subject mapping cache) ===' | tee -a \"\$OUT/pipeline.log\"
    python scripts/features/compute_all_features.py --config ${CFG_FEATURES} | tee -a \"\$OUT/pipeline.log\"

    echo '=== 2/3 extract_reve_embeddings ===' | tee -a \"\$OUT/pipeline.log\"
    python scripts/embeddings/extract_reve_embeddings.py --config ${CFG_REVE_EMB} | tee -a \"\$OUT/pipeline.log\"

    echo '=== 3/3 train_xgboost_reve ===' | tee -a \"\$OUT/pipeline.log\"
    python scripts/classification/train_xgboost_reve.py --config ${CFG_REVE_XGB} | tee -a \"\$OUT/pipeline.log\"

    echo \"Finished: \$(date -Is)\" | tee -a \"\$OUT/pipeline.log\"
  "

