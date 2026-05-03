#!/bin/bash
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/train_eegpt_%j.log
#SBATCH --error=logs/train_eegpt_%j.err

set -euo pipefail

# Project path on Zhores-2
PROJECT_DIR=/trinity/home/pavel.borisenko/lab/gpfs/neuro_ml/eeg_predict/eeg_foundation_model
GPFS_BIND=/trinity/home/pavel.borisenko/lab/gpfs
CONTAINER=docker://nvcr.io/nvidia/pytorch:24.04-py3
DATA_DIR_HOST=/trinity/home/pavel.borisenko/lab/gpfs/neuro_ml/eeg_predict/eeg_foundation_model/data
DATA_DIR_CONT=/gpfs/neuro_ml/eeg_predict/eeg_foundation_model/data

# Update this if your checkpoint is stored elsewhere
CKPT_PATH=EEGPT_files/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

if [ ! -d "${DATA_DIR_HOST}" ]; then
  echo "ERROR: data directory not found on host: ${DATA_DIR_HOST}"
  exit 1
fi

# Submodules must already be initialized before job submission.
if [ ! -d "${PROJECT_DIR}/eegpt/EEGPT/downstream" ] || [ ! -d "${PROJECT_DIR}/heegnet/HEEGNet" ]; then
  echo "ERROR: eegpt/heegnet submodules are missing. Initialize before sbatch:"
  echo "  git submodule update --init --recursive"
  exit 1
fi

singularity exec --nv \
  --pwd /gpfs/neuro_ml/eeg_predict/eeg_foundation_model \
  -B "${GPFS_BIND}:/gpfs" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    # Do not load ~/.local Python packages: they often install a user torch that
    # mismatches the cluster driver and shadows the container CUDA build.
    export PYTHONNOUSERSITE=1

    echo 'Container cwd:' \$(pwd)
    echo 'Container data dir:' ${DATA_DIR_CONT}
    ls -la ${DATA_DIR_CONT} | head -n 20
    python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device count:', torch.cuda.device_count())
    print('cuda device 0:', torch.cuda.get_device_name(0))
PY

    DEPS_DIR=\$(pwd)/.slurm_job_deps
    mkdir -p \"\$DEPS_DIR\"
    python -m pip install --upgrade pip
    python -m pip install --target \"\$DEPS_DIR\" --no-deps geoopt
    python -m pip install --target \"\$DEPS_DIR\" pyyaml mne antropy imbalanced-learn xgboost
    export PYTHONPATH=\"\$DEPS_DIR:\${PYTHONPATH:-}\"

    python - <<'PY'
import torch
print('After job deps (expect CUDA=True if driver matches container):', torch.cuda.is_available())
PY

    python scripts/train_eegpt_age4.py \
      --config configs/eegpt_age4.yaml \
      --checkpoint ${CKPT_PATH} \
      --data-dir ${DATA_DIR_CONT} \
      --device cuda
  "

