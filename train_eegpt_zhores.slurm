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
CONTAINER=docker://nvcr.io/nvidia/pytorch:24.07-py3
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
    echo 'Container cwd:' \$(pwd)
    echo 'Container data dir:' ${DATA_DIR_CONT}
    ls -la ${DATA_DIR_CONT} | head -n 20
    python -m pip install --user --upgrade pip &&
    # Do NOT install torch here: keep container's CUDA-matched torch build.
    python -m pip install --user pyyaml pandas geoopt mne antropy imbalanced-learn xgboost &&
    python scripts/train_eegpt_age4.py \
      --config configs/eegpt_age4.yaml \
      --checkpoint ${CKPT_PATH} \
      --data-dir ${DATA_DIR_CONT} \
      --device cuda
  "

