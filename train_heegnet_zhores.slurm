#!/bin/bash
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/train_heegnet_%j.log
#SBATCH --error=logs/train_heegnet_%j.err

set -euo pipefail

# Project path on Zhores-2
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

# Submodule must already be initialized before job submission.
if [ ! -d "${PROJECT_DIR}/heegnet/HEEGNet/nets" ]; then
  echo "ERROR: heegnet/HEEGNet is missing. Initialize submodules before sbatch:"
  echo "  git submodule update --init --recursive"
  exit 1
fi

# Run training inside NVIDIA PyTorch container
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
    # Pip stages under \$TMPDIR then moves to --target; /tmp vs GPFS -> EXDEV (cross-device).
    export TMPDIR=\$(pwd)/.pip_tmp
    rm -rf \"\$TMPDIR\" \"\$DEPS_DIR\"
    mkdir -p \"\$TMPDIR\" \"\$DEPS_DIR\"
    # Do not run \"pip install --upgrade pip\": container FS is read-only under /usr/local.
    # Only install what is missing from the NGC image. Use --no-deps so pip does not put a
    # second numpy/sklearn into \$DEPS_DIR (that shadows the image and breaks pandas ABI).
    # HEEGNet nets import geoopt; edf_loader imports mne.
    python -m pip install --target \"\$DEPS_DIR\" --no-deps geoopt
    python -m pip install --target \"\$DEPS_DIR\" --no-deps mne
    python -m pip install --target \"\$DEPS_DIR\" --no-deps imbalanced-learn
    export PYTHONPATH=\"\$DEPS_DIR:\${PYTHONPATH:-}\"

    python - <<'PY'
import torch
print('After job deps (expect CUDA=True if driver matches container):', torch.cuda.is_available())
PY

    OUT_RUN=\$(pwd)/results/heegnet_slurm_\${SLURM_JOB_ID:-local}
    mkdir -p \"\$OUT_RUN\"
    python scripts/train_heegnet.py \
      --config configs/heegnet_age4.yaml \
      --data-dir ${DATA_DIR_CONT} \
      --device cuda \
      --output-dir \"\$OUT_RUN\"
  "

