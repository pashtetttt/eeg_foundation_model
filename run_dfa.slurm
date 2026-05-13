#!/bin/bash
#SBATCH --job-name=dfa
#SBATCH --partition=ais-gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/dfa_%j.log
#SBATCH --error=logs/dfa_%j.err

set -euo pipefail

# --- Edit for your Zhores clone / GPFS layout ---
PROJECT_DIR=/trinity/home/pavel.borisenko/lab/gpfs/neuro_ml/eeg_predict/eeg_foundation_model
GPFS_BIND=/trinity/home/pavel.borisenko/lab/gpfs
CONTAINER=docker://nvcr.io/nvidia/pytorch:24.04-py3

# YAML: healthy_data_dir, patients_data_dir, eyes, out_dir, min_scale, order, max_per_class (optional)
CFG_DFA=configs/dfa/default.yml

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

if [ ! -f "${PROJECT_DIR}/${CFG_DFA}" ]; then
  echo "ERROR: DFA config not found: ${PROJECT_DIR}/${CFG_DFA}"
  exit 1
fi

singularity exec \
  --pwd /gpfs/neuro_ml/eeg_predict/eeg_foundation_model \
  -B "${GPFS_BIND}:/gpfs" \
  "${CONTAINER}" \
  bash -lc "
    set -euo pipefail
    export PYTHONNOUSERSITE=1

    echo \"Container pwd: \$(pwd)\"
    echo \"Config: ${CFG_DFA}\"
    head -n 40 ${CFG_DFA} || true

    export TMPDIR=\$(pwd)/.pip_tmp
    DEPS_DIR=\$(pwd)/.slurm_job_deps
    rm -rf \"\$TMPDIR\" \"\$DEPS_DIR\"
    mkdir -p \"\$TMPDIR\" \"\$DEPS_DIR\"
    # mne: EDF; PyYAML: config. Plots use seaborn if present in the image.
    python -m pip install --target \"\$DEPS_DIR\" --no-deps mne
    python -m pip install --target \"\$DEPS_DIR\" --no-deps PyYAML
    export PYTHONPATH=\"\$DEPS_DIR:\$(pwd)/src:\${PYTHONPATH:-}\"

    RUN_NAME=\"slurm_\${SLURM_JOB_ID:-local}\"
    LOG=\"\$(pwd)/logs/dfa_\${SLURM_JOB_ID:-local}_console.txt\"
    mkdir -p \"\$(dirname \"\$LOG\")\"

    set -o pipefail
    python -m scripts.dfa_analysis.run_pipeline --config ${CFG_DFA} --run-name \"\$RUN_NAME\" 2>&1 | tee \"\$LOG\"

    OUT=\$(grep '^DFA_OUTPUT_DIR=' \"\$LOG\" | tail -1 | sed 's/^DFA_OUTPUT_DIR=//')
    if [ -z \"\$OUT\" ]; then
      echo 'ERROR: could not parse DFA_OUTPUT_DIR from log'
      exit 1
    fi
    echo \"DFA results: \$OUT\"

    if [ -f \"\$OUT/dfa_table.csv\" ]; then
      python scripts/dfa_analysis/plot_violin_from_table.py --dfa-table \"\$OUT/dfa_table.csv\" --out \"\$OUT/dfa_violin_subject_level.png\" || echo 'WARN: violin plot failed (seaborn missing or empty table?)'
    fi
  "

echo "Slurm job finished. SLURM logs: logs/dfa_${SLURM_JOB_ID}.log; full console: logs/dfa_${SLURM_JOB_ID}_console.txt"
