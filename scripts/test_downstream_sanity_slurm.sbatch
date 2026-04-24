#!/bin/bash
#SBATCH --job-name=qft_sanity
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

# Usage:
#   sbatch scripts/test_downstream_sanity_slurm.sbatch                        # full run
#   sbatch --export=ALL,EPOCHS=50 scripts/test_downstream_sanity_slurm.sbatch # fewer epochs
#   sbatch --export=ALL,MAX_SAMPLES=30,EPOCHS=5 scripts/test_downstream_sanity_slurm.sbatch  # smoke test

EPOCHS="${EPOCHS:-100}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_sanity  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "EPOCHS=${EPOCHS}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"
ARGS="--epochs ${EPOCHS} --num_workers 8"
if [ -n "${MAX_SAMPLES}" ]; then
    ARGS="${ARGS} --max_samples ${MAX_SAMPLES}"
fi

python scripts/test_downstream_sanity.py ${ARGS}

echo "=== done ===" && date
