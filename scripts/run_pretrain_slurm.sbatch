#!/bin/bash
#SBATCH --job-name=qft_pretrain
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

# Usage:
#   sbatch scripts/run_pretrain_slurm.sbatch                            # FP mode
#   sbatch --export=ALL,CONFIG=configs/pretrain_ssql.yaml \
#          scripts/run_pretrain_slurm.sbatch                            # SSQL mode
#   sbatch --export=ALL,MAX_SAMPLES=3 scripts/run_pretrain_slurm.sbatch # smoke test

CONFIG="${CONFIG:-configs/pretrain_fp.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="/home/nlr950/Dir/quantized_ft/venv"

echo "=== qft_pretrain  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "CONFIG=${CONFIG}  MAX_SAMPLES=${MAX_SAMPLES}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

ARGS="--config ${CONFIG}"
if [ -n "${MAX_SAMPLES}" ]; then
    ARGS="${ARGS} --max_samples ${MAX_SAMPLES}"
fi

python run_pretrain.py ${ARGS}

echo "=== done ===" && date
