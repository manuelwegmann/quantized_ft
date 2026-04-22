#!/bin/bash
#SBATCH --job-name=qft_downstream
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# Usage:
#   sbatch scripts/run_downstream_slurm.sbatch
#   sbatch --export=ALL,PRETRAIN_CKPT=runs/pretrain_ssql/checkpoint_final.pt \
#          scripts/run_downstream_slurm.sbatch
#   sbatch --export=ALL,MAX_SAMPLES=5 scripts/run_downstream_slurm.sbatch

CONFIG="${CONFIG:-configs/downstream.yaml}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="/home/nlr950/Dir/quantized_ft/venv"

echo "=== qft_downstream  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "CONFIG=${CONFIG}  PRETRAIN_CKPT=${PRETRAIN_CKPT}  MAX_SAMPLES=${MAX_SAMPLES}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

ARGS="--config ${CONFIG}"
if [ -n "${PRETRAIN_CKPT}" ]; then
    ARGS="${ARGS} --pretrain_checkpoint ${PRETRAIN_CKPT}"
fi
if [ -n "${MAX_SAMPLES}" ]; then
    ARGS="${ARGS} --max_samples ${MAX_SAMPLES}"
fi

python run_downstream.py ${ARGS}

echo "=== done ===" && date
