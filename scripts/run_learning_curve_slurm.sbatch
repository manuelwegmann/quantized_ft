#!/bin/bash
#SBATCH --job-name=qft_lcurve
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

# Learning curve: pretrained vs random backbone for atelectasis.
# Loads cached features; no CT loading — runs in minutes.
# Submit after qft_cache has completed.
#
# Usage:
#   sbatch scripts/run_learning_curve_slurm.sbatch
#   sbatch --export=ALL,EPOCHS=500,SEEDS=10 scripts/run_learning_curve_slurm.sbatch

EPOCHS="${EPOCHS:-300}"
SEEDS="${SEEDS:-5}"
NS="${NS:-30,100,300,1000,all}"

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_lcurve  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "EPOCHS=${EPOCHS}  SEEDS=${SEEDS}  NS=${NS}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"
python scripts/run_learning_curve.py \
    --epochs "${EPOCHS}" \
    --seeds  "${SEEDS}"  \
    --ns     "${NS}"

echo "=== done ===" && date
