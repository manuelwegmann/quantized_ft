#!/bin/bash
#SBATCH --job-name=qft_cnn
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

# Usage:
#   sbatch scripts/cnn_baseline_slurm.sbatch
#   sbatch --export=ALL,EPOCHS=500 scripts/cnn_baseline_slurm.sbatch

EPOCHS="${EPOCHS:-300}"

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_cnn  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "EPOCHS=${EPOCHS}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"
python scripts/cnn_baseline.py --epochs "${EPOCHS}" --num_workers 8 --batch_size 4

echo "=== done ===" && date
