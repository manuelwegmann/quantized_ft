#!/bin/bash
#SBATCH --job-name=qft_cache_quant
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

# Cache pre-VQ features for (8,8), (4,8), (4,4), (2,4) quantization configs.
# Run after cache_all_features_slurm.sbatch has completed.
#
# Usage:
#   sbatch scripts/cache_quantized_features_slurm.sbatch

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_cache_quant  job=${SLURM_JOB_ID}  node=$(hostname) ==="
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"
python scripts/cache_quantized_features.py

echo "=== done ===" && date
