#!/bin/bash
#SBATCH --job-name=qft_cache_all
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

# Extract features for ALL on-disk scans for pretrained ViT, random ViT, random CNN.
# Output: runs/feature_cache_full/

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_cache_all  job=${SLURM_JOB_ID}  node=$(hostname) ==="
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"
python scripts/cache_all_features.py --num_workers 8 --batch_size 4

echo "=== done ===" && date
