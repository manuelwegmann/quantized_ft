#!/bin/bash
#SBATCH --job-name=qft_cache
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

# Extract and cache 512-dim features for all 1943 atelectasis-labeled scans.
# Runs pretrained and random backbones in sequence.
# Estimated runtime: ~4h per backbone, ~8h total.
# Output: runs/feature_cache/{pretrained,random}/{train,val,test}_{feats,labels}.pt

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_cache  job=${SLURM_JOB_ID}  node=$(hostname) ==="
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"
python scripts/cache_probe_features.py --num_workers 8 --batch_size 4

echo "=== done ===" && date
