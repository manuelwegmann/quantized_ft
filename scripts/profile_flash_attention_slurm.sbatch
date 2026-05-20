#!/bin/bash

#SBATCH --job-name=qft_flash_prof
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu17fl

# Lightweight profiling job — no memory threshold, accepts any GPU.
# Low resource request so it queues and starts quickly.

BATCH_SIZES="${BATCH_SIZES:-1,2,4,8}"
DTYPE="${DTYPE:-float32}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_flash_prof  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)"
echo "batch_sizes=${BATCH_SIZES}  dtype=${DTYPE}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

python scripts/profile_flash_attention.py \
    --batch_sizes "${BATCH_SIZES}" \
    --dtype       "${DTYPE}"

echo "=== done ===" && date