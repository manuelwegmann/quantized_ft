#!/bin/bash

#SBATCH --job-name=qft_crop_abl
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

echo "Checking GPU memory..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU memory: ${GPU_MEM} MB"

# FP-only at bs=2 uses ~39 GB — a 40 GB A100 is sufficient (unlike SSQL which needs 80 GB).
if [ "$GPU_MEM" -lt 38000 ]; then
    echo "Not enough GPU memory — requeueing."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi

echo "GPU OK — starting job"

# Wall-time estimate (H100 or A100-40G, bs=2, FP only):
#   N=3000, 3 epochs per crop ratio, 3 ratios: ~3-4h training
#   Diagnostics (500-scan eval, 3 checkpoints × 3 ratios): ~1-2h
#   Total: ~6h → 10h wall time with headroom

N_PRETRAIN="${N_PRETRAIN:-3000}"
EPOCHS="${EPOCHS:-3}"
CROP_RATIOS="${CROP_RATIOS:-0.75,0.65,0.55}"
N_EVAL="${N_EVAL:-500}"
LR="${LR:-3e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/crop_ablation}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_crop_abl  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "crop_ratios=${CROP_RATIOS}  n_pretrain=${N_PRETRAIN}  epochs=${EPOCHS}  lr=${LR}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

python scripts/run_crop_ablation.py \
    --n_pretrain  "${N_PRETRAIN}"  \
    --epochs      "${EPOCHS}"      \
    --crop_ratios "${CROP_RATIOS}" \
    --n_eval      "${N_EVAL}"      \
    --lr          "${LR}"          \
    --num_workers "${NUM_WORKERS}" \
    --output_dir  "${OUTPUT_DIR}"

echo "=== done ===" && date
