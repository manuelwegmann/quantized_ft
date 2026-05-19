#!/bin/bash

#SBATCH --job-name=qft_lr_sweep
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

echo "Checking GPU memory..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU memory: ${GPU_MEM} MB"

# FP-only SimSiam at bs=2 — same memory profile as mini experiment FP runs.
# Any 40 GB A100 is sufficient.
if [ "$GPU_MEM" -lt 38000 ]; then
    echo "Not enough GPU memory — requeueing."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi

echo "GPU OK — starting job"

# Wall-time estimate (A100-40G or H100, bs=2, FP only):
#   N=1500, 2 epochs × 4 conditions: ~90 min training
#   Diagnostics (500 scans, 2 checkpoints × 4 conditions): ~60 min
#   Total: ~2.5h → 6h wall time gives ample headroom

N_PRETRAIN="${N_PRETRAIN:-1500}"
EPOCHS="${EPOCHS:-2}"
CROP_RATIO="${CROP_RATIO:-0.65}"
LR_PAIRS="${LR_PAIRS:-3e-4:3e-4,3e-4:3e-5,3e-4:3e-6,3e-3:3e-5}"
N_EVAL="${N_EVAL:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/lr_sweep}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_lr_sweep  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "n_pretrain=${N_PRETRAIN}  epochs=${EPOCHS}  crop_ratio=${CROP_RATIO}"
echo "lr_pairs=${LR_PAIRS}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

python scripts/run_lr_sweep.py \
    --n_pretrain  "${N_PRETRAIN}"  \
    --epochs      "${EPOCHS}"      \
    --crop_ratio  "${CROP_RATIO}"  \
    --lr_pairs    "${LR_PAIRS}"    \
    --n_eval      "${N_EVAL}"      \
    --num_workers "${NUM_WORKERS}" \
    --output_dir  "${OUTPUT_DIR}"

echo "=== done ===" && date
