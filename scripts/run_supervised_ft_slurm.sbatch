#!/bin/bash

#SBATCH --job-name=qft_sup_ft
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu17fl

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)  (${GPU_MEM} MB)"

# Wall-time estimate (A100-40G, bs=2, ~3138 training scans, 15 epochs):
#   ~1569 batches/epoch × 15 epochs × ~1.5s/batch ≈ 9.5h training
#   + val/test evaluation ≈ 2h
#   → 16h wall time

EPOCHS="${EPOCHS:-15}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
HEAD_LR="${HEAD_LR:-1e-4}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
N_TRAIN="${N_TRAIN:--1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/supervised_ft}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_sup_ft  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "epochs=${EPOCHS}  backbone_lr=${BACKBONE_LR}  head_lr=${HEAD_LR}"
echo "accum_steps=${ACCUM_STEPS}  batch_size=${BATCH_SIZE}  n_train=${N_TRAIN}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

python scripts/run_supervised_ft.py \
    --epochs       "${EPOCHS}"       \
    --backbone_lr  "${BACKBONE_LR}"  \
    --head_lr      "${HEAD_LR}"      \
    --accum_steps  "${ACCUM_STEPS}"  \
    --batch_size   "${BATCH_SIZE}"   \
    --n_train      "${N_TRAIN}"      \
    --num_workers  "${NUM_WORKERS}"  \
    --output_dir   "${OUTPUT_DIR}"

echo "=== done ===" && date