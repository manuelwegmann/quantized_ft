#!/bin/bash

#SBATCH --job-name=qft_lora_ft
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu17fl

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)  (${GPU_MEM} MB)"

EPOCHS="${EPOCHS:-15}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
BATCH_SIZE="${BATCH_SIZE:-2}"
N_TRAIN="${N_TRAIN:--1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/lora_ft}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_lora_ft  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "epochs=${EPOCHS}  lr=${LR}  rank=${LORA_RANK}  alpha=${LORA_ALPHA}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

python scripts/run_lora_ft.py \
    --epochs      "${EPOCHS}"      \
    --lr          "${LR}"          \
    --lora_rank   "${LORA_RANK}"   \
    --lora_alpha  "${LORA_ALPHA}"  \
    --batch_size  "${BATCH_SIZE}"  \
    --n_train     "${N_TRAIN}"     \
    --num_workers "${NUM_WORKERS}" \
    --output_dir  "${OUTPUT_DIR}"

echo "=== done ===" && date