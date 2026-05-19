#!/bin/bash

#SBATCH --job-name=qft_lora_ft
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

echo "Checking GPU memory..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU memory: ${GPU_MEM} MB"

if [ "$GPU_MEM" -lt 38000 ]; then
    echo "Not enough GPU memory — requeueing."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi

echo "GPU OK — starting job"

# LoRA freezes most backbone weights — activation memory is identical to full FT
# (gradients still flow through frozen layers to reach LoRA params).
# Memory footprint is similar to run_supervised_ft; any 40 GB A100 suffices.

N_TRAIN="${N_TRAIN:-300}"
EPOCHS="${EPOCHS:-5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/lora_ft}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_lora_ft  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "n_train=${N_TRAIN}  epochs=${EPOCHS}  rank=${LORA_RANK}  alpha=${LORA_ALPHA}  lr=${LR}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

python scripts/run_lora_ft.py \
    --n_train     "${N_TRAIN}"     \
    --epochs      "${EPOCHS}"      \
    --lora_rank   "${LORA_RANK}"   \
    --lora_alpha  "${LORA_ALPHA}"  \
    --lr          "${LR}"          \
    --num_workers "${NUM_WORKERS}" \
    --output_dir  "${OUTPUT_DIR}"

echo "=== done ===" && date
