#!/bin/bash

#SBATCH --job-name=qft_sup_ft
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

# Single backbone forward + linear head at bs=2 — much lighter than SimSiam.
# Any 40 GB A100 is sufficient.
if [ "$GPU_MEM" -lt 38000 ]; then
    echo "Not enough GPU memory — requeueing."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi

echo "GPU OK — starting job"

# Wall-time estimate (A100, bs=2, n_train=300):
#   ~5-7 min/epoch × 5 epochs training + validation = ~45 min
#   4h wall time gives ample headroom for larger n_train overrides.

N_TRAIN="${N_TRAIN:-300}"
EPOCHS="${EPOCHS:-5}"
BACKBONE_LR="${BACKBONE_LR:-1e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/supervised_ft}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_sup_ft  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "n_train=${N_TRAIN}  epochs=${EPOCHS}  backbone_lr=${BACKBONE_LR}  head_lr=${HEAD_LR}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

python scripts/run_supervised_ft.py \
    --n_train     "${N_TRAIN}"     \
    --epochs      "${EPOCHS}"      \
    --backbone_lr "${BACKBONE_LR}" \
    --head_lr     "${HEAD_LR}"     \
    --num_workers "${NUM_WORKERS}" \
    --output_dir  "${OUTPUT_DIR}"

echo "=== done ===" && date
