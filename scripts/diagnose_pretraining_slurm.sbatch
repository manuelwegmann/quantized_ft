#!/bin/bash
#SBATCH --job-name=qft_diag
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

# Compute per-checkpoint feature geometry diagnostics:
#   uniformity, effective rank, kNN macro-AUROC
#
# Usage:
#   sbatch scripts/diagnose_pretraining_slurm.sbatch
#   sbatch --export=ALL,CHECKPOINT_DIRS="runs/exp_ln_1000/pretrain_fp runs/exp_ln_1000/pretrain_ssql",OUTPUT=runs/exp_ln_1000/diagnostics.json \
#          scripts/diagnose_pretraining_slurm.sbatch

echo "Checking GPU memory..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "GPU memory: ${GPU_MEM} MB"
if [ "$GPU_MEM" -lt 20000 ]; then
    echo "Not enough GPU memory — requeueing."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi
echo "GPU OK — starting job"

CHECKPOINT_DIRS="${CHECKPOINT_DIRS:-runs/mini_experiment_ln/pretrain_fp runs/mini_experiment_ln/pretrain_ssql}"
OUTPUT="${OUTPUT:-runs/mini_experiment_ln/diagnostics.json}"
N_EVAL="${N_EVAL:-500}"
K="${K:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_diag  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "checkpoint_dirs=${CHECKPOINT_DIRS}"
echo "output=${OUTPUT}"
date

source "${VENV}/bin/activate"
cd "${PROJECT_DIR}"

python scripts/diagnose_pretraining.py \
    --checkpoint_dirs ${CHECKPOINT_DIRS} \
    --output          "${OUTPUT}"        \
    --n_eval          "${N_EVAL}"        \
    --k               "${K}"            \
    --num_workers     "${NUM_WORKERS}"

echo "=== done ===" && date
