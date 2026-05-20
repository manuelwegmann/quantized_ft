#!/bin/bash

#SBATCH --job-name=qft_bn_exp
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# H100 required: bfloat16 + Flash Attention gives bs=8 at ~71 GB (out of 79 GB).
# bs=8 makes BatchNorm statistics meaningful (vs bs=2 with LayerNorm before).
#
# LR kept at 3e-4 (same as the stable bs=2 baseline, not upscaled).
# Conservative choice — supervised FT results showed lower LRs generalise better.
# Using cosine LR schedule as before.
#
# Wall-time estimate (H100, bs=8, FP only, N=5097, 5 epochs):
#   5097/bs=8 = 637 batches/epoch × 5 epochs × ~2s/batch ≈ 1.8h training
#   + diagnostics (5 checkpoints × 500 scans) ≈ 1h
#   → 12h wall time with ample headroom

N_PRETRAIN="${N_PRETRAIN:-5097}"
EPOCHS="${EPOCHS:-5}"
SAVE_EVERY="${SAVE_EVERY:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
NORM="${NORM:-bn}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/bn_experiment}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_bn_exp  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)"
echo "n_pretrain=${N_PRETRAIN}  epochs=${EPOCHS}  bs=${BATCH_SIZE}  lr=${LR}  norm=${NORM}"
echo "amp=True  flash=True"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"

# FP pretraining only — testing whether BN prevents collapse before committing to full pipeline
python scripts/run_mini_experiment.py \
    --phase        pretrain_fp         \
    --n_pretrain   "${N_PRETRAIN}"     \
    --epochs       "${EPOCHS}"         \
    --save_every   "${SAVE_EVERY}"     \
    --batch_size   "${BATCH_SIZE}"     \
    --lr           "${LR}"            \
    --norm         "${NORM}"           \
    --num_workers  "${NUM_WORKERS}"    \
    --output_dir   "${OUTPUT_DIR}"     \
    --use_amp                          \
    --use_flash

echo ""
echo "=== Pretraining done — running diagnostics ==="

python scripts/diagnose_pretraining.py \
    --checkpoint_dirs "${OUTPUT_DIR}/pretrain_fp" \
    --output          "${OUTPUT_DIR}/diagnostics.json"

echo "=== done ===" && date