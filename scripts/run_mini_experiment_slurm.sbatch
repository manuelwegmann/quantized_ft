#!/bin/bash
#SBATCH --job-name=qft_mini
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

# Mini experiment: FP pretrain → SSQL pretrain → feature extraction → linear probe.
#
# Run everything in one job (default, N=300):
#   sbatch scripts/run_mini_experiment_slurm.sbatch
#
# Run FP and SSQL as parallel jobs on two nodes, then probe separately:
#   sbatch --export=ALL,PHASE=pretrain_fp,N_PRETRAIN=1000,SAVE_EVERY=5,OUTPUT_DIR=runs/exp_1000 \
#          scripts/run_mini_experiment_slurm.sbatch
#   sbatch --export=ALL,PHASE=pretrain_ssql,N_PRETRAIN=1000,SAVE_EVERY=5,OUTPUT_DIR=runs/exp_1000 \
#          scripts/run_mini_experiment_slurm.sbatch
#   # after both complete:
#   sbatch --export=ALL,PHASE=probe,OUTPUT_DIR=runs/exp_1000 \
#          scripts/run_mini_experiment_slurm.sbatch
#
# Wall-time guidance (A100, batch_size=2):
#   N=300,  40 epochs: FP ~3.8h, SSQL ~3.8h, eval ~6h  → total ~14h (all-in-one)
#   N=1000, 40 epochs: FP ~9h,   SSQL ~9h               → each fits in 20h (parallel)
#   Probe alone (any N): ~6h

N_PRETRAIN="${N_PRETRAIN:-300}"
EPOCHS="${EPOCHS:-40}"
SAVE_EVERY="${SAVE_EVERY:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PROBE_EPOCHS="${PROBE_EPOCHS:-500}"
SEEDS="${SEEDS:-3}"
QUANT_W="${QUANT_W:-4}"
QUANT_A="${QUANT_A:-4}"
NORM="${NORM:-ln}"
LR="${LR:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_WORKERS_EVAL="${NUM_WORKERS_EVAL:-2}"
PHASE="${PHASE:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/mini_experiment_${NORM}}"

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_mini  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "phase=${PHASE}  N_PRETRAIN=${N_PRETRAIN}  EPOCHS=${EPOCHS}  bs=${BATCH_SIZE}  save_every=${SAVE_EVERY}  quant=W${QUANT_W}A${QUANT_A}  norm=${NORM}  lr=${LR:-auto}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"
LR_ARG=()
[ -n "${LR}" ] && LR_ARG=(--lr "${LR}")

python scripts/run_mini_experiment.py \
    --phase            "${PHASE}"            \
    --n_pretrain       "${N_PRETRAIN}"       \
    --epochs           "${EPOCHS}"           \
    --save_every       "${SAVE_EVERY}"       \
    --batch_size       "${BATCH_SIZE}"       \
    --probe_epochs     "${PROBE_EPOCHS}"     \
    --seeds            "${SEEDS}"            \
    --quant_w          "${QUANT_W}"          \
    --quant_a          "${QUANT_A}"          \
    --norm             "${NORM}"             \
    "${LR_ARG[@]}"                           \
    --num_workers      "${NUM_WORKERS}"      \
    --num_workers_eval "${NUM_WORKERS_EVAL}" \
    --output_dir       "${OUTPUT_DIR}"

echo "=== done ===" && date
