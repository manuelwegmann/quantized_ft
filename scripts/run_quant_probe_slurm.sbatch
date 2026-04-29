#!/bin/bash
#SBATCH --job-name=qft_quantprobe
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

# Linear probe evaluation across quantization levels (pre-VQ, FP → W8A8 → W4A8 → W4A4 → W2A4).
# Run after cache_quantized_features_slurm.sbatch has completed.
#
# Usage:
#   sbatch scripts/run_quant_probe_slurm.sbatch

EPOCHS="${EPOCHS:-1000}"
SEEDS="${SEEDS:-5}"
NS="${NS:-100,300,all}"

BACKBONES="pretrained_pre_vq,pretrained_pre_vq_w8a8,pretrained_pre_vq_w4a8,pretrained_pre_vq_w4a4,pretrained_pre_vq_w2a4"
OUTPUT_DIR="runs/quant_probe"

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_quantprobe  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "EPOCHS=${EPOCHS}  SEEDS=${SEEDS}  NS=${NS}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"
python scripts/run_multi_condition.py \
    --backbones   "${BACKBONES}"  \
    --output_dir  "${OUTPUT_DIR}" \
    --epochs      "${EPOCHS}"     \
    --seeds       "${SEEDS}"      \
    --ns          "${NS}"

echo "=== done ===" && date
