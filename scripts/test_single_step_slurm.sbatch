#!/bin/bash
#SBATCH --job-name=qft_step
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Usage:
#   sbatch scripts/test_single_step_slurm.sbatch           # both FP and SSQL
#   sbatch --export=ALL,MODE=fp   scripts/test_single_step_slurm.sbatch
#   sbatch --export=ALL,MODE=ssql scripts/test_single_step_slurm.sbatch

MODE="${MODE:-both}"

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_step  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "MODE=${MODE}"
date

source "${VENV}/bin/activate"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"
python scripts/test_single_step.py --mode "${MODE}"

echo "=== done ===" && date
