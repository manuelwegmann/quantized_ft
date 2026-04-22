#!/bin/bash
#SBATCH --job-name=qft_extract
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Usage:
#   sbatch scripts/extract_features_slurm.sbatch
#   sbatch --export=ALL,N_SCANS=20 scripts/extract_features_slurm.sbatch

N_SCANS="${N_SCANS:-5}"

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_extract  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "N_SCANS=${N_SCANS}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"
python scripts/extract_features.py --n_scans "${N_SCANS}"

echo "=== done ===" && date
