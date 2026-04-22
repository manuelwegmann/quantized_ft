#!/bin/bash
#SBATCH --job-name=qft_augtest
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

N_SCANS="${N_SCANS:-3}"

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_augtest  job=${SLURM_JOB_ID}  node=$(hostname) ==="
date

source "${VENV}/bin/activate"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"
python scripts/test_augmentations.py --n_scans "${N_SCANS}"

echo "=== done ===" && date
