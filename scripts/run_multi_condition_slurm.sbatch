#!/bin/bash
#SBATCH --job-name=qft_multicond
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl

# Run multi-condition probe evaluation. Submit after qft_cache_all completes.
#
# Usage:
#   sbatch scripts/run_multi_condition_slurm.sbatch
#   sbatch --export=ALL,EPOCHS=1000,SEEDS=5 scripts/run_multi_condition_slurm.sbatch

EPOCHS="${EPOCHS:-1000}"
SEEDS="${SEEDS:-5}"
NS="${NS:-100,300,all}"

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_multicond  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "EPOCHS=${EPOCHS}  SEEDS=${SEEDS}  NS=${NS}"
date

source "${VENV}/bin/activate"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "${PROJECT_DIR}"
python scripts/run_multi_condition.py \
    --epochs "${EPOCHS}" \
    --seeds  "${SEEDS}"  \
    --ns     "${NS}"

echo "=== done ===" && date
