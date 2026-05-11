#!/bin/bash
# 1-GPU compute benchmark for the old cluster (A100).
# Identical parameters to benchmark_slurm.sbatch — hardware difference is the signal.
#SBATCH --job-name=qft_bench_1gpu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_bench_1gpu (old cluster)  job=${SLURM_JOB_ID}  node=$(hostname) ==="
date

source "${VENV}/bin/activate"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"

cd "${PROJECT_DIR}"

python scripts/benchmark_throughput.py --warmup_steps 3 --steps 10

echo "=== done ===" && date
