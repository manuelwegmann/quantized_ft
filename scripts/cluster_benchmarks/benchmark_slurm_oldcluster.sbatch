#!/bin/bash
#SBATCH --job-name=qft_bench_1gpu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00

PROJECT_DIR="/home/nlr950/Dir/quantized_ft"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_bench_1gpu (old cluster)  job=${SLURM_JOB_ID}  node=$(hostname) ==="
date

source "${VENV}/bin/activate"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"

cd "${PROJECT_DIR}"

python scripts/benchmark_throughput.py --warmup_steps 5 --steps 1500

echo "=== done ===" && date
