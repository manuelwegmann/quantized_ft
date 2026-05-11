#!/bin/bash
#SBATCH --job-name=qft_bench_dataloader
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:45:00

# Identical parameters to benchmark_dataloader_slurm.sbatch — only the
# data path differs so the comparison is apples-to-apples.
DATA_DIR="${DATA_DIR:-/home/nlr950/Dir/CT-CLIP/data/merlin_data}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_bench_dataloader  job=${SLURM_JOB_ID}  node=$(hostname) ==="
echo "DATA_DIR=${DATA_DIR}"
date

source "${VENV}/bin/activate"
python --version

cd "${PROJECT_DIR}"

python scripts/benchmark_dataloader.py \
    --data_dir "${DATA_DIR}" \
    --max_files 50 \
    --warmup_batches 2 \
    --timed_batches 5 \
    --num_workers 1 4 8 16

echo "=== done ===" && date
