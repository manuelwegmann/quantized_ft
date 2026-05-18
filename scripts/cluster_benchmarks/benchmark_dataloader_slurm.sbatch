#!/bin/bash
#SBATCH --job-name=qft_bench_dataloader
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:45:00
# No GPU needed — this is a CPU / storage benchmark.
# Request enough CPUs to cover the highest num_workers you want to test.

DATA_DIR="${DATA_DIR:-/dcai/users/wegman/data/merlin_dataset/merlinabdominalctdataset/merlin_data}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
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
