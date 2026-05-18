#!/bin/bash
# GPU throughput benchmark — submitted via submit_benchmark.sh
# --gres, --job-name, and N_GPUS are set by the submit script.
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00

N_GPUS="${N_GPUS:-1}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV="${PROJECT_DIR}/venv"

echo "=== qft_benchmark  GPUs=${N_GPUS}  job=${SLURM_JOB_ID}  node=$(hostname) ==="
date

source "${VENV}/bin/activate"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| visible GPUs:', torch.cuda.device_count())"

cd "${PROJECT_DIR}"

BENCH_ARGS="--warmup_steps 3 --steps 10"

if [ "${N_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node="${N_GPUS}" scripts/benchmark_throughput.py ${BENCH_ARGS}
else
    python scripts/benchmark_throughput.py ${BENCH_ARGS}
fi

echo "=== done ===" && date
