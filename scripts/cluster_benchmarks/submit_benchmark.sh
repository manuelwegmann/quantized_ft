#!/bin/bash
# Submit all benchmark jobs: GPU throughput (1/2/4/8 GPUs) + data loading sweep.
# Run from the project root: bash scripts/submit_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- GPU compute throughput (1, 2, 4, 8 GPUs) ---
for N in 1 2 4 8; do
    JOB_ID=$(sbatch \
        --job-name="qft_bench_${N}gpu" \
        --gres="gpu:${N}" \
        --export=ALL,N_GPUS="${N}" \
        "${SCRIPT_DIR}/benchmark_slurm.sbatch" \
        | awk '{print $NF}')
    echo "Submitted ${N}-GPU compute job → ${JOB_ID}  (log: qft_bench_${N}gpu-${JOB_ID}.out)"
done

# --- Data loading throughput (CPU / storage benchmark) ---
JOB_ID=$(sbatch "${SCRIPT_DIR}/benchmark_dataloader_slurm.sbatch" | awk '{print $NF}')
echo "Submitted dataloader job   → ${JOB_ID}  (log: qft_bench_dataloader-${JOB_ID}.out)"
