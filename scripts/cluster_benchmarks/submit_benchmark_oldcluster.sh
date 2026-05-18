#!/bin/bash
# Submit GPU throughput benchmark jobs for the old cluster (A100s).
# Mirrors submit_benchmark.sh — GPU counts capped at 4 (max single-node A100s available).
# Run from the project root: bash scripts/submit_benchmark_oldcluster.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for N in 1 2 4; do
    JOB_ID=$(sbatch \
        --job-name="qft_bench_${N}gpu" \
        --gres="gpu:a100:${N}" \
        --export=ALL,N_GPUS="${N}" \
        "${SCRIPT_DIR}/benchmark_slurm_oldcluster.sbatch" \
        | awk '{print $NF}')
    echo "Submitted ${N}-GPU compute job → ${JOB_ID}  (log: qft_bench_${N}gpu-${JOB_ID}.out)"
done
