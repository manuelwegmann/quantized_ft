# quantized_ft

Self-supervised pretraining experiments comparing full-precision (FP) SimSiam
against SSQL (quantization-regularised SimSiam) on the CT-CLIP ViT image
encoder, with downstream evaluation on the Merlin abdominal CT dataset.

## Repository layout

```
quantized_ft/
├── configs/            YAML configs for run_pretrain.py / run_downstream.py
├── downstream/         Dataset, trainer, evaluator for the linear-probe stage
├── models/             Backbone wrapper, SimSiam heads, quantization utilities
├── pretrain/           Augmentations, dataset, trainer for SSL pretraining
├── scripts/            Experiment entry points + Slurm sbatch scripts
├── paths.py            Central path configuration (reads CT_CLIP_ROOT env var)
├── run_pretrain.py     Full-scale pretraining entry point
├── run_downstream.py   Full-scale downstream evaluation entry point
├── install.sh          One-shot venv + dependency installer
└── requirements.txt    Python dependencies (PyTorch installed separately)
```

## Setup

### 1. Clone dependencies

This project uses CT-CLIP as an external package (added to `sys.path` at
runtime). It must live at a known location:

```bash
# Option A — sibling directory (default, no env var needed)
cd ..
git clone https://github.com/Project-MONAI/CT-CLIP.git   # or your fork

# Option B — arbitrary location
export CT_CLIP_ROOT=/path/to/CT-CLIP
```

Also make sure the CT-CLIP checkpoint and Merlin data are available under
`$CT_CLIP_ROOT` at the standard sub-paths:

```
$CT_CLIP_ROOT/
├── checkpoints/CT-CLIP_v2.pt
└── data/
    ├── merlin_data/          # *.nii.gz scan files
    ├── reports_final.xlsx
    ├── metadata.csv
    └── zero_shot_findings_disease_cls.csv
```

### 2. Install Python dependencies

```bash
bash install.sh
```

This creates `venv/`, installs PyTorch 2.10 (cu128 build), and all other
requirements. For a different CUDA version edit the `--index-url` line inside
`install.sh`.

Activate the venv before running anything:

```bash
source venv/bin/activate
```

## Running experiments

### Mini end-to-end experiment (FP + SSQL, N=300 scans)

```bash
sbatch scripts/run_mini_experiment_slurm.sbatch
```

Run FP and SSQL as parallel jobs:

```bash
sbatch --export=ALL,PHASE=pretrain_fp  scripts/run_mini_experiment_slurm.sbatch
sbatch --export=ALL,PHASE=pretrain_ssql scripts/run_mini_experiment_slurm.sbatch
# after both complete:
sbatch --export=ALL,PHASE=probe        scripts/run_mini_experiment_slurm.sbatch
```

### Full-scale pretraining

```bash
sbatch scripts/run_pretrain_slurm.sbatch                                  # FP
sbatch --export=ALL,CONFIG=configs/pretrain_ssql.yaml \
       scripts/run_pretrain_slurm.sbatch                                  # SSQL
```

### Downstream linear probe

```bash
sbatch --export=ALL,PRETRAIN_CKPT=runs/pretrain_fp/checkpoint_final.pt \
       scripts/run_downstream_slurm.sbatch
```

## Cluster benchmarking

Two benchmarks for comparing GPU compute and storage speed across clusters.
Both are designed to be lightweight and produce directly comparable numbers.

### GPU compute throughput (synthetic data)

Measures raw forward+backward speed using random tensors — no real data or
checkpoint needed. Run on each cluster you want to compare.

**Gefion** (1 / 2 / 4 / 8 H100s):
```bash
bash scripts/submit_benchmark.sh
```

**Old cluster** (1 A100):
```bash
sbatch scripts/benchmark_slurm_oldcluster.sbatch
```

### Data loading throughput (real scans)

Measures the full preprocessing pipeline (NIfTI load → resize → augment)
from disk, sweeping `num_workers` 1 → 4 → 8 → 16. Uses a fixed 50-scan
subset so results are identical in scope across clusters.

**Gefion:**
```bash
sbatch scripts/benchmark_dataloader_slurm.sbatch
```

**Old cluster:**
```bash
sbatch scripts/benchmark_dataloader_slurm_oldcluster.sbatch
```

The data path defaults are already set correctly in each script. Both scripts
use identical `--max_files`, `--warmup_batches`, `--timed_batches`, and
`--num_workers` so the numbers can be compared directly.

### Reading the results

Each job writes a log file named `<job-name>-<job-id>.out`. Look for the
`===` summary block at the end:

```
====================================================
  GPU type       : NVIDIA H100 80GB HBM3
  GPU count      : 4
  ms / step      : 2340 ms
  Throughput     : 0.034 samples/sec  (all GPUs)
  Per-GPU        : 0.009 samples/sec
====================================================
```

For the dataloader benchmark, the table at the end shows the best
`num_workers` setting — use that value in the training configs when running
full pretraining on that cluster.

## Path configuration

All hardcoded paths have been removed. Path resolution happens in `paths.py`:

| Variable       | Env var         | Default                         |
|----------------|-----------------|---------------------------------|
| `CT_CLIP_ROOT` | `CT_CLIP_ROOT`  | `../CT-CLIP` (sibling dir)      |
| `QFT_ROOT`     | `QFT_ROOT`      | project root (auto-detected)    |

Set these in your shell or in the Slurm environment before submitting jobs:

```bash
export CT_CLIP_ROOT=/scratch/yourname/CT-CLIP
export QFT_ROOT=/scratch/yourname/quantized_ft
```
