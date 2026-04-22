# quantized_ft — Project Notes

## Goal

Compare two SimSiam self-supervised pretraining regimes on the CT-CLIP ViT
image encoder, then evaluate downstream classification performance on the
Merlin abdominal CT dataset:

1. **FP baseline** — standard full-precision SimSiam
2. **SSQL** — SimSiam where the prediction branch uses a uniform quantizer
   with randomly sampled bit-widths per step (weights 2–8 bit,
   activations 4–8 bit), following the SSQL recipe (ECCV 2022)

Downstream evaluation: linear probe (frozen backbone), 30-label multi-label
classification, macro-mean AUROC.

---

## Completed tests

### Backbone checkpoint loading
- Script: `scripts/extract_features.py`
- Loaded 157/159 tensors from `CT-CLIP_v2.pt` (`visual_transformer.*` keys)
- 2 missing keys (`vq._codebook.embed_avg`, `vq.zero`) are VQ EMA training
  buffers — not needed for forward inference
- Spot-checked: loaded weight values match checkpoint byte-for-byte
- **Result: checkpoint loading confirmed correct**

### Full-precision forward pass (5 scans)
- Script: `scripts/extract_features.py`, job `qft_extract-8169`
- GPU: NVIDIA A100-PCIE-40GB, CUDA 12.8, torch 2.10.0+cu128
- All 5 scans produced (512,) embeddings with no NaN
- Mean ≈ 0 and std ≈ 0.01 per scan — expected given VQ + global mean pool
  (CLT: 1/sqrt(13824 tokens) × sqrt(512 dims) ≈ 0.24 L2 norm, confirmed)
- Pairwise cosine similarities: 0.83–0.98 (high; reflects VQ averaging
  and domain shift — CT-CLIP trained on chest CTs, Merlin is abdominal)
- **Result: forward pass works, backbone produces valid embeddings**

---

## Intermediate tests remaining (in order)

### 1. Augmentation diversity test  ← next
- Script: `scripts/test_augmentations.py`
- For each scan: extract features from raw + two augmented views
- Check cosine similarities: raw vs v1, raw vs v2, v1 vs v2
- Key question: do augmentations produce meaningfully different representations?
- Reference: compare within-scan (v1 vs v2) to between-scan similarity
- **Pass criterion**: v1 vs v2 < raw similarity AND within-scan > between-scan

### 2. Single FP training step
- Load 1–2 scans, run one full forward + backward pass (FP SimSiam)
- Check: loss is finite, gradients are non-None and non-zero on all modules
- Validates the full FP training loop before committing to long runs

### 3. Single SSQL training step
- Same as above with `quantized_forward` context manager active
- Extra checks:
  - Loss finite under fake-quantized weights
  - Gradients flow back through STE (non-zero on backbone params)
  - Backbone weights are exactly restored after context exits
- Validates SSQL training loop and STE implementation

### 4. Downstream pipeline smoke test
- Load a small labelled subset, verify 70/15/15 split is consistent
- Run frozen backbone → LinearProbe → BCE loss → backward
- Confirm AUROC computation runs to completion
- Validates the evaluation pipeline before a multi-hour pretraining run

---

## Full experiments (after all tests pass)

1. FP SimSiam pretraining — full Merlin dataset, 100 epochs
   ```bash
   sbatch scripts/run_pretrain_slurm.sbatch
   ```

2. SSQL SimSiam pretraining — full Merlin dataset, 100 epochs
   ```bash
   sbatch --export=ALL,CONFIG=configs/pretrain_ssql.yaml scripts/run_pretrain_slurm.sbatch
   ```

3. Downstream linear probe evaluation for both checkpoints
   ```bash
   sbatch --export=ALL,PRETRAIN_CKPT=runs/pretrain_fp/checkpoint_final.pt \
          scripts/run_downstream_slurm.sbatch
   sbatch --export=ALL,PRETRAIN_CKPT=runs/pretrain_ssql/checkpoint_final.pt \
          scripts/run_downstream_slurm.sbatch
   ```

4. Compare macro-mean AUROC between FP and SSQL runs

---

## Environment

- Venv: `/home/nlr950/Dir/quantized_ft/venv` (Python 3.11)
- torch `2.10.0+cu128`, torchvision `0.25.0+cu128`
- Cluster GPU nodes expose CUDA 12.8; cu128 build required
- CT-CLIP packages added via `sys.path` (not pip-installed):
  - `/home/nlr950/Dir/CT-CLIP/transformer_maskgit`
  - `/home/nlr950/Dir/CT-CLIP/CT_CLIP`

## Key data paths

| Resource       | Path |
|----------------|------|
| Checkpoint     | `/home/nlr950/Dir/CT-CLIP/checkpoints/CT-CLIP_v2.pt` |
| Merlin scans   | `/home/nlr950/Dir/CT-CLIP/data/merlin_data/*.nii.gz` |
| Reports        | `/home/nlr950/Dir/CT-CLIP/data/reports_final.xlsx` |
| Labels (30)    | `/home/nlr950/Dir/CT-CLIP/data/zero_shot_findings_disease_cls.csv` |
| Metadata       | `/home/nlr950/Dir/CT-CLIP/data/metadata.csv` |
