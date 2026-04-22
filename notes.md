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
## To-Do
- Check CT preprocessing pipeline in CT-CLIP (and Merlin).

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

### Augmentation diversity test (3 scans)
- Script: `scripts/test_augmentations.py`, job `qft_augtest-8191`
- Mean within-scan similarity (v1 vs v2): 0.9524
- Mean between-scan similarity (raw): 0.8824
- Margin (within − between): +0.07 — small but positive ✓
- Augmentations create real diversity: raw vs augmented view drops to 0.58–0.97
  depending on the scan, while v1 vs v2 stays high (0.89–0.98), indicating
  the backbone has natural invariance to the applied transforms
- High between-scan similarity (0.83–0.98) reflects two compounding factors:
  (1) VQ + global mean pool discards fine-grained spatial information, leaving
  only a coarse histogram of codebook activations; (2) domain shift —
  CT-CLIP was trained on chest CTs, Merlin is abdominal, so all Merlin scans
  activate a similar subset of the codebook
- SimSiam does not use negative pairs so the small margin is sufficient signal;
  improved between-scan discrimination is expected as an indirect byproduct of
  view-alignment pretraining
- **Result: augmentations working as intended, ready to proceed**

### Single training step — FP and SSQL
- Script: `scripts/test_single_step.py`, job `qft_step-8315`
- FP: loss = −0.006 (finite ✓), all active backbone/projector/predictor params
  have non-zero gradients ✓
- SSQL: w_bits=3, a_bits=4; L_ssql = 0.025, L_fp = 0.020, total = 0.046
  (finite ✓); weight restoration confirmed True ✓; gradients non-zero ✓
- Dead parameters in both modes are benign and confirmed expected:
  - `grad=None`: decoder layers (to_pixels*), unused patch emb
    (to_patch_emb_first_frame), cross-attention norm (context_norm)
  - `grad=0` (structural): `null_kv` in every attention layer — pre-trained
    model never attends to null tokens; does not affect learning
- **Result: FP PASSED ✓, SSQL PASSED ✓ — training loop and STE validated**

---

## Intermediate tests remaining (in order)

### Downstream pipeline smoke test  ← next
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

5. Examine if quick SSQL protocol after pretraining can act as regularizing model for subsequent quantization.

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
