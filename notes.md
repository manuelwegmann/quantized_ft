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

### Downstream pipeline smoke test  ✓ done (job 1714, 2026-04-24)
- Pipeline runs end-to-end ✓
- Pretrained 0.6667 vs Random 0.8333 on 21/4/5 split — not meaningful at this n
- Two bugs fixed: `best_state=None` crash risk, misleading "per split" print

### Cache features for all 5097 scans  ✓ done (job 7125, 2026-04-27)
- Script: `scripts/cache_all_features.py`, sbatch: `scripts/cache_all_features_slurm.sbatch`
- Backbones: `pretrained` (CT-CLIP ViT), `random` (same arch, random init), `random_cnn`
- Features cached for all 5097 scans; took ~1h 38min on A100
- Output: `runs/feature_cache_full/{pretrained,random,random_cnn}/feats.pt`

### Pretrained vs random: feature quality learning curve  ✓ done (job 7036, 2026-04-27)
- Script: `scripts/run_learning_curve.py`, condition: atelectasis only
- N_train sweep [30, 100, 300, 1000, all=1360], 5 seeds, 1000 epochs
- Results saved to `runs/learning_curve/results.json`

  | N_train | ViT pretrained   | ViT random       | Δ pretrained |
  |---------|------------------|------------------|--------------|
  |      30 | 0.5723 ±0.0185   | 0.4967 ±0.0389   | +0.076       |
  |     100 | 0.5882 ±0.0437   | 0.4853 ±0.0301   | +0.103       |
  |     300 | 0.6376 ±0.0135   | 0.4946 ±0.0309   | +0.143       |
  |    1000 | 0.6690 ±0.0032   | 0.5542 ±0.0085   | +0.115       |
  |    1360 | 0.6749 ±0.0003   | 0.5791 ±0.0005   | +0.096       |

- Pretrained backbone consistently outperforms random init at all data scales;
  gap largest at moderate N (~300), shrinks slightly at full N (expected).

### Multi-condition linear probe evaluation  ✓ done (job 9610 → 9820, 2026-04-28)
- Script: `scripts/run_multi_condition.py`, sbatch: `scripts/run_multi_condition_slurm.sbatch`
- 6 conditions: atelectasis, surgically_absent_gallbladder, renal_cyst,
  pleural_effusion, cardiomegaly, gallstones
- 4 backbones: pretrained ViT (post-VQ), pretrained ViT (pre-VQ), random ViT, random CNN
- N_train sweep [100, 300, all], 5 seeds, 1000 epochs; 70/15/15 split per condition
- Output: `runs/multi_condition/results.json`

  **Cross-condition AUROC (N=all):**

  | Condition                      | ViT pretrained   | ViT random       | CNN random       |
  |-------------------------------|------------------|------------------|------------------|
  | atelectasis                   | **0.6749** ±0.000| 0.5688 ±0.002    | 0.6136 ±0.000    |
  | surgically_absent_gallbladder | **0.5858** ±0.000| 0.5352 ±0.000    | 0.5643 ±0.000    |
  | renal_cyst                    | **0.6351** ±0.001| 0.4664 ±0.001    | 0.4791 ±0.001    |
  | pleural_effusion              | 0.6700 ±0.000    | 0.5759 ±0.000    | **0.7193** ±0.001|
  | cardiomegaly                  | **0.7261** ±0.003| 0.5989 ±0.004    | 0.5384 ±0.005    |
  | gallstones                    | 0.5299 ±0.002    | 0.5015 ±0.007    | **0.5554** ±0.001|
  | **Macro mean**                | **0.637**        | 0.541            | 0.595            |

  **Key observations:**
  - Pretrained ViT is best on 4/6 conditions, but margins are modest (~0.1)
  - CNN random beats pretrained ViT on pleural_effusion (0.72 vs 0.67) and gallstones
  - ViT random is consistently worst — random CNN beats it on 4/6 conditions,
    suggesting CNN inductive biases (locality, translation equivariance) are more
    useful here than raw random ViT weights

  **Note:** The CNN and ViT random baselines are randomly initialized and kept fully
  frozen — they act as random projections of the input, not trained feature extractors.
  The CNN beating the pretrained ViT on pleural effusion and gallstones therefore means
  raw voxel statistics (projected randomly) outperform a chest-domain pretrained model
  on those conditions — a strong indicator of domain mismatch rather than CNN superiority.

  ---

  ### Q&A — interpretation of results

  **Q: Is it contradictory that a large domain-specific foundation model generalizes so
  poorly?**
  No. Foundation model generalization holds within the pretraining distribution. CT-CLIP
  was trained on chest CTs; Merlin is abdominal. Cross-anatomy transfer within medical
  imaging is well-documented to be limited — which is why region-specific models
  (BioViL-T, MedSAM) have emerged. Additionally, the VQ codebook (see below) compounds
  the problem: it discretizes features into a fixed vocabulary learned from chest CTs,
  so abdominal patterns map to whichever chest-domain entries are nearest.

  **Q: If large models don't improve significantly, is training small models from scratch
  more resource-efficient?**
  For a *mismatched* foundation model, yes — the tradeoff is unfavorable. A randomly
  initialized, frozen CNN already reaches 0.61–0.72 AUROC on some conditions. The
  pretrained ViT adds only ~0.05–0.12 AUROC over random ViT at full N, at much higher
  inference cost. The foundation model advantage is strongest in the low-data regime and
  when the model domain matches the target. The SimSiam pretraining experiments here test
  exactly this: self-supervised pretraining on Merlin itself (the matched domain) should
  produce much stronger transfer than the chest-pretrained baseline.

  **Q: Would LoRA-adapting CT-CLIP outperform a task-specific CNN trained from scratch?**
  Probably yes, but with a key caveat: LoRA adapts the transformer attention weights, not
  the VQ codebook. If the codebook is the primary bottleneck (which the results suggest),
  adapting attention over an already-wrong discrete representation gives limited gain.
  Full adaptation including the codebook — or extracting features before VQ — would
  likely outperform a task-specific CNN by leveraging genuine CT-domain priors (HU
  semantics, acquisition noise, low-level textures) that transfer regardless of body
  region.

  **Implication for the main experiment:** This motivates pretraining on Merlin itself
  (FP vs SSQL SimSiam). Even unlabelled self-supervised pretraining on abdominal CTs
  should yield far more transferable features than the chest-pretrained CT-CLIP backbone.

---

### Decision: use pre-VQ features throughout all future experiments

The multi-condition results showed pre-VQ features are better than post-VQ on most
conditions (e.g. atelectasis +0.034, pleural effusion +0.067 at full N). Two reasons:

1. **VQ domain mismatch**: The codebook was learned on chest CTs. For abdominal CTs,
   tokens are discretised into whichever chest-CT entries are nearest — a lossy,
   domain-wrong projection.
2. **VQ interacts badly with quantization**: The VQ is a hard nearest-neighbour lookup
   (non-smooth). Quantization noise in transformer activations can push tokens across
   Voronoi boundaries, causing discrete jumps in the final representation. Pre-VQ
   removes this interaction and gives a cleaner measurement of quantization's effect
   on the transformer representations alone.

**All subsequent experiments (quantization baselines and SimSiam pretraining) use
`use_pre_vq=True`.** Both `configs/pretrain_fp.yaml` and `configs/pretrain_ssql.yaml`
should be updated to reflect this before pretraining is launched.

---

### Quantization baseline — pre-VQ features under PTQ  ← in progress

Goal: establish how much performance degrades when the pretrained CT-CLIP backbone
is quantized at inference time (post-training quantization, no QAT). This sets the
baseline that FP and SSQL SimSiam pretraining will later be measured against.

**Method:** fake quantization (simulated/QAT-style) via `quantized_forward` context
manager. Weights and activations of all `nn.Linear` layers are constrained to an
n-bit uniform grid (dynamic per-tensor asymmetric min-max) for each forward pass;
arithmetic runs in float32. The VQ codebook is bypassed (`use_pre_vq=True`).
Feature vectors saved to disk are float32 (information loss is baked into the values).

**Configs evaluated:** (w_bits, a_bits) ∈ {(8,8), (4,8), (4,4), (2,4)}

| Config | Status |
|--------|--------|
| pre-VQ FP (unquantized) | ✓ cached (`pretrained_pre_vq/feats.pt`) |
| W8A8 | ✓ cached (job 818) |
| W4A8 | ✓ cached (job 818) |
| W4A4 | ⏳ pending resubmission |
| W2A4 | ⏳ pending resubmission |

Scripts:
- `scripts/cache_quantized_features.py` / `cache_quantized_features_slurm.sbatch`
- `scripts/run_quant_probe_slurm.sbatch` — runs `run_multi_condition.py` with
  `--backbones pretrained_pre_vq,...` and `--output_dir runs/quant_probe`

Note: `run_multi_condition.py` now accepts `--backbones` (custom backbone list)
and `--output_dir` to avoid overwriting `runs/multi_condition/results.json`.

---

## Full experiments  ← next

**Before submitting pretraining:** update both configs to `use_pre_vq: true`
(currently set to `false`).

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
