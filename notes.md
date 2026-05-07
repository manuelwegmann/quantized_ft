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

### Notes

Check that everything went well with quantized inference.

---

### Quantization baseline — pre-VQ features under PTQ  ✓ done

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
| W4A4 | ✓ cached (job 1203) |
| W2A4 | ✓ cached (job 1203) |

Scripts:
- `scripts/cache_quantized_features.py` / `cache_quantized_features_slurm.sbatch`
- `scripts/run_quant_probe_slurm.sbatch` — runs `run_multi_condition.py` with
  `--backbones pretrained_pre_vq,...` and `--output_dir runs/quant_probe`

Note: `run_multi_condition.py` now accepts `--backbones` (custom backbone list)
and `--output_dir` to avoid overwriting `runs/multi_condition/results.json`.

---

### Mini experiment — FP vs SSQL SimSiam (n=300, 40 epochs)  ✓ done (job 3213, 2026-05-01)

Script: `scripts/run_mini_experiment.py`, sbatch: `scripts/run_mini_experiment_slurm.sbatch`
Config: n_pretrain=300, epochs=40, norm=ln, lr=3e-4 (cosine LR not yet active), quant=W4A4
Output: `runs/mini_experiment_ln/`

**Job 1311 (2026-04-29) — FAILED:** BN + lr=0.05 → collapse (loss=0.0000 epochs 2–40). Killed by OOM.
**Job 3213 (2026-04-30 → 2026-05-01) — PASSED:** LN + lr=3e-4. No collapse.

**Training curves:**
- FP loss: -0.93 → **-1.99** (saturates epoch 3–5; theoretical max for 2-direction sum)
- SSQL loss: -2.45 → **-3.99** (saturates epoch 3–5; 2× FP because L_ssql + L_fp summed)
- Both plateau near-instantly → projector/predictor snap into place on pretrained features

**Linear probe — macro-mean AUROC (N=all, 6 conditions):**

| Backbone        | AUROC  | W4A4 drop | Retention |
|-----------------|--------|-----------|-----------|
| Original CT-CLIP (no SimSiam, ref) | 0.653 | 12.2 pp | 81.3% |
| FP SimSiam      | 0.641  | 13.3 pp   | 79.2%     |
| SSQL SimSiam    | 0.632  | **9.4 pp**| **85.1%** |

**Key finding:** SSQL pretraining improves quantization robustness even with only 300 scans
(9.4 pp drop vs 12.2 pp for original CT-CLIP). FP SimSiam makes it slightly worse.
Absolute FP accuracy slightly regresses — backbone barely adapts on 300 scans.

**Diagnosis — backbone not adapting:**
- Loss saturates by epoch 3 → projector/predictor converge → backbone gradient signal
  drops to near zero
- 300 scans insufficient to maintain training pressure on the backbone
- Constant LR provides no mechanism to overcome the plateau

**Fixes applied to `pretrain/trainer.py` (2026-05-01):**
1. **Cosine LR schedule**: `CosineAnnealingLR(T_max=epochs, eta_min=0)`, default for all new runs.
   Override with `config["training"]["lr_schedule"] = "constant"` to disable.
2. **Backbone gradient norm logging**: per-epoch `backbone_grad=X.XXe-XX` in training output.
   Confirms whether backbone is receiving signal each epoch.

**Diagnostic script added (2026-05-01):**
`scripts/diagnose_pretraining.py` — loads saved checkpoints and computes per-epoch:
uniformity, effective rank, kNN macro-AUROC. Run on any pretrain output dir:
```bash
python scripts/diagnose_pretraining.py \
    --checkpoint_dirs runs/mini_experiment_ln/pretrain_fp \
                      runs/mini_experiment_ln/pretrain_ssql
```
For per-epoch curves in future runs: add `--save_every 5` to `run_mini_experiment.py`.

---

### Diagnostic results — mini experiment (job 3894, 2026-05-01)

Script: `scripts/diagnose_pretraining.py` on `runs/mini_experiment_ln/pretrain_{fp,ssql}/`
Eval: 500 scans (seed 42), k=20, 80/20 kNN split, 6 conditions.

| Run | Uniformity (ep0→40) | Eff. rank (ep0→40) | kNN AUROC (ep0→40) |
|-----|--------------------|--------------------|---------------------|
| FP  | −0.544 → −0.076    | 104.2 → 86.4 (−18) | 0.569 → 0.505 (−0.065) |
| SSQL| −0.544 → −0.023    | 104.2 → 78.0 (−26) | 0.569 → 0.525 (−0.045) |

**Metric definitions:**
- **Uniformity** `log mean exp(−2‖zᵢ−zⱼ‖²)`, L2-normalised features. Range (−∞, 0].
  More negative = better spread on the unit sphere; → 0 means collapse.
  Typical healthy SSL: −2 to −3. Complete collapse: 0.
- **Effective rank** `exp(H(σ))`, H = Shannon entropy of normalised singular-value spectrum.
  Range [1, D=512]. High = many dimensions used; → 1 = dimensional collapse.
  Baseline CT-CLIP: 104 (typical for task-specific pretrained ViT).
- **kNN macro-AUROC** cosine kNN (k=20), macro-averaged over 6 conditions.
  Not directly comparable to linear probe AUROC (no trained head, smaller eval set).

**Interpretation:**
All three metrics move in the collapse direction for both runs.  The uniformity near 0
and rank drop are consistent with projector/predictor collapse (loss saturated at −2.0 by
epoch 3) slowly pulling backbone features toward a lower-dimensional, less-uniform manifold.
The backbone itself only drifted 2.67% (FP) / 2.67% (SSQL) in L2 norm from CT-CLIP, so the
backbone is degraded but not destroyed — linear probing still recovers 0.641 / 0.632 AUROC.

SSQL anomaly: worse geometry (uniformity −0.023 vs −0.076, rank 78 vs 86) but better kNN
AUROC (0.525 vs 0.505). Quantization noise may be concentrating the remaining discriminative
signal into fewer, more robust dimensions — consistent with the better W4A4 retention
(9.4 pp vs 13.3 pp drop) seen in the linear probe.

**Note on diagnostic message:** `[backbone] no checkpoint — using random initialisation`
appears for every non-epoch-0 checkpoint. This is a red herring: the `CTViTBackbone`
constructor prints it when `checkpoint_path=None`, but `load_state_dict` immediately
overwrites those weights with the SimSiam checkpoint. Verified: checkpoint contains 158
backbone keys, epoch=40, weights differ from CT-CLIP baseline.

---

### Weekend experiments (submitted 2026-05-01, results 2026-05-02)

All jobs used updated `run_mini_experiment_slurm.sbatch` with cosine LR.

| Job | Config | Node | Status | Notes |
|-----|--------|------|--------|-------|
| A (3887) | pretrain_fp, N=1000, bs=2, LN | gpu14 | ✓ Done | Backbone grad non-zero all 40 epochs |
| B (3888) | pretrain_ssql, N=1000, bs=2, LN | gpu02 | ✗ OOM ep1 | |
| D (3889) | all, N=300, bs=8, BN | gpu02 | ✗ OOM ep1 | |
| E (3891) | all, N=300, bs=16, BN | gpu02 | ✗ OOM      | |

**Job A key result:** backbone grad ~1.2–1.3e-02 throughout all 40 epochs (vs near-zero
after epoch 3 in N=300 run). Cosine LR is working. Feature cache extracted.

**OOM root cause:** SSQL + aux loss runs 4 backbone forward passes simultaneously
(2 FP + 2 quantized, all with full activation graphs), requiring ~2× FP memory.
FP at bs=2 uses ~39 GB on a 40 GB A100; SSQL genuinely doesn't fit.
BN at bs=8 also barely OOMed (38.96/39.49 GB). All jobs landed on gpu02 which had
less headroom than gpu14 (where job A succeeded), compounding the issue.

**Why batch size scaling is disproportionately expensive for this model:**
The bottleneck is not the input CT volume but the self-attention mechanism in the
spatial transformer. Attention materialises a `(B×t, heads, T, T)` matrix where:
- T = (480/20)² = 576 spatial tokens per frame (patch_size=20, image_size=480)
- t = 240/10 = 24 temporal steps (temporal_patch_size=10)
- heads = 8

Memory for this tensor scales as O(B × T²). At bs=2: ~510 MB per attention layer.
At bs=8: ~2.04 GB — confirmed by the job 7786 crash which failed trying to allocate
exactly 1.90 GiB for the `einsum('b h i d, b h j d -> b h i j', q, k)` call.
This is on top of SimSiam's 2× (or SSQL's 4×) multiplier for multiple forward passes
with full gradient graphs retained for backprop.

**Fix if large batch sizes are ever required:** replace standard attention with
Flash Attention (`F.scaled_dot_product_attention`), which computes the same result
in O(T) memory by fusing the softmax and never materialising the full T×T matrix.

**Fix:** `run_mini_experiment_slurm.sbatch` updated to request `gpu:h100:1` on
`hendrixgpu16fl` (H100, 80 GB) as default. FP probe-only jobs can override with
`--gres=gpu:a100:1 --nodelist=`.

**Re-submit commands:**
```bash
# Job B (SSQL pretrain, N=1000)
sbatch --export=ALL,PHASE=pretrain_ssql,N_PRETRAIN=1000,EPOCHS=40,SAVE_EVERY=5,OUTPUT_DIR=runs/exp_ln_1000 \
       scripts/run_mini_experiment_slurm.sbatch

# Job C (probe — after B finishes; FP features from job A already done)
sbatch --gres=gpu:a100:1 --nodelist= --exclude=hendrixgpu05fl,hendrixgpu06fl \
       --export=ALL,PHASE=probe,OUTPUT_DIR=runs/exp_ln_1000 \
       scripts/run_mini_experiment_slurm.sbatch

# Job D (BN feasibility, bs=8)
sbatch --export=ALL,PHASE=all,N_PRETRAIN=300,EPOCHS=40,BATCH_SIZE=8,NORM=bn,SAVE_EVERY=5,OUTPUT_DIR=runs/exp_bn_bs8 \
       scripts/run_mini_experiment_slurm.sbatch

# Job E (BN feasibility, bs=16)
sbatch --export=ALL,PHASE=all,N_PRETRAIN=300,EPOCHS=40,BATCH_SIZE=16,NORM=bn,SAVE_EVERY=5,OUTPUT_DIR=runs/exp_bn_bs16 \
       scripts/run_mini_experiment_slurm.sbatch
```

---

### Anti-collapse interventions

If N=1000 + cosine LR is insufficient to keep the backbone training healthily, the
following interventions are available, roughly in priority order for our failure mode
(projector snapping at epoch 3, backbone barely adapting):

**1. Stronger augmentations** *(highest impact)*
The projector snaps because the pretext task is too easy — weak augmentations produce
nearly identical views, so the projector trivially collapses onto a simple invariance.
Harder views force the backbone to do real work throughout training.
CT-specific options: more aggressive random cropping, intensity noise (scanner simulation),
random flipping/rotation, simulated slice-thickness variation.
Constraint: spatial and intensity transforms are valid; photometric transforms (colour
jitter, grayscale) are not applicable to CT Hounsfield units.

**2. Separate learning rates for projector/predictor vs. backbone**
Use lower LR for the projector/predictor (slows snap, extends useful backbone gradient
signal) while keeping or slightly raising backbone LR. Implement via two parameter groups
in the optimizer. The goal is to delay projector convergence, not to freeze the backbone.

**3. Gradient accumulation**
Current effective batch size = 2. Accumulating over k steps gives effective batch = 2k
with zero memory cost. Larger effective batch → better-estimated gradients per update,
less noisy backbone signal, and BN becomes viable at k≥4.

**4. VICReg-style variance regularization**
Add `max(0, γ − std(zᵢ))` over each embedding dimension as an auxiliary term on the
projector outputs. Directly penalises dimensional collapse without switching loss functions.
Complement to SimSiam loss, not a replacement.

**5. Momentum predictor (EMA)**
Replace predictor weights with an EMA: `θ ← α·θ + (1−α)·θ_current`. Prevents the
predictor from snapping to a trivial fixed point by always chasing a lagged version of
itself. Analogous to BYOL's momentum encoder. Pairs well with separate LRs (#2).

**6. Reduce projector capacity**
A more expressive projector collapses faster. Halving the projector hidden dim or
removing a layer makes the pretext task harder to shortcut.

**7. Freeze backbone for first K epochs (progressive unfreezing)**
Let projector/predictor reach a reasonable non-trivial state on frozen features, then
unfreeze backbone with low LR. Backbone gradients are more informative once the
projector has a useful (non-collapsed) objective.

---

## Collapse diagnostic experiments

The diagnostic data (jobs 3894, 7870) shows the bulk of feature geometry
deterioration happens in epochs 0–5 when backbone gradients are large (~0.39
at epoch 1), then plateaus. The mechanism is inferred but not yet directly
tested. The following experiments form a decision tree to isolate the cause.

**What we know (measured):**
- Backbone gradients large at epoch 1 (~0.39), fall rapidly, small after epoch 5
- Uniformity, effective rank, kNN AUROC all degrade almost entirely in epochs 0–5
- Features barely change after epoch 5 (consistent with near-zero gradients)

**What we are inferring (not yet measured):**
- Whether it is the gradient *direction* that is harmful, or just the magnitude
- Whether the projector collapses first and drags the backbone, or vice versa
- Whether weak augmentations (easy pretext task) are the primary driver

---

### Experiment 1 — Freeze backbone for first K epochs, then unfreeze

**Implementation:** call `backbone.freeze()` for the first K epochs in
`pretrain/trainer.py`, then `backbone.unfreeze()`. Run with K=5 and K=10.
Run the diagnostic script on saved checkpoints afterwards.

**What it tests:** whether the large early gradients are the direct cause of
collapse. If harmful gradients in epochs 0–5 drive the deterioration, freezing
should preserve feature geometry. Unfreezing onto a settled projector should
then produce healthy, informative gradients.

**Decision outcomes:**
- Features preserved after freeze, healthy gradients after unfreeze →
  early gradient signal is the culprit; delayed backbone training is a fix
- Features still collapse after unfreeze → projector/predictor state at
  epoch K is already degenerate regardless; problem is elsewhere

---

### Experiment 2 — Track projector output metrics alongside backbone

**Implementation:** extend `scripts/diagnose_pretraining.py` to also compute
uniformity and effective rank of the projector output z (extracted by running
backbone + projector, stopping before the predictor) at each checkpoint.
Plot both backbone and projector metrics on the same axes.

**What it tests:** whether the projector collapses first and pulls the backbone
with it, or whether the backbone loses diversity independently.

**Decision outcomes:**
- Projector output collapses before backbone → projector is learning to map
  diverse inputs to a degenerate subspace and dragging the backbone along;
  slowing the projector (experiment 3) is the right lever
- Backbone collapses first or simultaneously → backbone is directly learning
  augmentation invariance; stronger augmentations (experiment 4) is the lever

---

### Experiment 3 — Separate LRs: slow projector/predictor, keep backbone LR

**Implementation:** split the optimizer into two parameter groups —
backbone at the standard LR (3.9e-4), projector + predictor at a much lower
LR (e.g. 1e-5). Everything else unchanged.

**What it tests:** whether projector convergence speed is the controlling
variable. If slowing the projector delays the snap and keeps backbone gradients
informative for longer, this confirms the "fast projector, harmful backbone
gradients" hypothesis.

**Decision outcomes:**
- Snap delayed, backbone gradients sustained, better diagnostic metrics →
  projector speed is the lever; separate LRs are a viable fix
- Snap still happens at the same epoch → projector LR is not the bottleneck;
  augmentation difficulty or a different mechanism is driving collapse

---

### Experiment 4 — Stronger augmentations

**Implementation:** increase augmentation intensity in
`pretrain/augmentations.py` — more aggressive random cropping, stronger
intensity noise, larger rotation range. Run the same N=1000 setup and compare
diagnostic curves against the baseline.

**What it tests:** whether the pretext task being too easy (weak augmentations
producing near-identical views) allows the projector to trivially collapse.
If harder views require the backbone to preserve discriminative structure to
solve the task, collapse should be delayed or prevented.

**Decision outcomes:**
- Collapse delayed or absent, backbone gradients sustained → augmentation
  difficulty is the primary driver; this is the highest-leverage practical fix
- Collapse timing unchanged → augmentations are not the bottleneck; the
  problem is intrinsic to the small-batch / pretrained-backbone dynamics

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
