"""
Single training step diagnostic — FP and SSQL modes.

Loads one scan, produces two augmented views, and runs exactly one
forward + backward pass for each mode. Reports:
  - SimSiam loss value (must be finite and negative — cosine similarity
    of normalised vectors is in [-1, 1], so D(p, z) is in [-1, 1])
  - Whether gradients exist and are non-zero on backbone, projector,
    and predictor parameters
  - For SSQL: that backbone weights are exactly restored after the
    quantized_forward context exits

Usage:
    python scripts/test_single_step.py
    python scripts/test_single_step.py --mode fp
    python scripts/test_single_step.py --mode ssql
    python scripts/test_single_step.py --mode both   (default)
"""

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/transformer_maskgit")))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/CT_CLIP")))

from models.backbone import CTViTBackbone
from models.simsiam import Projector, Predictor
from models.quantization import quantized_forward, sample_bits
from pretrain.augmentations import CTAugmentation
from pretrain.dataset import _nii_to_tensor, _normalize_name, _first_column
from pretrain.loss import negative_cosine_similarity

import pandas as pd

CHECKPOINT   = "/home/nlr950/Dir/CT-CLIP/checkpoints/CT-CLIP_v2.pt"
DATA_FOLDER  = "/home/nlr950/Dir/CT-CLIP/data/merlin_data"
REPORTS_FILE = "/home/nlr950/Dir/CT-CLIP/data/reports_final.xlsx"
META_FILE    = "/home/nlr950/Dir/CT-CLIP/data/metadata.csv"


def load_one_scan(device):
    """Return two augmented views of the first valid scan as (1,1,D,H,W) tensors."""
    suffix = Path(REPORTS_FILE).suffix.lower()
    df = pd.read_excel(REPORTS_FILE) if suffix in {".xlsx", ".xls"} else pd.read_csv(REPORTS_FILE)
    id_col = _first_column(df, ["VolumeName", "study id", "StudyInstanceUID"])
    valid  = {_normalize_name(v) for v in df[id_col] if _normalize_name(v) is not None}

    meta_df  = pd.read_csv(META_FILE)
    meta_col = _first_column(meta_df, ["VolumeName", "study id", "StudyInstanceUID"])
    meta     = {_normalize_name(r[meta_col]): r.to_dict() for _, r in meta_df.iterrows()
                if _normalize_name(r[meta_col]) is not None} if meta_col else {}

    nii_files = sorted(Path(DATA_FOLDER).rglob("*.nii.gz"))
    for f in nii_files:
        name = _normalize_name(f.name)
        if name in valid:
            print(f"Using scan: {name}")
            raw      = _nii_to_tensor(str(f), meta.get(name, {}))
            augment  = CTAugmentation()
            x1 = augment(raw.clone()).unsqueeze(0).to(device)  # (1,1,D,H,W)
            x2 = augment(raw.clone()).unsqueeze(0).to(device)
            return x1, x2
    raise RuntimeError("No valid scan found.")


def check_gradients(label, *modules):
    """Print gradient stats for all parameters across the given modules."""
    print(f"\n  Gradient check — {label}")
    all_ok = True
    for mod in modules:
        mod_name = type(mod).__name__
        for name, p in mod.named_parameters():
            if p.grad is None:
                print(f"    {mod_name}.{name}: grad=None  ✗")
                all_ok = False
            else:
                gnorm = p.grad.norm().item()
                status = "✓" if gnorm > 0 else "✗ (zero)"
                print(f"    {mod_name}.{name}: grad_norm={gnorm:.6f}  {status}")
                if gnorm == 0:
                    all_ok = False
    return all_ok


def run_fp_step(backbone, projector, predictor, x1, x2):
    print("\n" + "="*55)
    print("MODE: Full-precision SimSiam")
    print("="*55)

    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(projector.parameters()) + list(predictor.parameters()),
        lr=0.05, momentum=0.9, weight_decay=1e-4,
    )
    optimizer.zero_grad()

    # Concatenate both views into a single batch of 2 so BatchNorm1d
    # has enough samples to compute statistics during training.
    x_batch = torch.cat([x1, x2], dim=0)           # (2, 1, D, H, W)
    z_batch = projector(backbone(x_batch))           # (2, 2048)
    z1, z2  = z_batch.chunk(2, dim=0)               # each (1, 2048)
    p_batch = predictor(z_batch)                     # (2, 2048)
    p1, p2  = p_batch.chunk(2, dim=0)

    loss = negative_cosine_similarity(p1, z2.detach()) + \
           negative_cosine_similarity(p2, z1.detach())

    print(f"\n  Loss = {loss.item():.6f}", end="  ")
    if torch.isfinite(loss):
        print("(finite ✓)")
    else:
        print("(NOT finite ✗)")
        return False

    loss.backward()

    grads_ok = check_gradients("FP", backbone, projector, predictor)
    optimizer.step()

    print(f"\n  FP step: {'PASSED ✓' if grads_ok else 'FAILED ✗'}")
    return grads_ok


def run_ssql_step(backbone, projector, predictor, x1, x2):
    print("\n" + "="*55)
    print("MODE: SSQL")
    print("="*55)

    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(projector.parameters()) + list(predictor.parameters()),
        lr=0.05, momentum=0.9, weight_decay=1e-4,
    )
    optimizer.zero_grad()

    # Both views batched together so BatchNorm1d sees 2 samples throughout
    x_batch = torch.cat([x1, x2], dim=0)           # (2, 1, D, H, W)

    # FP pass (targets)
    z_fp_batch = projector(backbone(x_batch))       # (2, 2048)
    z1_fp, z2_fp = z_fp_batch.chunk(2, dim=0)
    p_fp_batch = predictor(z_fp_batch)
    p1_fp, p2_fp = p_fp_batch.chunk(2, dim=0)

    # Snapshot a weight before quantized pass to verify restoration
    test_param = next(backbone.parameters())
    weight_before = test_param.data.clone()

    # Quantized pass
    w_bits, a_bits = sample_bits()
    print(f"\n  Sampled bits: w_bits={w_bits}, a_bits={a_bits}")
    with quantized_forward([backbone, projector], w_bits, a_bits):
        z_q_batch = projector(backbone(x_batch))    # (2, 2048)
    z1_q, z2_q = z_q_batch.chunk(2, dim=0)
    p_q_batch = predictor(z_q_batch)
    p1_q, p2_q = p_q_batch.chunk(2, dim=0)

    # Verify weight restoration
    weight_after = test_param.data
    weights_restored = torch.allclose(weight_before, weight_after)
    print(f"  Weights restored after quantized_forward: {weights_restored} {'✓' if weights_restored else '✗'}")

    L_ssql = negative_cosine_similarity(p1_q, z2_fp.detach()) + \
             negative_cosine_similarity(p2_q, z1_fp.detach())
    L_fp   = negative_cosine_similarity(p1_fp, z2_fp.detach()) + \
             negative_cosine_similarity(p2_fp, z1_fp.detach())
    loss   = L_ssql + L_fp

    print(f"\n  L_ssql = {L_ssql.item():.6f}")
    print(f"  L_fp   = {L_fp.item():.6f}")
    print(f"  Total  = {loss.item():.6f}", end="  ")
    if torch.isfinite(loss):
        print("(finite ✓)")
    else:
        print("(NOT finite ✗)")
        return False

    loss.backward()

    grads_ok = check_gradients("SSQL", backbone, projector, predictor)
    optimizer.step()

    passed = grads_ok and weights_restored
    print(f"\n  SSQL step: {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp", "ssql", "both"], default="both")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    x1, x2 = load_one_scan(device)
    print(f"View shapes: {tuple(x1.shape)}")

    results = {}

    if args.mode in ("fp", "both"):
        backbone  = CTViTBackbone(CHECKPOINT).to(device)
        projector = Projector().to(device)
        predictor = Predictor().to(device)
        results["fp"] = run_fp_step(backbone, projector, predictor, x1, x2)

    if args.mode in ("ssql", "both"):
        backbone  = CTViTBackbone(CHECKPOINT).to(device)
        projector = Projector().to(device)
        predictor = Predictor().to(device)
        results["ssql"] = run_ssql_step(backbone, projector, predictor, x1, x2)

    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    for mode, ok in results.items():
        print(f"  {mode.upper():5s}: {'PASSED ✓' if ok else 'FAILED ✗'}")


if __name__ == "__main__":
    main()
