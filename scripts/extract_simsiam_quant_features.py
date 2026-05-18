"""
Extract quantized features for already-trained SimSiam checkpoints.

Loads backbone weights from checkpoint_final.pt in each pretrain directory,
applies the requested quantization, and saves feats.pt into the experiment's
feature_cache. Existing cache entries are skipped automatically.

The probe phase of run_mini_experiment.py auto-discovers all subdirs in
feature_cache, so running --phase probe afterwards picks up the new entries.

Usage:
    python scripts/extract_simsiam_quant_features.py \\
        --output_dir runs/mini_experiment_ln \\
        --quant_w 8 --quant_a 8
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import paths  # noqa: E402 — registers CT-CLIP packages on sys.path

from models.backbone import CTViTBackbone
from models.quantization import quantized_forward
from cache_all_features import AllScansDataset

from paths import CHECKPOINT, DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE


def _load_backbone(pretrain_dir: Path) -> CTViTBackbone:
    ckpt_path = pretrain_dir / "checkpoint_final.pt"
    if not ckpt_path.exists():
        ckpt_path = pretrain_dir / "checkpoint_latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = CTViTBackbone(checkpoint_path=CHECKPOINT, use_pre_vq=True)
    backbone.load_state_dict(ckpt["backbone"])
    backbone.freeze()
    print(f"  Loaded backbone from {ckpt_path.name}  (epoch {ckpt.get('epoch', '?')})")
    return backbone


def _extract(backbone, loader, device, w_bits, a_bits):
    backbone.eval()
    feats = []
    for x, _ in tqdm.tqdm(loader, unit="batch", leave=False):
        with torch.no_grad():
            with quantized_forward([backbone], w_bits, a_bits):
                feats.append(backbone(x.to(device)).cpu())
    return torch.cat(feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/mini_experiment_ln")
    parser.add_argument("--quant_w",    type=int, default=8)
    parser.add_argument("--quant_a",    type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers",type=int, default=2)
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out      = _PROJECT_ROOT / args.output_dir
    feat_dir = out / "feature_cache"
    w, a     = args.quant_w, args.quant_a

    print(f"=== Extract W{w}A{a} features  output={out} ===")
    print(f"Device: {device}" + (
        f"  GPU: {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    ds = AllScansDataset(DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"Eval dataset: {len(ds)} scans")

    for prefix in ("fp", "ssql"):
        name    = f"{prefix}_w{w}a{a}"
        out_dir = feat_dir / name
        out_pt  = out_dir / "feats.pt"

        if out_pt.exists():
            print(f"\nSkipping {name} — feats.pt already exists")
            continue

        pretrain_dir = out / f"pretrain_{prefix}"
        if not pretrain_dir.exists():
            print(f"\nSkipping {name} — pretrain dir not found: {pretrain_dir}")
            continue

        print(f"\n{'='*60}\n  Extracting {name}\n{'='*60}")
        backbone = _load_backbone(pretrain_dir)
        backbone.to(device)

        feats = _extract(backbone, loader, device, w, a)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(feats, out_pt)
        print(f"  Saved {feats.shape}  →  {out_pt}")

        backbone.cpu()
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone. Run probe phase next:")
    print(f"  sbatch --export=ALL,PHASE=probe,OUTPUT_DIR={args.output_dir} "
          f"scripts/run_mini_experiment_slurm.sbatch")


if __name__ == "__main__":
    main()
