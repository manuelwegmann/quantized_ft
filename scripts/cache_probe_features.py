"""
Extract and cache 512-dim features for all atelectasis-labeled scans.

Runs both pretrained and random-init backbones once over the full labeled
dataset and saves per-split feature tensors to disk.  The learning curve
script loads from this cache — no CT reloading needed per experiment.

Estimated runtime on A100: ~4h per backbone → ~8h total.

Usage:
    python scripts/cache_probe_features.py
    python scripts/cache_probe_features.py --seed 0 --batch_size 4
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/transformer_maskgit")))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/CT_CLIP")))

from downstream.dataset import MerlinDataset
from models.backbone import CTViTBackbone

CHECKPOINT   = "/home/nlr950/Dir/CT-CLIP/checkpoints/CT-CLIP_v2.pt"
DATA_FOLDER  = "/home/nlr950/Dir/CT-CLIP/data/merlin_data"
REPORTS_FILE = "/home/nlr950/Dir/CT-CLIP/data/reports_final.xlsx"
LABELS_FILE  = "/home/nlr950/Dir/CT-CLIP/data/zero_shot_findings_disease_cls.csv"
META_FILE    = "/home/nlr950/Dir/CT-CLIP/data/metadata.csv"
CACHE_DIR    = _PROJECT_ROOT / "runs" / "feature_cache"


def build_loaders(seed: int, num_workers: int, batch_size: int) -> dict:
    loaders = {}
    for split in ("train", "val", "test"):
        ds = MerlinDataset(
            data_folder=DATA_FOLDER,
            reports_file=REPORTS_FILE,
            labels_file=LABELS_FILE,
            meta_file=META_FILE,
            label_cols=["atelectasis"],
            require_labeled=True,
            split=split,
            seed=seed,
        )
        pos = int(sum(l[0] for _, l in ds.samples))
        neg = len(ds) - pos
        print(f"  {split:5s}: {len(ds):5d} scans  pos={pos}  neg={neg}")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def extract_and_save(backbone: nn.Module, loaders: dict, device: torch.device,
                     out_dir: Path) -> None:
    backbone.eval()
    for split, loader in loaders.items():
        feats, labels = [], []
        for x, y in tqdm.tqdm(loader, desc=f"  {split}", unit="batch"):
            with torch.no_grad():
                feats.append(backbone(x.to(device)).cpu())
            labels.append(y)
        feats_t  = torch.cat(feats)
        labels_t = torch.cat(labels)
        torch.save(feats_t,  out_dir / f"{split}_feats.pt")
        torch.save(labels_t, out_dir / f"{split}_labels.pt")
        print(f"    saved {split}: {feats_t.shape}  labels {labels_t.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size",  type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"
    print(f"Device: {device}  GPU: {gpu}")

    print("\nBuilding dataset (atelectasis, explicit 0/1 only, seed={})...".format(args.seed))
    loaders = build_loaders(args.seed, args.num_workers, args.batch_size)

    for backbone_name, ckpt in [("pretrained", CHECKPOINT), ("random", None)]:
        print(f"\n{'='*60}")
        print(f"Backbone: {backbone_name}")
        print(f"{'='*60}")
        out_dir = CACHE_DIR / backbone_name
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(args.seed)
        backbone = CTViTBackbone(checkpoint_path=ckpt)
        backbone.freeze()
        backbone.to(device)

        extract_and_save(backbone, loaders, device, out_dir)

        backbone.cpu()
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    meta = {
        "seed": args.seed,
        "label": "atelectasis",
        "splits": ["train", "val", "test"],
        "backbones": ["pretrained", "random"],
    }
    with open(CACHE_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nCache complete → {CACHE_DIR}")


if __name__ == "__main__":
    main()
