"""
Cache pre-VQ features for the pretrained CT-CLIP backbone under four
quantization configurations: (w_bits, a_bits) in [(8,8), (4,8), (4,4), (2,4)].

Requires the shared metadata written by cache_all_features.py (accessions.json,
label_names.json, labels.pt) — run that script first if the cache is missing.

Output:
    runs/feature_cache_full/pretrained_pre_vq_w8a8/feats.pt
    runs/feature_cache_full/pretrained_pre_vq_w4a8/feats.pt
    runs/feature_cache_full/pretrained_pre_vq_w4a4/feats.pt
    runs/feature_cache_full/pretrained_pre_vq_w2a4/feats.pt

Usage:
    python scripts/cache_quantized_features.py
"""

import sys
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/transformer_maskgit")))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/CT_CLIP")))
sys.path.insert(0, str(Path(__file__).parent))

from models.backbone import CTViTBackbone
from models.quantization import quantized_forward
from cache_all_features import AllScansDataset

CHECKPOINT   = "/home/nlr950/Dir/CT-CLIP/checkpoints/CT-CLIP_v2.pt"
DATA_FOLDER  = "/home/nlr950/Dir/CT-CLIP/data/merlin_data"
REPORTS_FILE = "/home/nlr950/Dir/CT-CLIP/data/reports_final.xlsx"
LABELS_FILE  = "/home/nlr950/Dir/CT-CLIP/data/zero_shot_findings_disease_cls.csv"
META_FILE    = "/home/nlr950/Dir/CT-CLIP/data/metadata.csv"
CACHE_DIR    = _PROJECT_ROOT / "runs" / "feature_cache_full"

QUANT_CONFIGS = [(8, 8), (4, 8), (4, 4), (2, 4)]  # (w_bits, a_bits)


def extract(model, loader, device, w_bits, a_bits):
    model.eval()
    feats = []
    for x, _ in tqdm.tqdm(loader, unit="batch"):
        with torch.no_grad():
            with quantized_forward([model], w_bits, a_bits):
                feats.append(model(x.to(device)).cpu())
    return torch.cat(feats)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (
        f"  GPU: {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    backbone = CTViTBackbone(checkpoint_path=CHECKPOINT, use_pre_vq=True)
    backbone.freeze()
    backbone.to(device)
    backbone.eval()

    print("\nIndexing scans...")
    ds = AllScansDataset(DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE)
    print(f"  {len(ds)} scans found")

    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        num_workers=8, pin_memory=True)

    for w_bits, a_bits in QUANT_CONFIGS:
        name    = f"pretrained_pre_vq_w{w_bits}a{a_bits}"
        out_dir = CACHE_DIR / name
        out_pt  = out_dir / "feats.pt"

        if out_pt.exists():
            print(f"\nSkipping {name} — feats.pt already exists")
            continue

        print(f"\n{'='*60}")
        print(f"  {name}  (weights={w_bits}-bit, activations={a_bits}-bit)")
        print(f"{'='*60}")
        out_dir.mkdir(parents=True, exist_ok=True)

        feats = extract(backbone, loader, device, w_bits, a_bits)
        torch.save(feats, out_pt)
        print(f"  Saved {feats.shape}  →  {out_pt}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone → {CACHE_DIR}")


if __name__ == "__main__":
    main()
