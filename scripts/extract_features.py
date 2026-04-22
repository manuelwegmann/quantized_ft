"""
Diagnostic: pass Merlin CTs through the pretrained CT-CLIP ViT and inspect
the resulting feature embeddings.

No augmentations — raw preprocessing only (resize, HU clip, crop/pad).
Intended to validate checkpoint loading, preprocessing, and forward pass
before any training is attempted.

Usage:
    python scripts/extract_features.py
    python scripts/extract_features.py --n_scans 10 --output_dir /tmp/features
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make quantized_ft root importable
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# CT-CLIP packages
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/transformer_maskgit")))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/CT_CLIP")))

from models.backbone import CTViTBackbone
from pretrain.dataset import (
    _nii_to_tensor,
    _normalize_name,
    _first_column,
)

import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CHECKPOINT   = "/home/nlr950/Dir/CT-CLIP/checkpoints/CT-CLIP_v2.pt"
DATA_FOLDER  = "/home/nlr950/Dir/CT-CLIP/data/merlin_data"
REPORTS_FILE = "/home/nlr950/Dir/CT-CLIP/data/reports_final.xlsx"
META_FILE    = "/home/nlr950/Dir/CT-CLIP/data/metadata.csv"


# ---------------------------------------------------------------------------

def load_valid_accessions(reports_file: str) -> set:
    suffix = Path(reports_file).suffix.lower()
    df = pd.read_excel(reports_file) if suffix in {".xlsx", ".xls"} else pd.read_csv(reports_file)
    id_col = _first_column(df, ["VolumeName", "study id", "StudyInstanceUID"])
    return {_normalize_name(v) for v in df[id_col] if _normalize_name(v) is not None}


def load_meta(meta_file: str) -> dict:
    df = pd.read_csv(meta_file)
    id_col = _first_column(df, ["VolumeName", "study id", "StudyInstanceUID"])
    if id_col is None:
        return {}
    return {
        _normalize_name(row[id_col]): row.to_dict()
        for _, row in df.iterrows()
        if _normalize_name(row[id_col]) is not None
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_scans",    type=int,  default=5,              help="Number of scans to process")
    parser.add_argument("--checkpoint", default=CHECKPOINT,                help="Path to CT-CLIP_v2.pt")
    parser.add_argument("--data_folder",default=DATA_FOLDER)
    parser.add_argument("--reports",    default=REPORTS_FILE)
    parser.add_argument("--meta",       default=META_FILE)
    parser.add_argument("--use_pre_vq", action="store_true",               help="Bypass VQ codebook (pre-VQ tokens)")
    parser.add_argument("--output_dir", default=str(_PROJECT_ROOT / "runs" / "feature_extraction"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Mode   : {'pre-VQ' if args.use_pre_vq else 'post-VQ (default)'}\n")

    # ---- Backbone ----
    print("Loading backbone...")
    backbone = CTViTBackbone(checkpoint_path=args.checkpoint, use_pre_vq=args.use_pre_vq)
    backbone.to(device)
    backbone.eval()

    # ---- Discover scans ----
    valid = load_valid_accessions(args.reports)
    meta  = load_meta(args.meta)

    nii_files = sorted(Path(args.data_folder).rglob("*.nii.gz"))
    candidates = [f for f in nii_files if _normalize_name(f.name) in valid]

    if not candidates:
        print("ERROR: no matching NIfTI files found. Check data_folder and reports paths.")
        sys.exit(1)

    selected = candidates[: args.n_scans]
    print(f"Found {len(candidates)} valid scans — processing {len(selected)}\n")

    # ---- Forward passes ----
    all_embeddings = {}

    header = f"{'Scan':<20}  {'Shape':>18}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'NaN':>4}"
    print(header)
    print("-" * len(header))

    for nii_path in selected:
        name = _normalize_name(nii_path.name)
        meta_row = meta.get(name, {})

        # Preprocessing (no augmentation)
        tensor = _nii_to_tensor(str(nii_path), meta_row)   # (1, D, H, W)
        tensor = tensor.to(device)

        with torch.no_grad():
            emb = backbone(tensor.unsqueeze(0))             # (1, 512)

        emb_np = emb.squeeze(0).cpu().float().numpy()      # (512,)
        all_embeddings[name] = emb_np

        has_nan = bool(np.isnan(emb_np).any())
        print(
            f"{name:<20}  {str(tuple(emb_np.shape)):>18}  "
            f"{emb_np.mean():>8.4f}  {emb_np.std():>8.4f}  "
            f"{emb_np.min():>8.4f}  {emb_np.max():>8.4f}  "
            f"{'YES' if has_nan else 'no':>4}"
        )

    # ---- Save ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "embeddings.npz"
    np.savez(out_path, **all_embeddings)

    print(f"\nSaved {len(all_embeddings)} embeddings → {out_path}")
    print(f"Embedding dim : {next(iter(all_embeddings.values())).shape[0]}")


if __name__ == "__main__":
    main()
