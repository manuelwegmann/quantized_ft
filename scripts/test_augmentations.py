"""
Augmentation diversity test.

For each scan:
  - Extract features from the raw (unaugmented) volume
  - Extract features from two independently augmented views
  - Report cosine similarities: raw vs view1, raw vs view2, view1 vs view2

Also computes between-scan similarity on raw volumes as a reference baseline.

A healthy result for SimSiam:
  - view1 vs view2 similarity should be noticeably below 1.0 (augmentations
    are creating real diversity) but not near 0 (representations still share
    meaningful structure).
  - within-scan (view1 vs view2) should be clearly higher than between-scan
    (different scans), confirming the model treats the same scan as more
    similar to itself than to other scans.

Usage:
    python scripts/test_augmentations.py
    python scripts/test_augmentations.py --n_scans 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/transformer_maskgit")))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/CT_CLIP")))

from models.backbone import CTViTBackbone
from pretrain.augmentations import CTAugmentation
from pretrain.dataset import _nii_to_tensor, _normalize_name, _first_column

import pandas as pd

CHECKPOINT   = "/home/nlr950/Dir/CT-CLIP/checkpoints/CT-CLIP_v2.pt"
DATA_FOLDER  = "/home/nlr950/Dir/CT-CLIP/data/merlin_data"
REPORTS_FILE = "/home/nlr950/Dir/CT-CLIP/data/reports_final.xlsx"
META_FILE    = "/home/nlr950/Dir/CT-CLIP/data/metadata.csv"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_valid_accessions(reports_file):
    suffix = Path(reports_file).suffix.lower()
    df = pd.read_excel(reports_file) if suffix in {".xlsx", ".xls"} else pd.read_csv(reports_file)
    id_col = _first_column(df, ["VolumeName", "study id", "StudyInstanceUID"])
    return {_normalize_name(v) for v in df[id_col] if _normalize_name(v) is not None}


def load_meta(meta_file):
    df = pd.read_csv(meta_file)
    id_col = _first_column(df, ["VolumeName", "study id", "StudyInstanceUID"])
    if id_col is None:
        return {}
    return {
        _normalize_name(row[id_col]): row.to_dict()
        for _, row in df.iterrows()
        if _normalize_name(row[id_col]) is not None
    }


def get_embedding(backbone, tensor, device):
    """tensor: (1, D, H, W) → embedding: (512,) numpy array."""
    with torch.no_grad():
        emb = backbone(tensor.unsqueeze(0).to(device))
    return emb.squeeze(0).cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_scans",    type=int, default=3)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--data_folder",default=DATA_FOLDER)
    parser.add_argument("--reports",    default=REPORTS_FILE)
    parser.add_argument("--meta",       default=META_FILE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    backbone = CTViTBackbone(checkpoint_path=args.checkpoint)
    backbone.to(device)
    backbone.eval()

    augment = CTAugmentation()

    valid = load_valid_accessions(args.reports)
    meta  = load_meta(args.meta)
    nii_files = sorted(Path(args.data_folder).rglob("*.nii.gz"))
    selected  = [f for f in nii_files if _normalize_name(f.name) in valid][: args.n_scans]
    print(f"Processing {len(selected)} scans\n")

    raw_embs  = {}
    view1_embs = {}
    view2_embs = {}

    for nii_path in selected:
        name     = _normalize_name(nii_path.name)
        meta_row = meta.get(name, {})
        raw      = _nii_to_tensor(str(nii_path), meta_row)   # (1, D, H, W)

        raw_embs[name]   = get_embedding(backbone, raw,                  device)
        view1_embs[name] = get_embedding(backbone, augment(raw.clone()), device)
        view2_embs[name] = get_embedding(backbone, augment(raw.clone()), device)

    # ------------------------------------------------------------------ #
    # Within-scan similarity (augmented views vs raw, and vs each other)  #
    # ------------------------------------------------------------------ #
    print("=== Within-scan cosine similarities ===")
    print(f"{'Scan':<20}  {'raw vs v1':>10}  {'raw vs v2':>10}  {'v1 vs v2':>10}")
    print("-" * 57)

    within_v1v2 = []
    for name in raw_embs:
        r_v1 = cosine_sim(raw_embs[name], view1_embs[name])
        r_v2 = cosine_sim(raw_embs[name], view2_embs[name])
        v1v2 = cosine_sim(view1_embs[name], view2_embs[name])
        within_v1v2.append(v1v2)
        print(f"{name:<20}  {r_v1:>10.4f}  {r_v2:>10.4f}  {v1v2:>10.4f}")

    print()

    # ------------------------------------------------------------------ #
    # Between-scan similarity (raw volumes, reference baseline)           #
    # ------------------------------------------------------------------ #
    names = list(raw_embs.keys())
    between = []
    print("=== Between-scan cosine similarities (raw, reference baseline) ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = cosine_sim(raw_embs[names[i]], raw_embs[names[j]])
            between.append(sim)
            print(f"  {names[i]} vs {names[j]}: {sim:.4f}")

    print()

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    print("=== Summary ===")
    print(f"  Mean within-scan  (v1 vs v2) : {np.mean(within_v1v2):.4f}")
    if between:
        print(f"  Mean between-scan (raw)      : {np.mean(between):.4f}")
        margin = np.mean(within_v1v2) - np.mean(between)
        print(f"  Margin (within - between)    : {margin:+.4f}")
        if margin > 0:
            print("  → within-scan similarity is HIGHER than between-scan  ✓")
        else:
            print("  → within-scan similarity is LOWER than between-scan  ✗ (unexpected)")

    print()
    print("  Interpretation guide:")
    print("    v1 vs v2 near 1.0  → augmentations too weak, views nearly identical")
    print("    v1 vs v2 near 0.0  → augmentations too strong, shared structure lost")
    print("    within > between   → model treats same scan as more similar to itself ✓")


if __name__ == "__main__":
    main()
