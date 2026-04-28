"""
Extract features for ALL on-disk scans (no label filter) for three backbones:
pretrained ViT, random ViT, random CNN.

Saves to runs/feature_cache_full/ in a flat format — no train/val/test split.
The split is applied per condition at probe time in run_multi_condition.py.

Output layout:
  runs/feature_cache_full/
    accessions.json          ordered accession IDs (shared across backbones)
    label_names.json         30 condition names in CSV column order
    labels.pt                [N, 30] float32  (-1 = not mentioned in CSV)
    pretrained/feats.pt      [N, 512]
    random/feats.pt          [N, 512]
    random_cnn/feats.pt      [N, 256]

Usage:
    python scripts/cache_all_features.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/transformer_maskgit")))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/CT_CLIP")))
sys.path.insert(0, str(Path(__file__).parent))

from pretrain.dataset import _nii_to_tensor, _normalize_name, _first_column
from models.backbone import CTViTBackbone
from cnn_baseline import RandomCNN3D, CNN_DIM

CHECKPOINT   = "/home/nlr950/Dir/CT-CLIP/checkpoints/CT-CLIP_v2.pt"
DATA_FOLDER  = "/home/nlr950/Dir/CT-CLIP/data/merlin_data"
REPORTS_FILE = "/home/nlr950/Dir/CT-CLIP/data/reports_final.xlsx"
LABELS_FILE  = "/home/nlr950/Dir/CT-CLIP/data/zero_shot_findings_disease_cls.csv"
META_FILE    = "/home/nlr950/Dir/CT-CLIP/data/metadata.csv"
CACHE_DIR    = _PROJECT_ROOT / "runs" / "feature_cache_full"
SEED         = 42


# ── dataset ───────────────────────────────────────────────────────────────────

class AllScansDataset(Dataset):
    """All on-disk scans that appear in the reports whitelist."""

    def __init__(self, data_folder, reports_file, labels_file, meta_file):
        valid_acc  = self._load_reports(reports_file)
        label_df, self.label_names = self._load_labels(labels_file)
        self.meta  = self._load_meta(meta_file)

        self.samples = []   # (path, label_array [30])
        for nii in tqdm.tqdm(sorted(Path(data_folder).rglob("*.nii.gz")),
                             desc="indexing scans"):
            name = _normalize_name(nii.name)
            if name not in valid_acc:
                continue
            labels = label_df.get(name,
                       np.full(len(self.label_names), -1, dtype=np.float32))
            self.accessions.append(name) if hasattr(self, "accessions") else None
            self.samples.append((str(nii), labels))

        self.accessions = [_normalize_name(Path(p).name) for p, _ in self.samples]

    def _load_reports(self, reports_file):
        suffix = Path(reports_file).suffix.lower()
        df = pd.read_excel(reports_file) if suffix in {".xlsx", ".xls"} else pd.read_csv(reports_file)
        id_col = _first_column(df, ["VolumeName", "study id", "StudyInstanceUID"])
        return {_normalize_name(v) for v in df[id_col] if _normalize_name(v)}

    def _load_labels(self, labels_file):
        df = pd.read_csv(labels_file)
        id_col   = _first_column(df, ["VolumeName", "study id", "StudyInstanceUID"])
        lbl_cols = [c for c in df.columns if c != id_col]
        lookup = {}
        for _, row in df.iterrows():
            name = _normalize_name(row[id_col])
            if name is None:
                continue
            lookup[name] = pd.to_numeric(row[lbl_cols], errors="coerce").to_numpy(dtype=np.float32)
        return lookup, lbl_cols

    def _load_meta(self, meta_file):
        if meta_file is None:
            return {}
        df = pd.read_csv(meta_file)
        id_col = _first_column(df, ["VolumeName", "study id", "StudyInstanceUID"])
        if id_col is None:
            return {}
        return {_normalize_name(row[id_col]): row.to_dict()
                for _, row in df.iterrows() if _normalize_name(row[id_col])}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, labels = self.samples[idx]
        meta_row = self.meta.get(_normalize_name(Path(path).name), {})
        return _nii_to_tensor(path, meta_row), torch.tensor(labels)


# ── extraction ────────────────────────────────────────────────────────────────

def extract(model: nn.Module, loader: DataLoader,
            device: torch.device) -> torch.Tensor:
    model.eval()
    feats = []
    for x, _ in tqdm.tqdm(loader, unit="batch"):
        with torch.no_grad():
            feats.append(model(x.to(device)).cpu())
    return torch.cat(feats)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size",  type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    print("\nIndexing scans...")
    ds = AllScansDataset(DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE)
    print(f"  {len(ds)} scans found")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # save shared metadata once
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "accessions.json", "w") as f:
        json.dump(ds.accessions, f)
    with open(CACHE_DIR / "label_names.json", "w") as f:
        json.dump(ds.label_names, f)

    all_labels = torch.tensor(
        np.stack([lbl for _, lbl in ds.samples]), dtype=torch.float32
    )
    torch.save(all_labels, CACHE_DIR / "labels.pt")
    print(f"  Saved accessions, label_names, labels [{all_labels.shape}]")

    # backbone definitions: (name, constructor)
    # pretrained_pre_vq: same weights as pretrained but features extracted
    # before the VQ codebook step — used to isolate whether the codebook
    # is the primary bottleneck for cross-domain transfer.
    torch.manual_seed(SEED)
    backbones = [
        ("pretrained",        CTViTBackbone(checkpoint_path=CHECKPOINT)),
        ("pretrained_pre_vq", CTViTBackbone(checkpoint_path=CHECKPOINT, use_pre_vq=True)),
        ("random",            CTViTBackbone(checkpoint_path=None)),
        ("random_cnn",        RandomCNN3D()),
    ]

    for name, model in backbones:
        out_dir = CACHE_DIR / name
        if (out_dir / "feats.pt").exists():
            print(f"\nSkipping {name} — feats.pt already exists")
            del model
            continue

        print(f"\n{'='*60}\nBackbone: {name}\n{'='*60}")
        out_dir.mkdir(exist_ok=True)

        # freeze / eval
        if hasattr(model, "freeze"):
            model.freeze()
        else:
            for p in model.parameters():
                p.requires_grad_(False)
        model.to(device)

        feats = extract(model, loader, device)
        torch.save(feats, out_dir / "feats.pt")
        print(f"  Saved {name}/feats.pt  {feats.shape}")

        model.cpu()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone → {CACHE_DIR}")


if __name__ == "__main__":
    main()
