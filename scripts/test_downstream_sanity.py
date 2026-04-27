"""
Sanity check: pretrained CT-CLIP ViT vs randomly initialised ViT as feature extractor.

Trains a linear probe for atelectasis binary classification under both conditions
and reports test AUROC side by side.  Only scans with explicit 0/1 atelectasis
labels are used (-1 = not mentioned is excluded).

Feature extraction is done once per backbone (frozen, no grad), then the linear
probe is trained purely on cached (512,) embeddings — no CT reloading per epoch.

Usage:
    python scripts/test_downstream_sanity.py
    python scripts/test_downstream_sanity.py --epochs 50 --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import tqdm

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/transformer_maskgit")))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/CT_CLIP")))

from downstream.dataset import MerlinDataset
from models.backbone import CTViTBackbone, EMBED_DIM

CHECKPOINT   = "/home/nlr950/Dir/CT-CLIP/checkpoints/CT-CLIP_v2.pt"
DATA_FOLDER  = "/home/nlr950/Dir/CT-CLIP/data/merlin_data"
REPORTS_FILE = "/home/nlr950/Dir/CT-CLIP/data/reports_final.xlsx"
LABELS_FILE  = "/home/nlr950/Dir/CT-CLIP/data/zero_shot_findings_disease_cls.csv"
META_FILE    = "/home/nlr950/Dir/CT-CLIP/data/metadata.csv"
OUTPUT_DIR   = str(_PROJECT_ROOT / "runs" / "sanity_check")


def make_loaders(seed: int, num_workers: int, batch_size: int) -> dict:
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
        print(f"  {split:5s}: {len(ds)} scans  "
              f"pos={int(sum(l[0] for _, l in ds.samples))}  "
              f"neg={int(sum(1 - l[0] for _, l in ds.samples))}")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def extract_features(backbone: nn.Module, loaders: dict, device: torch.device) -> dict:
    """Single frozen forward pass over all splits; returns {split: (feats, labels)}."""
    backbone.eval()
    result = {}
    for split, loader in loaders.items():
        feats, labels = [], []
        for x, y in tqdm.tqdm(loader, desc=f"    {split}", leave=False):
            with torch.no_grad():
                feats.append(backbone(x.to(device)).cpu())
            labels.append(y)
        result[split] = (torch.cat(feats), torch.cat(labels))  # (N, 512), (N, 1)
    return result


def train_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float = 1e-3,
    probe_seed: int = 0,
) -> tuple:
    torch.manual_seed(probe_seed)
    probe = nn.Linear(EMBED_DIM, 1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    feat_loader = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=64,
        shuffle=True,
    )

    best_val_auroc = -1.0
    best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss = 0.0
        for xb, yb in feat_loader:
            logits = probe(xb.to(device)).squeeze(1)
            loss = criterion(logits, yb[:, 0].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(val_feats.to(device)).squeeze(1)
            val_auroc = float(roc_auc_score(
                val_labels[:, 0].numpy(),
                torch.sigmoid(val_logits).cpu().numpy(),
            ))
            train_loss = total_loss / len(feat_loader)
            print(f"    epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_auroc={val_auroc:.4f}")

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    return probe, best_val_auroc


def test_auroc(
    probe: nn.Module,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
) -> float:
    probe.eval()
    with torch.no_grad():
        logits = probe(test_feats.to(device)).squeeze(1)
    probs = torch.sigmoid(logits).cpu().numpy()
    return float(roc_auc_score(test_labels[:, 0].numpy(), probs))


def run_condition(
    name: str,
    backbone: nn.Module,
    loaders: dict,
    device: torch.device,
    epochs: int,
    probe_seed: int,
) -> float:
    print(f"\n{'='*60}")
    print(f"Condition: {name}")
    print(f"{'='*60}")

    backbone.freeze()
    backbone.to(device)

    print("  Extracting features...")
    splits = extract_features(backbone, loaders, device)
    train_feats, train_labels = splits["train"]
    val_feats,   val_labels   = splits["val"]
    test_feats,  test_labels  = splits["test"]

    print(f"  Feature shape: {train_feats.shape[1]}-dim  "
          f"train={len(train_feats)}  val={len(val_feats)}  test={len(test_feats)}")

    print(f"  Training linear probe ({epochs} epochs)...")
    probe, best_val_auroc = train_probe(
        train_feats, train_labels,
        val_feats,   val_labels,
        device, epochs, probe_seed=probe_seed,
    )

    auroc = test_auroc(probe, test_feats, test_labels, device)
    print(f"  Best val AUROC : {best_val_auroc:.4f}")
    print(f"  Test AUROC     : {auroc:.4f}")

    backbone.cpu()
    return auroc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--seed",        type=int,   default=42, help="Dataset split seed")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--max_samples", type=int,   default=None,
                        help="Cap total scans per split (smoke test)")
    args = parser.parse_args()

    if args.max_samples is not None:
        import os
        os.environ["CT_CLIP_MAX_SAMPLES"] = str(args.max_samples)
        print(f"[smoke test] max_samples={args.max_samples} total (split before probe)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print("\nBuilding datasets (atelectasis only, explicit 0/1 labels)...")
    loaders = make_loaders(args.seed, args.num_workers, args.batch_size)

    results = {}

    backbone_pretrained = CTViTBackbone(checkpoint_path=CHECKPOINT)
    results["pretrained"] = run_condition(
        "pretrained", backbone_pretrained, loaders, device, args.epochs, probe_seed=args.seed
    )
    del backbone_pretrained
    if device.type == "cuda":
        torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    backbone_random = CTViTBackbone(checkpoint_path=None)
    results["random"] = run_condition(
        "random", backbone_random, loaders, device, args.epochs, probe_seed=args.seed
    )
    del backbone_random

    delta = results["pretrained"] - results["random"]
    sign  = "+" if delta >= 0 else ""

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY  (atelectasis, AUROC)")
    print(f"{'='*60}")
    print(f"  Pretrained : {results['pretrained']:.4f}")
    print(f"  Random     : {results['random']:.4f}")
    print(f"  Delta      : {sign}{delta:.4f}")
    print(f"{'='*60}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump({"pretrained": results["pretrained"], "random": results["random"],
                   "delta": delta, "epochs": args.epochs, "seed": args.seed}, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
