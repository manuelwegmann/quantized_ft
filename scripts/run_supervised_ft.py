"""
Supervised fine-tuning diagnostic.

Fine-tunes the pre-VQ CT-CLIP backbone with a linear multi-label head on
labeled Merlin data to establish an upper bound on backbone improvability.

Interpretation:
  - If AUROC rises well above the frozen probe baseline (0.637 macro-mean),
    the backbone has representational headroom and SSL collapse is the
    primary obstacle to better SimSiam performance.
  - If AUROC barely moves, the backbone is already near its ceiling for
    this domain regardless of training method.

Usage:
    python scripts/run_supervised_ft.py
    python scripts/run_supervised_ft.py --n_train 300 --epochs 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import paths  # noqa: E402 — registers CT-CLIP packages on sys.path

from models.backbone import CTViTBackbone
from cache_all_features import AllScansDataset
from paths import CHECKPOINT, DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE

CONDITIONS = [
    "atelectasis",
    "surgically_absent_gallbladder",
    "renal_cyst",
    "pleural_effusion",
    "cardiomegaly",
    "gallstones",
]

# Frozen linear probe macro-mean AUROC from the multi-condition experiment.
FROZEN_PROBE_AUROC = 0.637


# ── loss ──────────────────────────────────────────────────────────────────────

def masked_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE loss that ignores -1 (unknown) entries."""
    valid = labels != -1
    if not valid.any():
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(
        logits, labels.clamp(min=0), reduction="none",
    )
    return (loss * valid).sum() / valid.sum()


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(backbone, head, loader, n_conditions, device):
    """Returns (per-condition AUROC list, macro mean)."""
    backbone.eval()
    head.eval()
    all_logits, all_labels = [], []
    for x, lbl in tqdm.tqdm(loader, desc="eval", leave=False):
        all_logits.append(head(backbone(x.to(device))).cpu())
        all_labels.append(lbl)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    aurocs = []
    for ci in range(n_conditions):
        y_true  = labels[:, ci]
        y_score = logits[:, ci]
        valid   = (y_true == 0) | (y_true == 1)
        if valid.sum() < 10 or y_true[valid].sum() == 0 or (1 - y_true[valid]).sum() == 0:
            aurocs.append(float("nan"))
            continue
        aurocs.append(roc_auc_score(y_true[valid], y_score[valid]))

    valid_aurocs = [a for a in aurocs if not np.isnan(a)]
    macro = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    return aurocs, macro


def _print_aurocs(conditions, aurocs, macro, ref=None):
    ref_str = f"  Δ vs frozen: {macro - ref:+.4f}" if ref is not None else ""
    print(f"  macro-AUROC = {macro:.4f}{ref_str}")
    for c, a in zip(conditions, aurocs):
        a_str = f"{a:.4f}" if not np.isnan(a) else "  n/a"
        print(f"    {c:<35} {a_str}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train",     type=int,   default=300,
                        help="Training scans (randomly sampled from labeled set).")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--backbone_lr", type=float, default=1e-4,
                        help="LR for pretrained backbone weights.")
    parser.add_argument("--head_lr",     type=float, default=1e-3,
                        help="LR for linear classification head.")
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--output_dir",  type=str,   default="runs/supervised_ft")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Supervised fine-tuning diagnostic ===")
    print(f"  n_train={args.n_train}  epochs={args.epochs}  "
          f"backbone_lr={args.backbone_lr:.1e}  head_lr={args.head_lr:.1e}")
    print(f"  reference (frozen probe): {FROZEN_PROBE_AUROC:.3f} macro-AUROC")
    print(f"  output → {out}\n")

    # ── dataset ───────────────────────────────────────────────────────────
    print("Indexing scans...")
    full_ds = AllScansDataset(DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE)

    label_names = full_ds.label_names
    condition_indices = [label_names.index(c) for c in CONDITIONS if c in label_names]
    conditions = [c for c in CONDITIONS if c in label_names]
    n_cond = len(conditions)
    print(f"  {len(full_ds)} total scans  |  {n_cond} conditions")

    # keep scans with at least one valid (0 or 1) label for our conditions
    labeled = [
        i for i, (_, lbl) in enumerate(full_ds.samples)
        if any(lbl[ci] in (0.0, 1.0) for ci in condition_indices)
    ]
    print(f"  {len(labeled)} scans have at least one valid label")

    # 70/15/15 split over labeled scans
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(labeled))
    n_va = int(0.15 * len(labeled))
    n_te = int(0.15 * len(labeled))
    n_tr_full = len(labeled) - n_va - n_te

    tr_indices = [labeled[i] for i in perm[:n_tr_full]]
    va_indices = [labeled[i] for i in perm[n_tr_full:n_tr_full + n_va]]
    te_indices = [labeled[i] for i in perm[n_tr_full + n_va:]]

    tr_indices = tr_indices[:args.n_train]
    print(f"  train={len(tr_indices)}  val={len(va_indices)}  test={len(te_indices)}\n")

    def _loader(indices, shuffle):
        return DataLoader(
            Subset(full_ds, indices),
            batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True,
        )

    tr_loader = _loader(tr_indices, shuffle=True)
    va_loader = _loader(va_indices, shuffle=False)
    te_loader = _loader(te_indices, shuffle=False)

    # Subset labels to our conditions for loss computation
    def _sub_labels(lbl):
        return lbl[:, condition_indices].to(device)

    # ── model ─────────────────────────────────────────────────────────────
    backbone = CTViTBackbone(checkpoint_path=str(CHECKPOINT), use_pre_vq=True).to(device)
    head     = nn.Linear(512, n_cond).to(device)
    nn.init.zeros_(head.bias)

    optimizer = torch.optim.Adam([
        {"params": backbone.parameters(), "lr": args.backbone_lr},
        {"params": head.parameters(),     "lr": args.head_lr},
    ])

    results = {}

    # ── training ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        backbone.train()
        head.train()
        total_loss, n_batches = 0.0, 0

        for x, lbl in tqdm.tqdm(tr_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            logits = head(backbone(x.to(device)))
            loss   = masked_bce(logits, _sub_labels(lbl))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_aurocs, val_macro = evaluate(backbone, head, va_loader, n_cond, device)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}")
        _print_aurocs(conditions, val_aurocs, val_macro, ref=FROZEN_PROBE_AUROC)
        results[epoch] = {"loss": avg_loss, "aurocs": val_aurocs, "macro": val_macro}

    # ── test set (final epoch) ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Test set (final epoch):")
    te_aurocs, te_macro = evaluate(backbone, head, te_loader, n_cond, device)
    _print_aurocs(conditions, te_aurocs, te_macro, ref=FROZEN_PROBE_AUROC)

    # ── save ──────────────────────────────────────────────────────────────
    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "conditions":      conditions,
            "n_train":         args.n_train,
            "epochs":          args.epochs,
            "backbone_lr":     args.backbone_lr,
            "head_lr":         args.head_lr,
            "frozen_baseline": FROZEN_PROBE_AUROC,
            "val":  {str(ep): m for ep, m in results.items()},
            "test": {"aurocs": te_aurocs, "macro": te_macro},
        }, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()