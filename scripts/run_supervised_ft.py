"""
Supervised fine-tuning of the CT-CLIP backbone.

Fine-tunes the pre-VQ backbone with a linear multi-label head on labeled
Merlin data. Uses masked BCE with per-condition positive-class weighting to
handle label sparsity and class imbalance, cosine LR decay, and optional
gradient accumulation for a larger effective batch size.

Conditions: atelectasis, pleural_effusion, renal_cyst,
            surgically_absent_gallbladder

Usage:
    python scripts/run_supervised_ft.py
    python scripts/run_supervised_ft.py \\
        --backbone_lr 1e-5 --head_lr 1e-4 --epochs 15 --accum_steps 1
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import paths  # noqa: E402

from models.backbone import CTViTBackbone
from cache_all_features import AllScansDataset
from paths import CHECKPOINT, DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE

CONDITIONS = [
    "atelectasis",
    "pleural_effusion",
    "renal_cyst",
    "surgically_absent_gallbladder",
]

FROZEN_PROBE_AUROC = 0.637  # macro-mean from multi-condition experiment


# ── loss ──────────────────────────────────────────────────────────────────────

def masked_bce(logits: torch.Tensor, labels: torch.Tensor,
               pos_weights: torch.Tensor) -> torch.Tensor:
    """
    BCE loss with per-condition positive-class weighting.
    Ignores entries where label == -1 (not assessed).
    pos_weights: (n_conditions,) tensor, weight applied to positive targets.
    """
    valid = labels != -1
    if not valid.any():
        return logits.sum() * 0.0
    pw = pos_weights.to(logits.device).unsqueeze(0).expand_as(logits)
    loss = F.binary_cross_entropy_with_logits(
        logits, labels.clamp(min=0), weight=None,
        pos_weight=pw, reduction="none",
    )
    return (loss * valid).sum() / valid.sum()


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(backbone, head, loader, n_cond, device):
    backbone.eval()
    head.eval()
    all_logits, all_labels = [], []
    for x, lbl in tqdm.tqdm(loader, desc="eval", leave=False):
        all_logits.append(head(backbone(x.to(device))).cpu())
        all_labels.append(lbl)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    aurocs = []
    for ci in range(n_cond):
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
        print(f"    {c:<35} {'n/a' if np.isnan(a) else f'{a:.4f}'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train",     type=int,   default=-1,
                        help="Training scans. -1 = use full 70%% split.")
    parser.add_argument("--epochs",      type=int,   default=15)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr",     type=float, default=1e-4)
    parser.add_argument("--accum_steps", type=int,   default=1,
                        help="Gradient accumulation steps. 1 = no accumulation.")
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--output_dir",  type=str,   default="runs/supervised_ft")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eff_bs = args.batch_size * args.accum_steps
    print("=== Supervised fine-tuning ===")
    print(f"  conditions    : {CONDITIONS}")
    print(f"  epochs        : {args.epochs}")
    print(f"  backbone_lr   : {args.backbone_lr:.1e}  head_lr: {args.head_lr:.1e}")
    print(f"  batch_size    : {args.batch_size}  accum_steps: {args.accum_steps}"
          f"  (effective bs={eff_bs})")
    print(f"  output        : {out}\n")

    # ── dataset ───────────────────────────────────────────────────────────
    print("Indexing scans...")
    full_ds = AllScansDataset(DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE)

    label_names       = full_ds.label_names
    condition_indices = [label_names.index(c) for c in CONDITIONS if c in label_names]
    conditions        = [c for c in CONDITIONS if c in label_names]
    n_cond            = len(conditions)

    # scans with at least one valid label for our conditions
    labeled = [
        i for i, (_, lbl) in enumerate(full_ds.samples)
        if any(lbl[ci] in (0.0, 1.0) for ci in condition_indices)
    ]
    print(f"  {len(full_ds)} total scans")
    print(f"  {len(labeled)} scans with at least 1 label for our conditions")

    # 70/15/15 split
    rng      = np.random.default_rng(42)
    perm     = rng.permutation(len(labeled))
    n_va     = int(0.15 * len(labeled))
    n_te     = int(0.15 * len(labeled))
    n_tr_all = len(labeled) - n_va - n_te

    tr_indices = [labeled[i] for i in perm[:n_tr_all]]
    va_indices = [labeled[i] for i in perm[n_tr_all:n_tr_all + n_va]]
    te_indices = [labeled[i] for i in perm[n_tr_all + n_va:]]

    if args.n_train > 0:
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

    def _sub(lbl):
        return lbl[:, condition_indices].to(device)

    # ── class weights from training labels ────────────────────────────────
    tr_labels = torch.tensor(
        np.stack([full_ds.samples[i][1][condition_indices] for i in tr_indices]),
        dtype=torch.float32,
    )
    pos_counts = (tr_labels == 1).sum(0).float().clamp(min=1)
    neg_counts = (tr_labels == 0).sum(0).float().clamp(min=1)
    pos_weights = neg_counts / pos_counts
    print("Class weights (neg/pos per condition):")
    for c, pw in zip(conditions, pos_weights.tolist()):
        print(f"  {c:<35} {pw:.3f}")
    print()

    # ── model ─────────────────────────────────────────────────────────────
    backbone = CTViTBackbone(checkpoint_path=str(CHECKPOINT), use_pre_vq=True).to(device)
    head     = nn.Linear(512, n_cond).to(device)
    nn.init.zeros_(head.bias)

    optimizer = torch.optim.Adam([
        {"params": backbone.parameters(), "lr": args.backbone_lr},
        {"params": head.parameters(),     "lr": args.head_lr},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    results = {}

    # ── training ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        backbone.train()
        head.train()
        total_loss, n_batches = 0.0, 0
        optimizer.zero_grad()

        for step, (x, lbl) in enumerate(
            tqdm.tqdm(tr_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        ):
            logits = head(backbone(x.to(device)))
            loss   = masked_bce(logits, _sub(lbl), pos_weights)
            loss   = loss / args.accum_steps
            loss.backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(tr_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.accum_steps
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        lr_bb    = optimizer.param_groups[0]["lr"]
        lr_hd    = optimizer.param_groups[1]["lr"]
        val_aurocs, val_macro = evaluate(backbone, head, va_loader, n_cond, device)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"lr_bb={lr_bb:.2e}  lr_hd={lr_hd:.2e}")
        _print_aurocs(conditions, val_aurocs, val_macro, ref=FROZEN_PROBE_AUROC)
        results[epoch] = {"loss": avg_loss, "aurocs": val_aurocs, "macro": val_macro}

    # ── test set ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Test set (final epoch):")
    te_aurocs, te_macro = evaluate(backbone, head, te_loader, n_cond, device)
    _print_aurocs(conditions, te_aurocs, te_macro, ref=FROZEN_PROBE_AUROC)

    # ── save ──────────────────────────────────────────────────────────────
    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "conditions":    conditions,
            "n_train":       len(tr_indices),
            "epochs":        args.epochs,
            "backbone_lr":   args.backbone_lr,
            "head_lr":       args.head_lr,
            "accum_steps":   args.accum_steps,
            "effective_bs":  eff_bs,
            "pos_weights":   pos_weights.tolist(),
            "frozen_baseline": FROZEN_PROBE_AUROC,
            "val":  {str(ep): m for ep, m in results.items()},
            "test": {"aurocs": te_aurocs, "macro": te_macro},
        }, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()