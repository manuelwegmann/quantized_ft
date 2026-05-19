"""
Separate backbone / projector-predictor LR sweep for SimSiam.

Hypothesis: the projector/predictor snap early because their LR is too high
relative to the backbone, causing backbone gradients to drop before the
backbone has adapted meaningfully. Slowing the projector (lower LR) while
keeping or raising the backbone LR should delay or prevent collapse.

Each condition runs FP SimSiam for --epochs epochs on --n_pretrain scans,
saving a checkpoint every epoch. Geometry diagnostics (uniformity, effective
rank, kNN AUROC) are computed inline after each condition.

Default conditions (backbone_lr:projector_lr):
  3e-4:3e-4   baseline (everything at the same LR used previously)
  3e-4:3e-5   slow projector 10x
  3e-4:3e-6   slow projector 100x
  3e-3:3e-5   fast backbone + slow projector

Usage:
    python scripts/run_lr_sweep.py
    python scripts/run_lr_sweep.py --n_pretrain 1500 --epochs 2 \\
        --lr_pairs "3e-4:3e-4,3e-4:3e-5,3e-4:3e-6,3e-3:3e-5"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import paths  # noqa: E402

from models.backbone import CTViTBackbone
from models.simsiam import Projector, Predictor
from pretrain.augmentations import CTAugmentation
from pretrain.dataset import CTPretrainDataset
from pretrain.loss import negative_cosine_similarity
from cache_all_features import AllScansDataset
import diagnose_pretraining as diag

from paths import CHECKPOINT, DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE


# ── training ──────────────────────────────────────────────────────────────────

def _backbone_grad_norm(backbone: nn.Module) -> float:
    sq = sum(
        p.grad.data.norm(2).item() ** 2
        for p in backbone.parameters() if p.grad is not None
    )
    return sq ** 0.5


def _save(backbone, projector, predictor, optimizer, epoch, path):
    torch.save({
        "epoch":     epoch,
        "backbone":  backbone.state_dict(),
        "projector": projector.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def train_condition(backbone, projector, predictor, loader,
                    backbone_lr, projector_lr, epochs, output_dir, device):
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.SGD([
        {"params": backbone.parameters(),
         "lr": backbone_lr},
        {"params": list(projector.parameters()) + list(predictor.parameters()),
         "lr": projector_lr},
    ], momentum=0.9, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    # Resume if a previous attempt was interrupted
    start_epoch = 0
    resume = output_dir / "checkpoint_latest.pt"
    if resume.exists():
        ckpt = torch.load(resume, map_location="cpu")
        backbone.load_state_dict(ckpt["backbone"])
        projector.load_state_dict(ckpt["projector"])
        predictor.load_state_dict(ckpt["predictor"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        for _ in range(start_epoch):
            scheduler.step()
        print(f"  resumed from epoch {start_epoch}")

    backbone.to(device)
    projector.to(device)
    predictor.to(device)

    last_ckpt = None
    for epoch in range(start_epoch, epochs):
        backbone.train()
        projector.train()
        predictor.train()

        total_loss, total_grad, n_batches = 0.0, 0.0, 0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)

            z1 = projector(backbone(x1))
            z2 = projector(backbone(x2))
            p1 = predictor(z1)
            p2 = predictor(z2)
            loss = (
                negative_cosine_similarity(p1, z2.detach())
                + negative_cosine_similarity(p2, z1.detach())
            )

            optimizer.zero_grad()
            loss.backward()
            total_grad  += _backbone_grad_norm(backbone)
            optimizer.step()

            total_loss  += loss.item()
            n_batches   += 1

        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        avg_grad = total_grad / max(n_batches, 1)
        lr_bb    = optimizer.param_groups[0]["lr"]
        lr_pp    = optimizer.param_groups[1]["lr"]
        print(f"  epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  "
              f"backbone_grad={avg_grad:.3e}  lr_bb={lr_bb:.2e}  lr_pp={lr_pp:.2e}")

        last_ckpt = {
            "epoch":     epoch + 1,
            "backbone":  backbone.state_dict(),
            "projector": projector.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        _save(backbone, projector, predictor, optimizer, epoch + 1,
              output_dir / "checkpoint_latest.pt")
        _save(backbone, projector, predictor, optimizer, epoch + 1,
              output_dir / f"checkpoint_ep{epoch + 1:04d}.pt")

    if last_ckpt is not None:
        torch.save(last_ckpt, output_dir / "checkpoint_final.pt")
    print(f"  done → {output_dir / 'checkpoint_final.pt'}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _lr_tag(bb_lr, pp_lr):
    return f"bb{bb_lr:.0e}_pp{pp_lr:.0e}"


def _build_eval_set(n_eval):
    full_ds = AllScansDataset(DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE)
    n = min(n_eval, len(full_ds))
    indices = np.sort(np.random.default_rng(42).choice(len(full_ds), n, replace=False))
    eval_ds = Subset(full_ds, indices.tolist())
    eval_loader = DataLoader(eval_ds, batch_size=4, shuffle=False,
                             num_workers=2, pin_memory=True)

    all_labels_full = torch.tensor(
        np.stack([lbl for _, lbl in full_ds.samples]), dtype=torch.float32
    )
    all_labels = all_labels_full[torch.from_numpy(indices)]

    conditions = [c.strip() for c in diag.DEFAULT_CONDITIONS.split(",")]
    condition_indices = [full_ds.label_names.index(c)
                         for c in conditions if c in full_ds.label_names]

    perm  = np.random.default_rng(0).permutation(n)
    n_tr  = int(0.8 * n)
    knn_train_idx = torch.from_numpy(perm[:n_tr])
    knn_test_idx  = torch.from_numpy(perm[n_tr:])

    print(f"Eval: {n} scans  kNN: {n_tr} train / {n - n_tr} test")
    return eval_loader, all_labels, condition_indices, knn_train_idx, knn_test_idx


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pretrain",  type=int,   default=1500)
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--crop_ratio",  type=float, default=0.65)
    parser.add_argument("--lr_pairs",    type=str,
                        default="3e-4:3e-4,3e-4:3e-5,3e-4:3e-6,3e-3:3e-5")
    parser.add_argument("--n_eval",      type=int,   default=500)
    parser.add_argument("--k",           type=int,   default=20)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--output_dir",  type=str,   default="runs/lr_sweep")
    args = parser.parse_args()

    lr_pairs = [
        (float(pair.split(":")[0]), float(pair.split(":")[1]))
        for pair in args.lr_pairs.split(",")
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== LR sweep (separate backbone / projector-predictor) ===")
    print(f"  n_pretrain={args.n_pretrain}  epochs={args.epochs}  "
          f"crop_ratio={args.crop_ratio}")
    print(f"  conditions: {lr_pairs}")
    print(f"  output → {out}\n")

    # ── shared eval set ───────────────────────────────────────────────────
    eval_loader, all_labels, condition_indices, knn_train_idx, knn_test_idx = \
        _build_eval_set(args.n_eval)

    # ── base CT-CLIP reference ────────────────────────────────────────────
    print("\nExtracting base CT-CLIP features (epoch-0 reference)...")
    backbone0 = CTViTBackbone(checkpoint_path=str(CHECKPOINT), use_pre_vq=True).to(device)
    feats0    = diag.extract_features(backbone0, eval_loader, device)
    del backbone0
    torch.cuda.empty_cache()
    base_metrics = diag._compute_metrics(feats0, all_labels, condition_indices,
                                         knn_train_idx, knn_test_idx, args.k)
    print(f"  [base]  ", end="")
    diag._print_row(0, base_metrics)

    # ── build shared pretrain dataset (same scans across all conditions) ──
    print(f"\nBuilding pretrain dataset ({args.n_pretrain} scans, "
          f"crop_ratio={args.crop_ratio})...")
    os.environ["CT_CLIP_MAX_SAMPLES"] = str(args.n_pretrain)
    pretrain_ds = CTPretrainDataset(
        data_folder=DATA_FOLDER, reports_file=REPORTS_FILE,
        meta_file=META_FILE,
        augmentation=CTAugmentation(crop_ratio=args.crop_ratio),
    )
    os.environ.pop("CT_CLIP_MAX_SAMPLES", None)
    print(f"  {len(pretrain_ds)} scans\n")

    # ── per-condition training + diagnostics ──────────────────────────────
    all_results = {}

    for bb_lr, pp_lr in lr_pairs:
        tag          = _lr_tag(bb_lr, pp_lr)
        pretrain_dir = out / tag / "pretrain"

        print(f"\n{'='*60}")
        print(f"backbone_lr={bb_lr:.0e}  projector_lr={pp_lr:.0e}  →  {pretrain_dir}")
        print(f"{'='*60}")

        pretrain_loader = DataLoader(
            pretrain_ds, batch_size=2, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )

        backbone  = CTViTBackbone(checkpoint_path=str(CHECKPOINT), use_pre_vq=True)
        projector = Projector(in_dim=512, hidden_dim=2048, out_dim=2048, norm="ln")
        predictor = Predictor(in_dim=2048, hidden_dim=512, out_dim=2048, norm="ln")

        train_condition(backbone, projector, predictor, pretrain_loader,
                        bb_lr, pp_lr, args.epochs, pretrain_dir, device)

        del pretrain_loader, projector, predictor
        backbone.cpu()
        torch.cuda.empty_cache()

        # diagnostics
        run_results = {0: base_metrics}
        print(f"\n  Diagnostics:")
        for epoch, ckpt_path in diag.find_checkpoints(pretrain_dir):
            bb = diag.load_backbone_from_ckpt(ckpt_path, device)
            feats = diag.extract_features(bb, eval_loader, device)
            del bb
            torch.cuda.empty_cache()
            m = diag._compute_metrics(feats, all_labels, condition_indices,
                                      knn_train_idx, knn_test_idx, args.k)
            run_results[epoch] = m
            print(f"    [ep {epoch:2d}]  ", end="")
            diag._print_row(epoch, m)

        all_results[tag] = run_results
        del backbone
        torch.cuda.empty_cache()

    # ── summary tables ────────────────────────────────────────────────────
    epochs = list(range(1, args.epochs + 1))
    print(f"\n{'='*60}")
    print(f"Summary  (base: uniformity={base_metrics['uniformity']:.4f}  "
          f"eff_rank={base_metrics['effective_rank']:.1f}  "
          f"knn_auroc={base_metrics['knn_auroc']:.4f})")
    print(f"{'='*60}")

    for metric, label in [
        ("uniformity",     "Uniformity   (closer to 0 = collapsed)"),
        ("effective_rank", "Eff. rank    (↓ = collapsed)"),
        ("knn_auroc",      "kNN AUROC"),
    ]:
        print(f"\n{label}:")
        print(f"  {'condition':<22}" + "".join(f"  ep{e}" for e in epochs))
        for bb_lr, pp_lr in lr_pairs:
            tag = _lr_tag(bb_lr, pp_lr)
            row = f"  {tag:<22}"
            for e in epochs:
                v = all_results[tag].get(e, {}).get(metric, float("nan"))
                row += f"  {v:6.3f}" if not np.isnan(v) else f"  {'n/a':>6}"
            print(row)

    # ── save ──────────────────────────────────────────────────────────────
    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "lr_pairs":    [[bb, pp] for bb, pp in lr_pairs],
            "n_pretrain":  args.n_pretrain,
            "epochs":      args.epochs,
            "crop_ratio":  args.crop_ratio,
            "base_metrics": base_metrics,
            "results": {
                tag: {str(ep): m for ep, m in res.items()}
                for tag, res in all_results.items()
            },
        }, f, indent=2)
    print(f"\nResults saved → {out_path}")

    if diag._MATPLOTLIB:
        diag.plot_results(all_results, out / "diagnostics.png")


if __name__ == "__main__":
    main()
