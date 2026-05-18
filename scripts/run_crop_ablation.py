"""
Crop-ratio ablation for SimSiam pretraining.

Trains FP SimSiam at different crop ratios on N scans for E epochs each,
saving a checkpoint every epoch. Then evaluates per-epoch feature geometry
(uniformity, effective rank, kNN AUROC) to identify which crop strength
best prevents collapse before committing to full-scale pretraining.

Usage:
    python scripts/run_crop_ablation.py
    python scripts/run_crop_ablation.py --n_pretrain 3000 --epochs 3 --crop_ratios 0.75,0.65,0.55
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import paths  # noqa: E402 — registers CT-CLIP packages on sys.path

from models.backbone import CTViTBackbone
from models.simsiam import Projector, Predictor
from pretrain.augmentations import CTAugmentation
from pretrain.dataset import CTPretrainDataset
from pretrain.trainer import train as pretrain_train
from cache_all_features import AllScansDataset
import diagnose_pretraining as diag

from paths import CHECKPOINT, DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE


def _pretrain_cfg(output_dir: Path, epochs: int, lr: float) -> dict:
    return {
        "mode": "fp",
        "use_aux_loss": False,
        "backbone": {"checkpoint": str(CHECKPOINT), "use_pre_vq": True},
        "training": {
            "batch_size": 2,
            "lr": lr,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "epochs": epochs,
            "num_workers": 4,
            "save_every": 1,  # checkpoint every epoch — we only have 3
            "lr_schedule": "cosine",
            "freeze_epochs": 0,
        },
        "projector": {"in_dim": 512, "hidden_dim": 2048, "out_dim": 2048},
        "predictor": {"in_dim": 2048, "hidden_dim": 512, "out_dim": 2048},
        "output_dir": str(output_dir),
    }


def _build_eval_set(n_eval: int):
    """Build a fixed eval set shared across all crop-ratio conditions."""
    full_ds = AllScansDataset(DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE)
    n = min(n_eval, len(full_ds))
    indices = np.sort(np.random.default_rng(42).choice(len(full_ds), size=n, replace=False))
    eval_ds = Subset(full_ds, indices.tolist())
    eval_loader = DataLoader(
        eval_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True,
    )

    all_labels_full = torch.tensor(
        np.stack([lbl for _, lbl in full_ds.samples]), dtype=torch.float32,
    )
    all_labels = all_labels_full[torch.from_numpy(indices)]

    conditions = [c.strip() for c in diag.DEFAULT_CONDITIONS.split(",")]
    label_names = full_ds.label_names
    condition_indices = [label_names.index(c) for c in conditions if c in label_names]

    perm = np.random.default_rng(0).permutation(n)
    n_tr = int(0.8 * n)
    knn_train_idx = torch.from_numpy(perm[:n_tr])
    knn_test_idx  = torch.from_numpy(perm[n_tr:])

    print(f"Eval: {n} scans  kNN: {n_tr} train / {n - n_tr} test")
    return eval_loader, all_labels, condition_indices, knn_train_idx, knn_test_idx


def _run_diagnostics(pretrain_dir, eval_loader, all_labels, condition_indices,
                     knn_train_idx, knn_test_idx, k, base_metrics, device):
    """Load each per-epoch checkpoint and compute geometry metrics."""
    run_results = {0: base_metrics}
    for epoch, ckpt_path in diag.find_checkpoints(pretrain_dir):
        bb = diag.load_backbone_from_ckpt(ckpt_path, device)
        feats = diag.extract_features(bb, eval_loader, device)
        del bb
        torch.cuda.empty_cache()
        m = diag._compute_metrics(feats, all_labels, condition_indices,
                                  knn_train_idx, knn_test_idx, k)
        run_results[epoch] = m
        print(f"    [ep {epoch:2d}]  ", end="")
        diag._print_row(epoch, m)
    return run_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pretrain",  type=int,   default=3000)
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--crop_ratios", type=str,   default="0.75,0.65,0.55")
    parser.add_argument("--n_eval",      type=int,   default=500)
    parser.add_argument("--k",           type=int,   default=20)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--output_dir",  type=str,   default="runs/crop_ablation")
    args = parser.parse_args()

    crop_ratios = [float(r) for r in args.crop_ratios.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== Crop ratio ablation ===")
    print(f"  crop_ratios={crop_ratios}  n_pretrain={args.n_pretrain}  "
          f"epochs={args.epochs}  lr={args.lr:.2e}")
    print(f"  output → {out}\n")

    # ── shared eval set ───────────────────────────────────────────────────────
    eval_loader, all_labels, condition_indices, knn_train_idx, knn_test_idx = \
        _build_eval_set(args.n_eval)

    # ── base CT-CLIP reference (epoch 0, shared across all conditions) ────────
    print("\nExtracting base CT-CLIP features (epoch-0 reference)...")
    backbone0 = CTViTBackbone(checkpoint_path=CHECKPOINT, use_pre_vq=True).to(device)
    feats0 = diag.extract_features(backbone0, eval_loader, device)
    del backbone0
    torch.cuda.empty_cache()
    base_metrics = diag._compute_metrics(feats0, all_labels, condition_indices,
                                         knn_train_idx, knn_test_idx, args.k)
    print(f"  [base]  ", end="")
    diag._print_row(0, base_metrics)

    # ── train + diagnose each crop ratio ──────────────────────────────────────
    all_results = {}

    for crop_ratio in crop_ratios:
        tag = f"crop_{round(crop_ratio * 100)}"
        pretrain_dir = out / tag / "pretrain"
        print(f"\n{'='*60}")
        print(f"crop_ratio={crop_ratio}  →  {pretrain_dir}")
        print(f"{'='*60}")

        os.environ["CT_CLIP_MAX_SAMPLES"] = str(args.n_pretrain)
        pretrain_ds = CTPretrainDataset(
            data_folder=DATA_FOLDER, reports_file=REPORTS_FILE,
            meta_file=META_FILE,
            augmentation=CTAugmentation(crop_ratio=crop_ratio),
        )
        os.environ.pop("CT_CLIP_MAX_SAMPLES", None)
        pretrain_loader = DataLoader(
            pretrain_ds, batch_size=2, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        print(f"  Pretrain dataset: {len(pretrain_ds)} scans")

        backbone  = CTViTBackbone(checkpoint_path=CHECKPOINT, use_pre_vq=True)
        projector = Projector(in_dim=512, hidden_dim=2048, out_dim=2048, norm="ln")
        predictor = Predictor(in_dim=2048, hidden_dim=512, out_dim=2048, norm="ln")

        pretrain_train(backbone, projector, predictor, pretrain_loader,
                       _pretrain_cfg(pretrain_dir, args.epochs, args.lr), device)

        del pretrain_loader, projector, predictor
        backbone.cpu()
        torch.cuda.empty_cache()

        print(f"\n  Diagnostics (crop_ratio={crop_ratio}):")
        all_results[tag] = _run_diagnostics(
            pretrain_dir, eval_loader, all_labels, condition_indices,
            knn_train_idx, knn_test_idx, args.k, base_metrics, device,
        )

        del backbone
        torch.cuda.empty_cache()

    # ── summary tables ────────────────────────────────────────────────────────
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
        print(f"  {'crop':>8}" + "".join(f"  ep{e}" for e in epochs))
        for crop_ratio in crop_ratios:
            tag = f"crop_{round(crop_ratio * 100)}"
            row = f"  {crop_ratio:>8.2f}"
            for e in epochs:
                v = all_results[tag].get(e, {}).get(metric, float("nan"))
                row += f"  {v:6.3f}" if not np.isnan(v) else f"  {'n/a':>6}"
            print(row)

    # ── save results ──────────────────────────────────────────────────────────
    out_path = out / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "crop_ratios": crop_ratios,
            "n_pretrain": args.n_pretrain,
            "epochs": args.epochs,
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
