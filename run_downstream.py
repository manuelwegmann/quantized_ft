"""
Entry point for downstream linear-probe evaluation.

Usage:
    python run_downstream.py --config configs/downstream.yaml
    python run_downstream.py --config configs/downstream.yaml \
        --pretrain_checkpoint runs/pretrain_ssql/checkpoint_final.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from downstream.dataset import MerlinDataset
from downstream.evaluate import evaluate
from downstream.trainer import train
from models.backbone import CTViTBackbone, EMBED_DIM
from models.classifier import LinearProbe, FineTuneHead


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--pretrain_checkpoint",
        default=None,
        help="Override pretrain_checkpoint in config",
    )
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.pretrain_checkpoint:
        cfg["pretrain_checkpoint"] = args.pretrain_checkpoint
    if args.max_samples is not None:
        os.environ["CT_CLIP_MAX_SAMPLES"] = str(args.max_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_downstream] device={device}")

    # ---- Datasets ----
    data_cfg = cfg["data"]

    def make_loader(split, shuffle):
        ds = MerlinDataset(
            data_folder=data_cfg["data_folder"],
            reports_file=data_cfg["reports_file"],
            labels_file=data_cfg["labels_file"],
            meta_file=data_cfg.get("meta_file"),
            split=split,
            seed=data_cfg.get("split_seed", 42),
        )
        print(f"[run_downstream] {split} split: {len(ds)} scans")
        return DataLoader(
            ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=shuffle,
            num_workers=cfg["training"].get("num_workers", 4),
            pin_memory=True,
        ), ds

    train_loader, train_ds = make_loader("train", shuffle=True)
    val_loader,   _        = make_loader("val",   shuffle=False)
    test_loader,  test_ds  = make_loader("test",  shuffle=False)

    label_names = train_ds.label_names
    n_classes   = cfg.get("n_classes", len(label_names))

    # ---- Backbone ----
    backbone = CTViTBackbone(
        checkpoint_path=cfg["backbone"]["checkpoint"],
        use_pre_vq=cfg["backbone"].get("use_pre_vq", False),
    )

    # Load SimSiam pretrain weights into backbone (if available)
    pretrain_ckpt = cfg.get("pretrain_checkpoint")
    if pretrain_ckpt and Path(pretrain_ckpt).exists():
        ckpt = torch.load(pretrain_ckpt, map_location="cpu")
        backbone.load_state_dict(ckpt["backbone"])
        print(f"[run_downstream] loaded pretrain backbone from {pretrain_ckpt}")
    else:
        print("[run_downstream] no pretrain checkpoint found — using CT-CLIP init only")

    # ---- Classifier ----
    freeze = cfg.get("freeze_backbone", True)
    if freeze:
        classifier = LinearProbe(in_dim=EMBED_DIM, n_classes=n_classes)
    else:
        classifier = FineTuneHead(in_dim=EMBED_DIM, n_classes=n_classes)

    # ---- Train ----
    train(backbone, classifier, train_loader, val_loader, cfg, device)

    # ---- Evaluate on test set ----
    best_ckpt = Path(cfg["output_dir"]) / "checkpoint_best.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location="cpu")
        backbone.load_state_dict(state["backbone"])
        classifier.load_state_dict(state["classifier"])
        backbone.to(device)
        classifier.to(device)

    output_csv = str(Path(cfg["output_dir"]) / "auroc_results.csv")
    evaluate(backbone, classifier, test_loader, label_names, device, output_csv=output_csv)


if __name__ == "__main__":
    main()
