"""
Entry point for SimSiam pretraining (FP or SSQL).

Usage:
    python run_pretrain.py --config configs/pretrain_fp.yaml
    python run_pretrain.py --config configs/pretrain_ssql.yaml
    python run_pretrain.py --config configs/pretrain_fp.yaml --max_samples 10
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Make quantized_ft/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from models.backbone import CTViTBackbone
from models.simsiam import Projector, Predictor
from pretrain.augmentations import CTAugmentation
from pretrain.dataset import CTPretrainDataset
from pretrain.trainer import train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Override CT_CLIP_MAX_SAMPLES (useful for smoke tests)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.max_samples is not None:
        os.environ["CT_CLIP_MAX_SAMPLES"] = str(args.max_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_pretrain] device={device}  mode={cfg['mode']}")

    # ---- Dataset ----
    aug_cfg = cfg.get("augmentation", {})
    augmentation = CTAugmentation(
        crop_ratio=aug_cfg.get("crop_ratio", 0.85),
        flip_p=aug_cfg.get("flip_p", 0.5),
        intensity_jitter=aug_cfg.get("intensity_jitter", True),
        gaussian_noise=aug_cfg.get("gaussian_noise", True),
        noise_std=aug_cfg.get("noise_std", 0.01),
    )
    dataset = CTPretrainDataset(
        data_folder=cfg["data"]["data_folder"],
        reports_file=cfg["data"]["reports_file"],
        meta_file=cfg["data"].get("meta_file"),
        augmentation=augmentation,
    )
    print(f"[run_pretrain] dataset size: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # ---- Models ----
    backbone = CTViTBackbone(
        checkpoint_path=cfg["backbone"]["checkpoint"],
        use_pre_vq=cfg["backbone"].get("use_pre_vq", False),
    )
    proj_cfg = cfg.get("projector", {})
    pred_cfg = cfg.get("predictor", {})
    projector = Projector(
        in_dim=proj_cfg.get("in_dim", 512),
        hidden_dim=proj_cfg.get("hidden_dim", 2048),
        out_dim=proj_cfg.get("out_dim", 2048),
    )
    predictor = Predictor(
        in_dim=pred_cfg.get("in_dim", 2048),
        hidden_dim=pred_cfg.get("hidden_dim", 512),
        out_dim=pred_cfg.get("out_dim", 2048),
    )

    # ---- Train ----
    train(backbone, projector, predictor, loader, cfg, device)


if __name__ == "__main__":
    main()
