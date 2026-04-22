"""
SimSiam pretraining trainer.

Supports two modes:
  'fp'   — standard full-precision SimSiam baseline
  'ssql' — SSQL recipe: quantized prediction branch + optional FP auxiliary loss

The same function handles both; mode is selected via config['mode'].
"""

import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.quantization import quantized_forward, sample_bits
from pretrain.loss import negative_cosine_similarity


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    backbone: nn.Module,
    projector: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
):
    torch.save(
        {
            "epoch": epoch,
            "backbone": backbone.state_dict(),
            "projector": projector.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    backbone: nn.Module,
    projector: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    backbone.load_state_dict(ckpt["backbone"])
    projector.load_state_dict(ckpt["projector"])
    predictor.load_state_dict(ckpt["predictor"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def _train_epoch(
    backbone: nn.Module,
    projector: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str,
    use_aux_loss: bool,
    w_bits_range: tuple,
    a_bits_range: tuple,
) -> float:
    backbone.train()
    projector.train()
    predictor.train()

    total_loss = 0.0
    for x1, x2 in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)

        # Full-precision forward (always computed — used as target in SSQL
        # and as the training loss in FP mode)
        z1_fp = projector(backbone(x1))
        z2_fp = projector(backbone(x2))
        p1_fp = predictor(z1_fp)
        p2_fp = predictor(z2_fp)

        if mode == "fp":
            loss = (
                negative_cosine_similarity(p1_fp, z2_fp.detach())
                + negative_cosine_similarity(p2_fp, z1_fp.detach())
            )

        else:  # ssql
            w_bits, a_bits = sample_bits(w_bits_range, a_bits_range)
            with quantized_forward([backbone, projector], w_bits, a_bits):
                z1_q = projector(backbone(x1))
                z2_q = projector(backbone(x2))
            # Predictor is always full precision
            p1_q = predictor(z1_q)
            p2_q = predictor(z2_q)

            L_ssql = (
                negative_cosine_similarity(p1_q, z2_fp.detach())
                + negative_cosine_similarity(p2_q, z1_fp.detach())
            )
            if use_aux_loss:
                L_fp = (
                    negative_cosine_similarity(p1_fp, z2_fp.detach())
                    + negative_cosine_similarity(p2_fp, z1_fp.detach())
                )
                loss = L_ssql + L_fp
            else:
                loss = L_ssql

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    backbone: nn.Module,
    projector: nn.Module,
    predictor: nn.Module,
    train_loader: DataLoader,
    config: Dict,
    device: torch.device,
):
    mode        = config["mode"]               # 'fp' or 'ssql'
    use_aux     = config.get("use_aux_loss", True)
    epochs      = config["training"]["epochs"]
    output_dir  = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    w_bits_range = (
        config.get("quantization", {}).get("w_bits_min", 2),
        config.get("quantization", {}).get("w_bits_max", 8),
    )
    a_bits_range = (
        config.get("quantization", {}).get("a_bits_min", 4),
        config.get("quantization", {}).get("a_bits_max", 8),
    )

    params = (
        list(backbone.parameters())
        + list(projector.parameters())
        + list(predictor.parameters())
    )
    optimizer = torch.optim.SGD(
        params,
        lr=config["training"]["lr"],
        momentum=config["training"].get("momentum", 0.9),
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )

    start_epoch = 0
    resume_path = output_dir / "checkpoint_latest.pt"
    if resume_path.exists():
        start_epoch = load_checkpoint(backbone, projector, predictor, optimizer, str(resume_path))
        print(f"[trainer] resumed from epoch {start_epoch}")

    backbone.to(device)
    projector.to(device)
    predictor.to(device)

    for epoch in range(start_epoch, epochs):
        avg_loss = _train_epoch(
            backbone, projector, predictor, train_loader, optimizer,
            device, mode, use_aux, w_bits_range, a_bits_range,
        )
        print(f"[pretrain] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

        save_checkpoint(
            backbone, projector, predictor, optimizer, epoch + 1,
            str(output_dir / "checkpoint_latest.pt"),
        )

        if (epoch + 1) % config["training"].get("save_every", 10) == 0:
            save_checkpoint(
                backbone, projector, predictor, optimizer, epoch + 1,
                str(output_dir / f"checkpoint_ep{epoch + 1:04d}.pt"),
            )

    save_checkpoint(
        backbone, projector, predictor, optimizer, epochs,
        str(output_dir / "checkpoint_final.pt"),
    )
    print(f"[pretrain] done. Final checkpoint → {output_dir / 'checkpoint_final.pt'}")
