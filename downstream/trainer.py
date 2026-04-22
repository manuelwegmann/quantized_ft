"""
Downstream linear-probe training.

The backbone is frozen by default (linear probe evaluation).
Setting freeze_backbone=False enables full fine-tuning (future use).
"""

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train(
    backbone: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
):
    freeze_backbone = config.get("freeze_backbone", True)
    epochs     = config["training"]["epochs"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if freeze_backbone:
        backbone.freeze()
        backbone.eval()
        trained_params = classifier.parameters()
    else:
        backbone.unfreeze()
        backbone.train()
        trained_params = list(backbone.parameters()) + list(classifier.parameters())

    backbone.to(device)
    classifier.to(device)

    optimizer = torch.optim.Adam(
        trained_params,
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ------ train ------
        classifier.train()
        if not freeze_backbone:
            backbone.train()

        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad() if freeze_backbone else torch.enable_grad():
                emb = backbone(x)
            logits = classifier(emb)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # ------ validate ------
        backbone.eval()
        classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = classifier(backbone(x))
                val_loss += criterion(logits, y).item()
        val_loss /= max(len(val_loader), 1)

        print(
            f"[downstream] epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"backbone": backbone.state_dict(), "classifier": classifier.state_dict()},
                output_dir / "checkpoint_best.pt",
            )

    torch.save(
        {"backbone": backbone.state_dict(), "classifier": classifier.state_dict()},
        output_dir / "checkpoint_final.pt",
    )
    print(f"[downstream] done. Best val_loss={best_val_loss:.4f}")
