"""
AUROC evaluation on the test split.

Runs the frozen backbone + classifier over the test loader, collects
all predictions and ground-truth labels, then computes per-label AUROC
and macro mean.  Results are printed and saved as a CSV.
"""

import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


def evaluate(
    backbone: nn.Module,
    classifier: nn.Module,
    test_loader: DataLoader,
    label_names: List[str],
    device: torch.device,
    output_csv: Optional[str] = None,
) -> dict:
    """
    Returns a dict mapping each label name → AUROC, plus 'macro_auroc'.
    Labels with no positive examples are skipped (NaN).
    """
    backbone.eval()
    classifier.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = classifier(backbone(x))
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    y_pred = torch.sigmoid(torch.cat(all_logits, dim=0)).numpy()   # (N, C)
    y_true = torch.cat(all_labels, dim=0).numpy()                  # (N, C)

    results = {}
    for i, name in enumerate(label_names):
        pos = y_true[:, i].sum()
        neg = (1 - y_true[:, i]).sum()
        if pos > 0 and neg > 0:
            results[name] = float(roc_auc_score(y_true[:, i], y_pred[:, i]))
        else:
            results[name] = float("nan")
            print(f"[evaluate] {name}: skipped (no positives or no negatives)")

    valid_aurocs = [v for v in results.values() if not np.isnan(v)]
    results["macro_auroc"] = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")

    # Print summary
    print(f"\n{'Label':<45} AUROC")
    print("-" * 55)
    for name, auc in results.items():
        tag = f"{auc:.4f}" if not np.isnan(auc) else "  n/a"
        print(f"{name:<45} {tag}")

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "auroc"])
            for name, auc in results.items():
                writer.writerow([name, "" if np.isnan(auc) else f"{auc:.6f}"])
        print(f"\n[evaluate] results saved → {output_csv}")

    return results
