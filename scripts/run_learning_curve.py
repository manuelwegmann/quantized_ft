"""
Learning curve: pretrained vs random-init backbone for atelectasis classification.

Loads cached 512-dim features (from cache_probe_features.py) and sweeps
N_train samples, reporting mean ± std over multiple random seeds for:
  AUROC, accuracy, precision, recall, F1  (threshold=0.5 for the latter four)

AUROC is used for val checkpoint selection (threshold-free, more stable).
Accuracy/precision/recall/F1 are computed on the test set at threshold=0.5.

Runtime is minutes (no CT loading).

Usage:
    python scripts/run_learning_curve.py
    python scripts/run_learning_curve.py --ns 30,100,300,1000,all --seeds 10 --epochs 300
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from models.backbone import EMBED_DIM

CACHE_DIR  = _PROJECT_ROOT / "runs" / "feature_cache"
OUTPUT_DIR = _PROJECT_ROOT / "runs" / "learning_curve"

METRICS = ("auroc", "accuracy", "precision", "recall", "f1")


def load_split(cache_dir: Path, backbone: str, split: str):
    d = cache_dir / backbone
    feats  = torch.load(d / f"{split}_feats.pt",  map_location="cpu")
    labels = torch.load(d / f"{split}_labels.pt", map_location="cpu")
    return feats, labels


def train_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    seed: int,
    lr: float = 1e-3,
    patience: int = 10,
    eval_every: int = 20,
) -> nn.Module:
    """
    patience: number of consecutive val checks without improvement before stopping.
              Each check is every eval_every epochs, so patience=10 + eval_every=20
              means stop after 200 epochs of no progress.
    """
    torch.manual_seed(seed)
    probe = nn.Linear(EMBED_DIM, 1).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr)
    crit  = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_auroc    = -1.0
    best_state    = {k: v.clone() for k, v in probe.state_dict().items()}
    checks_no_imp = 0

    for epoch in range(1, epochs + 1):
        probe.train()
        for xb, yb in loader:
            logits = probe(xb.to(device)).squeeze(1)
            loss   = crit(logits, yb[:, 0].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % eval_every == 0 or epoch == epochs:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(val_feats.to(device)).squeeze(1)
            try:
                auroc = float(roc_auc_score(
                    val_labels[:, 0].numpy(),
                    torch.sigmoid(val_logits).cpu().numpy(),
                ))
            except ValueError:
                auroc = 0.5

            if auroc > best_auroc:
                best_auroc    = auroc
                best_state    = {k: v.clone() for k, v in probe.state_dict().items()}
                checks_no_imp = 0
            else:
                checks_no_imp += 1
                if checks_no_imp >= patience:
                    break

    probe.load_state_dict(best_state)
    return probe, epoch


def compute_metrics(probe: nn.Module, feats: torch.Tensor, labels: torch.Tensor,
                    device: torch.device) -> dict:
    probe.eval()
    with torch.no_grad():
        logits = probe(feats.to(device)).squeeze(1)
    probs = torch.sigmoid(logits).cpu().numpy()
    y     = labels[:, 0].numpy().astype(int)
    preds = (probs >= 0.5).astype(int)
    return {
        "auroc":     float(roc_auc_score(y, probs)),
        "accuracy":  float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall":    float(recall_score(y, preds, zero_division=0)),
        "f1":        float(f1_score(y, preds, zero_division=0)),
    }


def run_n(train_feats, train_labels, val_feats, val_labels,
          test_feats, test_labels, n_train, seeds, device, epochs) -> dict:
    """Returns {metric: [value_per_seed]} for all METRICS."""
    per_seed = {m: [] for m in METRICS}
    per_seed["stopped_epoch"] = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        n   = min(n_train, len(train_feats))
        idx = torch.from_numpy(rng.choice(len(train_feats), n, replace=False))
        probe, stopped_epoch = train_probe(
            train_feats[idx], train_labels[idx],
            val_feats,        val_labels,
            device, epochs, seed=seed,
        )
        per_seed["stopped_epoch"].append(stopped_epoch)
        m = compute_metrics(probe, test_feats, test_labels, device)
        for k in METRICS:
            per_seed[k].append(m[k])
    return per_seed


def _fmt(mean, std):
    return f"{mean:.4f} ±{std:.4f}"


def print_head_to_head(results: dict, ns: list) -> None:
    for n in ns:
        p = results["pretrained"][n]
        r = results["random"][n]
        header = f"  N_train = {n}"
        print(f"\n{'='*62}")
        print(header)
        print(f"{'='*62}")
        print(f"  {'Metric':<12}  {'Pretrained':>16}  {'Random':>16}  {'Delta':>8}")
        print(f"  {'-'*12}  {'-'*16}  {'-'*16}  {'-'*8}")
        for m in METRICS:
            pm = float(np.mean(p[m]))
            ps = float(np.std(p[m]))
            rm = float(np.mean(r[m]))
            rs = float(np.std(r[m]))
            delta = pm - rm
            sign  = "+" if delta >= 0 else ""
            print(f"  {m:<12}  {_fmt(pm, ps):>16}  {_fmt(rm, rs):>16}  {sign}{delta:.4f}")
        p_ep = "  ".join(str(e) for e in p["stopped_epoch"])
        r_ep = "  ".join(str(e) for e in r["stopped_epoch"])
        print(f"  {'epochs':<12}  [{p_ep}]")
        print(f"  {'':12}  [{r_ep}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default=str(CACHE_DIR))
    parser.add_argument("--epochs",    type=int, default=300)
    parser.add_argument("--seeds",     type=int, default=5,
                        help="Number of random subsampling seeds per N")
    parser.add_argument("--ns",        default="30,100,300,1000,all",
                        help="Comma-separated N_train values; 'all' = full training set")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds  = list(range(args.seeds))
    cache  = Path(args.cache_dir)

    print("Loading cached features...")
    data = {}
    for backbone in ("pretrained", "random"):
        data[backbone] = {
            split: load_split(cache, backbone, split)
            for split in ("train", "val", "test")
        }
    n_train_total = len(data["pretrained"]["train"][0])
    n_val         = len(data["pretrained"]["val"][0])
    n_test        = len(data["pretrained"]["test"][0])
    print(f"  train={n_train_total}  val={n_val}  test={n_test}")

    ns_raw = [x.strip() for x in args.ns.split(",")]
    ns = []
    for x in ns_raw:
        if x == "all":
            ns.append(n_train_total)
        else:
            v = int(x)
            if v <= n_train_total:
                ns.append(v)
    ns = sorted(set(ns))
    print(f"N_train sweep : {ns}")
    print(f"Seeds         : {seeds}  (per N)")
    print(f"Epochs        : {args.epochs}")

    results = {}
    for backbone in ("pretrained", "random"):
        print(f"\n=== {backbone} ===")
        train_feats, train_labels = data[backbone]["train"]
        val_feats,   val_labels   = data[backbone]["val"]
        test_feats,  test_labels  = data[backbone]["test"]
        results[backbone] = {}
        for n in ns:
            per_seed = run_n(
                train_feats, train_labels,
                val_feats,   val_labels,
                test_feats,  test_labels,
                n, seeds, device, args.epochs,
            )
            results[backbone][n] = per_seed
            auroc_str   = "  ".join(f"{a:.4f}" for a in per_seed["auroc"])
            epochs_str  = "  ".join(str(e) for e in per_seed["stopped_epoch"])
            mean_a = float(np.mean(per_seed["auroc"]))
            std_a  = float(np.std(per_seed["auroc"]))
            print(f"  N={n:6d}  AUROC {mean_a:.4f} ±{std_a:.4f}   [{auroc_str}]")
            print(f"          stopped at epoch: [{epochs_str}]")

    print_head_to_head(results, ns)

    # AUROC learning curve summary table
    print(f"\n{'='*62}")
    print("AUROC learning curve")
    print(f"{'='*62}")
    print(f"  {'N_train':>8}  {'Pretrained':>16}  {'Random':>16}  {'Delta':>8}")
    print(f"  {'-'*8}  {'-'*16}  {'-'*16}  {'-'*8}")
    for n in ns:
        pm = float(np.mean(results["pretrained"][n]["auroc"]))
        ps = float(np.std(results["pretrained"][n]["auroc"]))
        rm = float(np.mean(results["random"][n]["auroc"]))
        rs = float(np.std(results["random"][n]["auroc"]))
        delta = pm - rm
        sign  = "+" if delta >= 0 else ""
        print(f"  {n:>8d}  {_fmt(pm, ps):>16}  {_fmt(rm, rs):>16}  {sign}{delta:.4f}")

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    # Serialise: convert numpy scalars and lists to plain Python
    def serialise(obj):
        if isinstance(obj, dict):
            return {str(k): serialise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [float(x) for x in obj]
        return float(obj)

    with open(out_path, "w") as f:
        json.dump({
            "ns":      ns,
            "epochs":  args.epochs,
            "seeds":   seeds,
            "metrics": list(METRICS),
            "results": serialise(results),
        }, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
