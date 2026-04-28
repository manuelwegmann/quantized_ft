"""
Learning curve: pretrained ViT vs random ViT vs random CNN for atelectasis.

Loads cached features and sweeps N_train samples, reporting mean ± std over
multiple random seeds for: AUROC, accuracy, precision, recall, F1.

The CNN baseline (random_cnn) is included automatically if its feature cache
exists (runs/feature_cache/random_cnn/). Feature dim is inferred from the
cached tensors, so ViT (512-dim) and CNN (256-dim) are handled transparently.

AUROC drives val checkpoint selection and early stopping (threshold-free).
Accuracy/precision/recall/F1 are computed on the test set at threshold=0.5.

Usage:
    python scripts/run_learning_curve.py
    python scripts/run_learning_curve.py --ns 30,100,300,1000,all --seeds 5 --epochs 1000
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

CACHE_DIR  = _PROJECT_ROOT / "runs" / "feature_cache"
OUTPUT_DIR = _PROJECT_ROOT / "runs" / "learning_curve"

METRICS = ("auroc", "accuracy", "precision", "recall", "f1")

DISPLAY = {
    "pretrained": "ViT pretrained",
    "random":     "ViT random",
    "random_cnn": "CNN random",
}


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
):
    """
    Probe input dim is inferred from train_feats, so this works for any
    backbone (ViT 512-dim, CNN 256-dim, etc.) without configuration.

    patience: consecutive val checks without improvement before stopping.
              patience=10 + eval_every=20 → stop after 200 stagnant epochs.
    """
    torch.manual_seed(seed)
    embed_dim = train_feats.shape[1]
    probe = nn.Linear(embed_dim, 1).to(device)
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


def print_results(results: dict, backbones: list, ns: list) -> None:
    col_w = 18
    # ── per-N detailed table ───────────────────────────────────────────────
    for n in ns:
        print(f"\n{'='*72}")
        print(f"  N_train = {n}")
        print(f"{'='*72}")
        header = f"  {'Metric':<12}" + "".join(f"  {DISPLAY[b]:>{col_w}}" for b in backbones)
        print(header)
        print(f"  {'-'*12}" + f"  {'-'*col_w}" * len(backbones))
        for m in METRICS:
            row = f"  {m:<12}"
            for b in backbones:
                mean_m = float(np.mean(results[b][n][m]))
                std_m  = float(np.std(results[b][n][m]))
                row += f"  {_fmt(mean_m, std_m):>{col_w}}"
            print(row)
        # stopped epochs per backbone
        for b in backbones:
            ep_str = "  ".join(str(e) for e in results[b][n]["stopped_epoch"])
            print(f"  {'epochs' if b == backbones[0] else '':12}  {DISPLAY[b]}: [{ep_str}]")

    # ── AUROC learning curve summary ───────────────────────────────────────
    print(f"\n{'='*72}")
    print("AUROC learning curve")
    print(f"{'='*72}")
    header = f"  {'N_train':>8}" + "".join(f"  {DISPLAY[b]:>{col_w}}" for b in backbones)
    if len(backbones) > 1:
        header += f"  {'Δ pretrained':>14}"
    print(header)
    print(f"  {'-'*8}" + f"  {'-'*col_w}" * len(backbones) + ("  " + "-"*14 if len(backbones) > 1 else ""))
    for n in ns:
        row = f"  {n:>8d}"
        p_mean = float(np.mean(results["pretrained"][n]["auroc"])) if "pretrained" in results else None
        for b in backbones:
            mean_a = float(np.mean(results[b][n]["auroc"]))
            std_a  = float(np.std(results[b][n]["auroc"]))
            row += f"  {_fmt(mean_a, std_a):>{col_w}}"
        if p_mean is not None and len(backbones) > 1:
            # show delta of each non-pretrained backbone vs pretrained
            deltas = []
            for b in backbones:
                if b != "pretrained":
                    d = p_mean - float(np.mean(results[b][n]["auroc"]))
                    deltas.append(f"+{d:.4f}" if d >= 0 else f"{d:.4f}")
            row += f"  {' / '.join(deltas):>14}"
        print(row)


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

    # ── load whichever caches exist ────────────────────────────────────────
    candidates = ["pretrained", "random", "random_cnn"]
    backbones  = [b for b in candidates if (cache / b / "train_feats.pt").exists()]
    if not backbones:
        raise FileNotFoundError(f"No feature caches found under {cache}")

    print("Loading cached features...")
    data = {}
    for b in backbones:
        data[b] = {s: load_split(cache, b, s) for s in ("train", "val", "test")}
        dim = data[b]["train"][0].shape[1]
        print(f"  {DISPLAY[b]:<18}  train={len(data[b]['train'][0])}  dim={dim}")

    n_train_total = len(data["pretrained"]["train"][0])

    ns_raw = [x.strip() for x in args.ns.split(",")]
    ns = sorted({
        n_train_total if x == "all" else int(x)
        for x in ns_raw
        if x == "all" or int(x) <= n_train_total
    })
    print(f"N_train sweep : {ns}")
    print(f"Seeds         : {seeds}  (per N)")
    print(f"Epochs        : {args.epochs}")
    print(f"Backbones     : {[DISPLAY[b] for b in backbones]}")

    results = {}
    for b in backbones:
        print(f"\n=== {DISPLAY[b]} ===")
        train_feats, train_labels = data[b]["train"]
        val_feats,   val_labels   = data[b]["val"]
        test_feats,  test_labels  = data[b]["test"]
        results[b] = {}
        for n in ns:
            per_seed = run_n(
                train_feats, train_labels,
                val_feats,   val_labels,
                test_feats,  test_labels,
                n, seeds, device, args.epochs,
            )
            results[b][n] = per_seed
            auroc_str  = "  ".join(f"{a:.4f}" for a in per_seed["auroc"])
            epochs_str = "  ".join(str(e) for e in per_seed["stopped_epoch"])
            mean_a = float(np.mean(per_seed["auroc"]))
            std_a  = float(np.std(per_seed["auroc"]))
            print(f"  N={n:6d}  AUROC {mean_a:.4f} ±{std_a:.4f}   [{auroc_str}]")
            print(f"          stopped at epoch: [{epochs_str}]")

    print_results(results, backbones, ns)

    # ── save ──────────────────────────────────────────────────────────────
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    def serialise(obj):
        if isinstance(obj, dict):
            return {str(k): serialise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [int(x) if isinstance(x, (int, np.integer)) else float(x) for x in obj]
        return float(obj)

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "ns":       ns,
            "epochs":   args.epochs,
            "seeds":    seeds,
            "metrics":  list(METRICS),
            "backbones": backbones,
            "results":  serialise(results),
        }, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
