"""
Multi-condition linear probe evaluation: pretrained ViT vs random ViT vs random CNN.

Loads full-scan features (from cache_all_features.py), applies a per-condition
label filter and seeded 70/15/15 split, then sweeps N_train for each condition
and backbone, reporting AUROC (and threshold-based metrics at full N).

Usage:
    python scripts/run_multi_condition.py
    python scripts/run_multi_condition.py --conditions atelectasis,pleural_effusion
    python scripts/run_multi_condition.py --epochs 1000 --seeds 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

CACHE_DIR  = _PROJECT_ROOT / "runs" / "feature_cache_full"
OUTPUT_DIR = _PROJECT_ROOT / "runs" / "multi_condition"

DEFAULT_CONDITIONS = (
    "atelectasis,surgically_absent_gallbladder,renal_cyst,"
    "pleural_effusion,cardiomegaly,gallstones"
)
DISPLAY = {
    "pretrained": "ViT pretrained",
    "pretrained_pre_vq": "ViT pretrained pre-VQ",
    "random":     "ViT random",
    "random_cnn": "CNN random",
}
METRICS = ("auroc", "accuracy", "precision", "recall", "f1")


# ── split ─────────────────────────────────────────────────────────────────────

def condition_split(accessions: list, valid_mask: torch.Tensor, seed: int):
    """
    70/15/15 split on the subset of scans with explicit labels for a condition.
    Returns (train_idx, val_idx, test_idx) as index arrays into the full N.
    """
    valid_idx = torch.where(valid_mask)[0].tolist()
    valid_acc = [accessions[i] for i in valid_idx]

    sorted_acc = sorted(set(valid_acc))
    rng = np.random.RandomState(seed)
    rng.shuffle(sorted_acc)

    n      = len(sorted_acc)
    n_tr   = int(0.70 * n)
    n_val  = int(0.15 * n)
    train_set = set(sorted_acc[:n_tr])
    val_set   = set(sorted_acc[n_tr: n_tr + n_val])
    test_set  = set(sorted_acc[n_tr + n_val:])

    tr, va, te = [], [], []
    for i, acc in zip(valid_idx, valid_acc):
        if acc in train_set:   tr.append(i)
        elif acc in val_set:   va.append(i)
        else:                  te.append(i)
    return tr, va, te


# ── probe ─────────────────────────────────────────────────────────────────────

def train_probe(tf, tl, vf, vl, device, epochs, seed,
                patience=10, eval_every=20):
    torch.manual_seed(seed)
    probe = nn.Linear(tf.shape[1], 1).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=1e-3)
    crit  = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(tf, tl), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    best_auroc, best_state, no_imp = -1.0, {k: v.clone() for k, v in probe.state_dict().items()}, 0
    for epoch in range(1, epochs + 1):
        probe.train()
        for xb, yb in loader:
            loss = nn.functional.binary_cross_entropy_with_logits(
                probe(xb.to(device)).squeeze(1), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()

        if epoch % eval_every == 0 or epoch == epochs:
            probe.eval()
            with torch.no_grad():
                vl_sig = torch.sigmoid(probe(vf.to(device)).squeeze(1)).cpu().numpy()
            try:
                auroc = float(roc_auc_score(vl.numpy(), vl_sig))
            except ValueError:
                auroc = 0.5
            if auroc > best_auroc:
                best_auroc, best_state, no_imp = auroc, {k: v.clone() for k, v in probe.state_dict().items()}, 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

    probe.load_state_dict(best_state)
    return probe, epoch


def compute_metrics(probe, feats, labels, device):
    probe.eval()
    with torch.no_grad():
        probs = torch.sigmoid(probe(feats.to(device)).squeeze(1)).cpu().numpy()
    y     = labels.numpy().astype(int)
    preds = (probs >= 0.5).astype(int)
    return {
        "auroc":     float(roc_auc_score(y, probs)),
        "accuracy":  float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall":    float(recall_score(y, preds, zero_division=0)),
        "f1":        float(f1_score(y, preds, zero_division=0)),
    }


def run_n(tf_all, tl_all, vf, vl, xf, xl, n_train, seeds, device, epochs):
    per_seed = {m: [] for m in METRICS}
    per_seed["stopped_epoch"] = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = torch.from_numpy(
            rng.choice(len(tf_all), min(n_train, len(tf_all)), replace=False))
        probe, ep = train_probe(tf_all[idx], tl_all[idx], vf, vl,
                                device, epochs, seed)
        per_seed["stopped_epoch"].append(ep)
        m = compute_metrics(probe, xf, xl, device)
        for k in METRICS:
            per_seed[k].append(m[k])
    return per_seed


# ── main ──────────────────────────────────────────────────────────────────────

def _fmt(mean, std):
    return f"{mean:.4f} ±{std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir",   default=str(CACHE_DIR))
    parser.add_argument("--conditions",  default=DEFAULT_CONDITIONS)
    parser.add_argument("--epochs",      type=int, default=1000)
    parser.add_argument("--seeds",       type=int, default=5)
    parser.add_argument("--ns",          default="100,300,all")
    parser.add_argument("--split_seed",  type=int, default=42)
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds      = list(range(args.seeds))
    cache      = Path(args.cache_dir)
    conditions = [c.strip() for c in args.conditions.split(",")]

    # ── load shared metadata ───────────────────────────────────────────────
    with open(cache / "accessions.json")  as f: accessions  = json.load(f)
    with open(cache / "label_names.json") as f: label_names = json.load(f)
    all_labels = torch.load(cache / "labels.pt", map_location="cpu")  # [N, 30]
    print(f"Full cache: {len(accessions)} scans, {len(label_names)} conditions")

    # ── load features for available backbones ─────────────────────────────
    candidates = ["pretrained", "pretrained_pre_vq", "random", "random_cnn"]
    backbones  = [b for b in candidates if (cache / b / "feats.pt").exists()]
    feats_all  = {}
    for b in backbones:
        feats_all[b] = torch.load(cache / b / "feats.pt", map_location="cpu")
        print(f"  {DISPLAY[b]:<18}  {feats_all[b].shape}  dim={feats_all[b].shape[1]}")

    # ── build N list (use first backbone as reference for n_train_total) ──
    # actual per-condition N determined after filtering; use None as placeholder
    ns_raw = [x.strip() for x in args.ns.split(",")]

    all_results = {}

    for cond in conditions:
        if cond not in label_names:
            print(f"\n[skip] '{cond}' not in label file")
            continue

        cond_idx  = label_names.index(cond)
        cond_lbl  = all_labels[:, cond_idx]          # [N]
        valid_mask = (cond_lbl == 0) | (cond_lbl == 1)
        n_valid   = valid_mask.sum().item()

        print(f"\n{'='*72}")
        print(f"Condition: {cond}  ({n_valid} labeled scans)")
        print(f"{'='*72}")

        tr_idx, va_idx, te_idx = condition_split(
            accessions, valid_mask, args.split_seed)
        n_tr = len(tr_idx)
        print(f"  split → train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

        # build concrete N list now that we know n_train
        ns = sorted({n_tr if x == "all" else int(x)
                     for x in ns_raw if x == "all" or int(x) <= n_tr})

        tr_idx_t = torch.tensor(tr_idx)
        va_idx_t = torch.tensor(va_idx)
        te_idx_t = torch.tensor(te_idx)
        tr_lbl = cond_lbl[tr_idx_t]
        va_lbl = cond_lbl[va_idx_t]
        te_lbl = cond_lbl[te_idx_t]

        all_results[cond] = {}

        for b in backbones:
            print(f"\n  --- {DISPLAY[b]} ---")
            tf_all = feats_all[b][tr_idx_t]
            vf     = feats_all[b][va_idx_t]
            xf     = feats_all[b][te_idx_t]
            all_results[cond][b] = {}

            for n in ns:
                per_seed = run_n(tf_all, tr_lbl, vf, va_lbl, xf, te_lbl,
                                 n, seeds, device, args.epochs)
                all_results[cond][b][n] = per_seed
                mean_a = float(np.mean(per_seed["auroc"]))
                std_a  = float(np.std(per_seed["auroc"]))
                ep_str = "  ".join(str(e) for e in per_seed["stopped_epoch"])
                a_str  = "  ".join(f"{a:.4f}" for a in per_seed["auroc"])
                print(f"    N={n:6d}  AUROC {mean_a:.4f} ±{std_a:.4f}   [{a_str}]")
                print(f"            stopped at epoch: [{ep_str}]")

        # ── per-condition AUROC summary ────────────────────────────────────
        col_w = 18
        print(f"\n  AUROC summary — {cond}")
        header = f"  {'N_train':>8}" + "".join(f"  {DISPLAY[b]:>{col_w}}" for b in backbones)
        print(header)
        print(f"  {'-'*8}" + f"  {'-'*col_w}" * len(backbones))
        ref_b = "pretrained" if "pretrained" in backbones else backbones[0]
        for n in ns:
            row = f"  {n:>8d}"
            for b in backbones:
                mean_a = float(np.mean(all_results[cond][b][n]["auroc"]))
                std_a  = float(np.std(all_results[cond][b][n]["auroc"]))
                row += f"  {_fmt(mean_a, std_a):>{col_w}}"
            print(row)

    # ── cross-condition summary (full N only) ─────────────────────────────
    print(f"\n{'='*72}")
    print("Cross-condition AUROC  (N=all, full training set)")
    print(f"{'='*72}")
    col_w = 18
    header = f"  {'Condition':<32}" + "".join(f"  {DISPLAY[b]:>{col_w}}" for b in backbones)
    print(header)
    print(f"  {'-'*32}" + f"  {'-'*col_w}" * len(backbones))
    for cond in conditions:
        if cond not in all_results:
            continue
        cond_res = all_results[cond]
        max_n    = max(cond_res[backbones[0]].keys())
        row = f"  {cond:<32}"
        for b in backbones:
            mean_a = float(np.mean(cond_res[b][max_n]["auroc"]))
            std_a  = float(np.std(cond_res[b][max_n]["auroc"]))
            row += f"  {_fmt(mean_a, std_a):>{col_w}}"
        print(row)

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
        json.dump({"conditions": conditions, "backbones": backbones,
                   "epochs": args.epochs, "seeds": seeds,
                   "results": serialise(all_results)}, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
