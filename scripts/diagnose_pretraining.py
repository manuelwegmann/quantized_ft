"""
Diagnostic script for SimSiam pretraining health.

Loads saved checkpoints from one or more pretrain output directories and
computes three metrics at each checkpoint epoch:

  1. Uniformity   — log mean pairwise Gaussian kernel on the unit sphere
                    (Wang & Isola 2020).  More negative = better spread.
                    Approaches 0 under dimensional collapse.

  2. Effective rank — exp(Shannon entropy of normalised singular-value
                    spectrum).  Full rank = 512 for our backbone.
                    Drops toward 1 under dimensional collapse.

  3. kNN macro-AUROC — k-nearest-neighbour classifier (cosine similarity)
                    on a small labelled eval split. Directly measures
                    whether the geometry is useful for downstream tasks.

Epoch 0 is always the base CT-CLIP checkpoint (before any SimSiam fine-tuning)
and serves as the reference point.

Usage (minimal):
    python scripts/diagnose_pretraining.py \\
        --checkpoint_dirs runs/mini_experiment_ln/pretrain_fp \\
                          runs/mini_experiment_ln/pretrain_ssql

Usage (full options):
    python scripts/diagnose_pretraining.py \\
        --checkpoint_dirs runs/mini_experiment_ln/pretrain_fp \\
        --n_eval 500 --k 20 --batch_size 4 --num_workers 2 \\
        --conditions "Pleural Effusion,Consolidation" \\
        --output runs/mini_experiment_ln/diagnostics.json

Notes:
  • Only checkpoints named checkpoint_ep*.pt and checkpoint_final.pt are
    scanned.  checkpoint_latest.pt is ignored (epoch number ambiguous).
  • To get per-epoch curves, re-run run_mini_experiment.py with
    --save_every 5 (or lower).  Currently save_every defaults to epochs,
    so only checkpoint_final.pt is written.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))          # scripts/ — for cache_all_features
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/transformer_maskgit")))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/CT_CLIP")))

from models.backbone import CTViTBackbone
from cache_all_features import AllScansDataset

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    print("[warn] sklearn not found — kNN AUROC will be skipped")

CHECKPOINT   = "/home/nlr950/Dir/CT-CLIP/checkpoints/CT-CLIP_v2.pt"
DATA_FOLDER  = "/home/nlr950/Dir/CT-CLIP/data/merlin_data"
REPORTS_FILE = "/home/nlr950/Dir/CT-CLIP/data/reports_final.xlsx"
LABELS_FILE  = "/home/nlr950/Dir/CT-CLIP/data/zero_shot_findings_disease_cls.csv"
META_FILE    = "/home/nlr950/Dir/CT-CLIP/data/metadata.csv"

DEFAULT_CONDITIONS = (
    "atelectasis,surgically_absent_gallbladder,renal_cyst,"
    "pleural_effusion,cardiomegaly,gallstones"
)


# ── metrics ───────────────────────────────────────────────────────────────────

def uniformity(features: torch.Tensor) -> float:
    """
    log E[exp(-2 ||z_i - z_j||^2)]  (Wang & Isola 2020).
    Input: (N, D) raw features — L2-normalised internally.
    Returns a scalar <= 0; closer to 0 means more collapsed.
    """
    z = F.normalize(features.float(), dim=1)
    sq_dists = torch.pdist(z, p=2).pow(2)
    return sq_dists.mul(-2).exp().mean().log().item()


def effective_rank(features: torch.Tensor) -> float:
    """
    exp(H(σ)) where σ is the normalised singular-value spectrum.
    Equals 512 (full rank) for perfectly spread features; 1 for collapsed.
    """
    f = features.float()
    f = f - f.mean(dim=0)
    s = torch.linalg.svdvals(f)
    s = s / (s.sum() + 1e-12)
    entropy = -(s * (s + 1e-12).log()).sum()
    return entropy.exp().item()


def knn_auroc(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    condition_indices: list[int],
    k: int = 20,
) -> float:
    """
    Cosine kNN macro-AUROC over the specified label columns.
    Labels may contain -1 (unknown); only 0/1 entries are used per condition.
    Returns NaN if no condition has enough valid samples.
    """
    if not _SKLEARN:
        return float("nan")

    tr_z = F.normalize(train_feats.float(), dim=1)
    te_z = F.normalize(test_feats.float(), dim=1)

    sim = te_z @ tr_z.T                             # (N_test, N_train)
    k_actual = min(k, tr_z.shape[0])
    topk_idx = sim.topk(k_actual, dim=1).indices    # (N_test, k)

    pred_probs = train_labels[topk_idx].float().mean(dim=1)  # (N_test, C_sub)

    aurocs = []
    for ci, c in enumerate(condition_indices):
        y_true  = test_labels[:, ci].numpy()
        y_score = pred_probs[:, ci].numpy()
        valid   = (y_true == 0) | (y_true == 1)
        if valid.sum() < 10 or y_true[valid].sum() == 0 or (1 - y_true[valid]).sum() == 0:
            continue
        aurocs.append(roc_auc_score(y_true[valid], y_score[valid]))

    return float(np.mean(aurocs)) if aurocs else float("nan")


# ── feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(backbone: CTViTBackbone, loader: DataLoader,
                     device: torch.device) -> torch.Tensor:
    backbone.eval()
    feats = []
    for x, _ in loader:
        feats.append(backbone(x.to(device)).cpu())
    return torch.cat(feats)


# ── checkpoint discovery ──────────────────────────────────────────────────────

def find_checkpoints(pretrain_dir: Path) -> list[tuple[int, Path]]:
    """
    Returns [(epoch, path), ...] sorted by epoch.
    Scans checkpoint_ep*.pt and checkpoint_final.pt.
    """
    ckpts = []
    for p in pretrain_dir.glob("checkpoint_ep*.pt"):
        m = re.search(r"checkpoint_ep(\d+)\.pt", p.name)
        if m:
            ckpts.append((int(m.group(1)), p))

    final = pretrain_dir / "checkpoint_final.pt"
    if final.exists():
        # Infer epoch from the latest known ep checkpoint, or from the
        # checkpoint metadata itself.
        try:
            ep = torch.load(final, map_location="cpu")["epoch"]
        except Exception:
            ep = max((e for e, _ in ckpts), default=0) + 1
        # Only add if not already covered by an ep-numbered file
        existing_epochs = {e for e, _ in ckpts}
        if ep not in existing_epochs:
            ckpts.append((ep, final))

    return sorted(ckpts)


def load_backbone_from_ckpt(ckpt_path: Path, device: torch.device) -> CTViTBackbone:
    backbone = CTViTBackbone(checkpoint_path=None, use_pre_vq=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone.load_state_dict(ckpt["backbone"])
    backbone.to(device)
    return backbone


# ── per-run evaluation ────────────────────────────────────────────────────────

def evaluate_run(
    pretrain_dir: Path,
    eval_loader: DataLoader,
    all_labels: torch.Tensor,
    condition_indices: list[int],
    knn_train_idx: torch.Tensor,
    knn_test_idx: torch.Tensor,
    device: torch.device,
    k: int,
) -> dict:
    """
    Evaluates epoch-0 (base CT-CLIP) + all saved checkpoints in pretrain_dir.
    Returns a dict mapping epoch -> {uniformity, effective_rank, knn_auroc}.
    """
    results = {}

    # Epoch 0: base CT-CLIP weights
    print("  [epoch 0] base CT-CLIP backbone")
    backbone0 = CTViTBackbone(checkpoint_path=CHECKPOINT, use_pre_vq=True).to(device)
    feats0 = extract_features(backbone0, eval_loader, device)
    del backbone0
    torch.cuda.empty_cache()

    results[0] = _compute_metrics(feats0, all_labels, condition_indices,
                                  knn_train_idx, knn_test_idx, k)
    _print_row(0, results[0])

    # One checkpoint at a time to keep GPU memory low
    ckpts = find_checkpoints(pretrain_dir)
    if not ckpts:
        print(f"  [warn] no checkpoints found in {pretrain_dir}")
        return results

    for epoch, ckpt_path in ckpts:
        print(f"  [epoch {epoch:3d}] {ckpt_path.name}")
        backbone = load_backbone_from_ckpt(ckpt_path, device)
        feats = extract_features(backbone, eval_loader, device)
        del backbone
        torch.cuda.empty_cache()

        results[epoch] = _compute_metrics(feats, all_labels, condition_indices,
                                          knn_train_idx, knn_test_idx, k)
        _print_row(epoch, results[epoch])

    return results


def _compute_metrics(
    feats: torch.Tensor,
    all_labels: torch.Tensor,
    condition_indices: list[int],
    knn_train_idx: torch.Tensor,
    knn_test_idx: torch.Tensor,
    k: int,
) -> dict:
    uni  = uniformity(feats)
    rank = effective_rank(feats)

    tr_feats  = feats[knn_train_idx]
    te_feats  = feats[knn_test_idx]
    tr_labels = all_labels[knn_train_idx][:, condition_indices]
    te_labels = all_labels[knn_test_idx][:, condition_indices]
    auroc = knn_auroc(tr_feats, tr_labels, te_feats, te_labels,
                      list(range(len(condition_indices))), k)

    return {"uniformity": uni, "effective_rank": rank, "knn_auroc": auroc}


def _print_row(epoch: int, m: dict):
    auroc_str = f"{m['knn_auroc']:.4f}" if not np.isnan(m["knn_auroc"]) else "  n/a "
    print(f"    uniformity={m['uniformity']:7.4f}  "
          f"eff_rank={m['effective_rank']:6.1f}  "
          f"knn_auroc={auroc_str}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose SimSiam pretraining health via feature geometry metrics."
    )
    parser.add_argument(
        "--checkpoint_dirs", nargs="+", required=True,
        help="One or more pretrain output dirs (each contains checkpoint_*.pt files).",
    )
    parser.add_argument("--n_eval",      type=int,   default=500,
                        help="Number of scans to use for evaluation (default 500).")
    parser.add_argument("--k",           type=int,   default=20,
                        help="Number of neighbours for kNN (default 20).")
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--conditions",  type=str,   default=DEFAULT_CONDITIONS)
    parser.add_argument("--output",      type=str,   default=None,
                        help="Path to save results JSON (default: <first_dir>/diagnostics.json).")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditions = [c.strip() for c in args.conditions.split(",")]

    # ── build eval dataset ────────────────────────────────────────────────────
    print(f"Loading eval dataset (up to {args.n_eval} scans)…")
    full_ds = AllScansDataset(DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE)

    # Deterministic subset
    n = min(args.n_eval, len(full_ds))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(full_ds), size=n, replace=False)
    indices.sort()
    eval_ds = Subset(full_ds, indices.tolist())

    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Labels for the subset
    all_labels_full = torch.tensor(
        np.stack([lbl for _, lbl in full_ds.samples]), dtype=torch.float32
    )
    all_labels = all_labels_full[torch.from_numpy(indices)]

    # Condition indices
    label_names = full_ds.label_names
    condition_indices = []
    for c in conditions:
        if c in label_names:
            condition_indices.append(label_names.index(c))
        else:
            print(f"[warn] condition '{c}' not found in label file — skipping")
    if not condition_indices:
        print("[warn] no valid conditions — kNN AUROC will be NaN")

    # Fixed 80/20 train/test split for kNN
    rng2 = np.random.default_rng(0)
    perm = rng2.permutation(n)
    n_train = int(0.8 * n)
    knn_train_idx = torch.from_numpy(perm[:n_train])
    knn_test_idx  = torch.from_numpy(perm[n_train:])

    print(f"Eval set: {n} scans  kNN split: {n_train} train / {n - n_train} test")
    print(f"Conditions: {conditions}\n")

    # ── evaluate each run ─────────────────────────────────────────────────────
    all_results = {}
    for ckpt_dir_str in args.checkpoint_dirs:
        ckpt_dir = Path(ckpt_dir_str)
        run_name = ckpt_dir.name
        print(f"{'='*60}\nRun: {run_name}  ({ckpt_dir})\n{'='*60}")
        all_results[run_name] = evaluate_run(
            ckpt_dir, eval_loader, all_labels, condition_indices,
            knn_train_idx, knn_test_idx, device, args.k,
        )

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nSummary (epoch 0 → final)\n{'='*60}")
    for run_name, run_res in all_results.items():
        epochs_sorted = sorted(run_res.keys())
        if len(epochs_sorted) < 2:
            continue
        e0, ef = epochs_sorted[0], epochs_sorted[-1]
        m0, mf = run_res[e0], run_res[ef]
        print(f"\n{run_name}:")
        print(f"  uniformity:    {m0['uniformity']:7.4f} → {mf['uniformity']:7.4f}  "
              f"(Δ {mf['uniformity'] - m0['uniformity']:+.4f})")
        print(f"  eff_rank:      {m0['effective_rank']:6.1f} → {mf['effective_rank']:6.1f}  "
              f"(Δ {mf['effective_rank'] - m0['effective_rank']:+.1f})")
        if not (np.isnan(m0["knn_auroc"]) or np.isnan(mf["knn_auroc"])):
            print(f"  knn_auroc:     {m0['knn_auroc']:.4f} → {mf['knn_auroc']:.4f}  "
                  f"(Δ {mf['knn_auroc'] - m0['knn_auroc']:+.4f})")

    # ── save JSON ─────────────────────────────────────────────────────────────
    out_path = (
        Path(args.output) if args.output
        else Path(args.checkpoint_dirs[0]).parent / "diagnostics.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "conditions": conditions,
                "n_eval": n,
                "k": args.k,
                "results": {
                    run: {str(ep): m for ep, m in epochs.items()}
                    for run, epochs in all_results.items()
                },
            },
            f, indent=2,
        )
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
