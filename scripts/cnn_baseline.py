"""
Random CNN baseline for atelectasis classification.

Extracts features from a randomly initialised lightweight 3D CNN, trains a
linear probe at the same N_train values as the main experiment, and prints
AUROC alongside the pretrained CT-CLIP ViT results from results.json.

Self-contained except for MerlinDataset (data loading only, same seed/split).
Features are cached to runs/feature_cache/random_cnn/ and reused on re-runs.

Usage:
    python scripts/cnn_baseline.py
    python scripts/cnn_baseline.py --force_extract   # re-extract even if cache exists
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/transformer_maskgit")))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/CT_CLIP")))

from downstream.dataset import MerlinDataset

# ── paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT   = "/home/nlr950/Dir/CT-CLIP/checkpoints/CT-CLIP_v2.pt"
DATA_FOLDER  = "/home/nlr950/Dir/CT-CLIP/data/merlin_data"
REPORTS_FILE = "/home/nlr950/Dir/CT-CLIP/data/reports_final.xlsx"
LABELS_FILE  = "/home/nlr950/Dir/CT-CLIP/data/zero_shot_findings_disease_cls.csv"
META_FILE    = "/home/nlr950/Dir/CT-CLIP/data/metadata.csv"
CACHE_DIR    = _PROJECT_ROOT / "runs" / "feature_cache" / "random_cnn"
RESULTS_JSON = _PROJECT_ROOT / "runs" / "learning_curve" / "results.json"

SEED         = 42
CNN_DIM      = 256


# ── model ─────────────────────────────────────────────────────────────────────

class RandomCNN3D(nn.Module):
    """
    Lightweight randomly-initialised 3D CNN.
    Input  : (B, 1, D, H, W)  — same format as CTViTBackbone
    Output : (B, CNN_DIM)

    Spatial reduction per layer (stride 4→4→2→2):
      (1, 240, 480, 480) → (32, 60, 120, 120) → (64, 15, 30, 30)
                         → (128, 8, 15, 15)   → (256, 4, 8, 8) → pool → (256,)
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1,   32,  kernel_size=7, stride=4, padding=3), nn.BatchNorm3d(32),  nn.ReLU(),
            nn.Conv3d(32,  64,  kernel_size=5, stride=4, padding=2), nn.BatchNorm3d(64),  nn.ReLU(),
            nn.Conv3d(64,  128, kernel_size=3, stride=2, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, CNN_DIM, kernel_size=3, stride=2, padding=1), nn.BatchNorm3d(CNN_DIM), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(device: torch.device, num_workers: int, batch_size: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    model = RandomCNN3D().to(device).eval()
    print(f"CNN parameters: {sum(p.numel() for p in model.parameters()):,}")

    for split in ("train", "val", "test"):
        ds = MerlinDataset(
            data_folder=DATA_FOLDER, reports_file=REPORTS_FILE,
            labels_file=LABELS_FILE, meta_file=META_FILE,
            label_cols=["atelectasis"], require_labeled=True,
            split=split, seed=SEED,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        feats, labels = [], []
        for x, y in tqdm.tqdm(loader, desc=f"  {split}"):
            with torch.no_grad():
                feats.append(model(x.to(device)).cpu())
            labels.append(y)
        torch.save(torch.cat(feats),  CACHE_DIR / f"{split}_feats.pt")
        torch.save(torch.cat(labels), CACHE_DIR / f"{split}_labels.pt")
        print(f"    saved {split}: {torch.cat(feats).shape}")


# ── probe ─────────────────────────────────────────────────────────────────────

def train_probe(tf, tl, vf, vl, device, epochs, seed,
                patience=10, eval_every=20):
    torch.manual_seed(seed)
    probe = nn.Linear(CNN_DIM, 1).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=1e-3)
    crit  = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(tf, tl), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    best_auroc, best_state, no_imp = -1.0, {k: v.clone() for k, v in probe.state_dict().items()}, 0

    for epoch in range(1, epochs + 1):
        probe.train()
        for xb, yb in loader:
            loss = crit(probe(xb.to(device)).squeeze(1), yb[:, 0].to(device))
            opt.zero_grad(); loss.backward(); opt.step()

        if epoch % eval_every == 0 or epoch == epochs:
            probe.eval()
            with torch.no_grad():
                vl_sig = torch.sigmoid(probe(vf.to(device)).squeeze(1)).cpu().numpy()
            try:
                auroc = float(roc_auc_score(vl[:, 0].numpy(), vl_sig))
            except ValueError:
                auroc = 0.5
            if auroc > best_auroc:
                best_auroc = auroc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

    probe.load_state_dict(best_state)
    return probe, epoch


def eval_auroc(probe, feats, labels, device):
    probe.eval()
    with torch.no_grad():
        probs = torch.sigmoid(probe(feats.to(device)).squeeze(1)).cpu().numpy()
    return float(roc_auc_score(labels[:, 0].numpy(), probs))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int, default=300)
    parser.add_argument("--seeds",        type=int, default=5)
    parser.add_argument("--ns",           default="30,100,300,1000,all")
    parser.add_argument("--num_workers",  type=int, default=8)
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--force_extract", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    # ── feature extraction (skip if cache exists) ──────────────────────────
    if args.force_extract or not (CACHE_DIR / "train_feats.pt").exists():
        print("\nExtracting CNN features...")
        extract_features(device, args.num_workers, args.batch_size)
    else:
        print(f"\nUsing cached CNN features from {CACHE_DIR}")

    # ── load features ──────────────────────────────────────────────────────
    splits = {
        s: (torch.load(CACHE_DIR / f"{s}_feats.pt",  map_location="cpu"),
            torch.load(CACHE_DIR / f"{s}_labels.pt", map_location="cpu"))
        for s in ("train", "val", "test")
    }
    n_train_total = len(splits["train"][0])
    print(f"  train={n_train_total}  val={len(splits['val'][0])}  test={len(splits['test'][0])}")

    # ── build N list ───────────────────────────────────────────────────────
    ns = sorted({n_train_total if x.strip() == "all" else int(x)
                 for x in args.ns.split(",") if x.strip() == "all" or int(x) <= n_train_total})
    seeds = list(range(args.seeds))
    print(f"N_train sweep : {ns}  |  seeds: {seeds}  |  max epochs: {args.epochs}")

    tf_all, tl_all = splits["train"]
    vf, vl         = splits["val"]
    xf, xl         = splits["test"]

    # ── probe sweep ────────────────────────────────────────────────────────
    cnn_results = {}
    print("\n=== random CNN ===")
    for n in ns:
        aurocs, stopped = [], []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            idx = torch.from_numpy(rng.choice(len(tf_all), min(n, len(tf_all)), replace=False))
            probe, ep = train_probe(tf_all[idx], tl_all[idx], vf, vl,
                                    device, args.epochs, seed)
            aurocs.append(eval_auroc(probe, xf, xl, device))
            stopped.append(ep)
        cnn_results[n] = {"auroc": aurocs, "stopped_epoch": stopped}
        mean_a, std_a = float(np.mean(aurocs)), float(np.std(aurocs))
        auroc_str   = "  ".join(f"{a:.4f}" for a in aurocs)
        stopped_str = "  ".join(str(e) for e in stopped)
        print(f"  N={n:6d}  AUROC {mean_a:.4f} ±{std_a:.4f}   [{auroc_str}]")
        print(f"          stopped at epoch: [{stopped_str}]")

    # ── comparison table ───────────────────────────────────────────────────
    vit_results = None
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            vit_results = json.load(f)["results"]

    print(f"\n{'='*72}")
    print("AUROC comparison (test set)")
    print(f"{'='*72}")
    header = f"  {'N_train':>8}  {'CNN (random)':>16}"
    if vit_results:
        header += f"  {'ViT (pretrained)':>16}  {'ViT (random)':>16}"
    print(header)
    print(f"  {'-'*8}  {'-'*16}" + (f"  {'-'*16}  {'-'*16}" if vit_results else ""))
    for n in ns:
        cm = float(np.mean(cnn_results[n]["auroc"]))
        cs = float(np.std(cnn_results[n]["auroc"]))
        row = f"  {n:>8d}  {cm:.4f} ±{cs:.4f}"
        if vit_results:
            sn = str(n)
            if sn in vit_results.get("pretrained", {}):
                pm = float(np.mean(vit_results["pretrained"][sn]["auroc"]))
                ps = float(np.std(vit_results["pretrained"][sn]["auroc"]))
                rm = float(np.mean(vit_results["random"][sn]["auroc"]))
                rs = float(np.std(vit_results["random"][sn]["auroc"]))
                row += f"  {pm:.4f} ±{ps:.4f}  {rm:.4f} ±{rs:.4f}"
        print(row)
    print(f"{'='*72}")

    out = CACHE_DIR.parent.parent / "learning_curve" / "cnn_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"ns": ns, "epochs": args.epochs, "seeds": seeds,
                   "results": {"random_cnn": {str(k): {
                       "auroc": [float(x) for x in v["auroc"]],
                       "stopped_epoch": v["stopped_epoch"],
                   } for k, v in cnn_results.items()}}}, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
