"""
Mini end-to-end experiment: FP SimSiam → SSQL SimSiam → linear probes.

  1. Pretrain FP backbone on N_pretrain scans for EPOCHS epochs.
  2. Extract features for the full labeled dataset (FP and quantized).
  3. Pretrain SSQL backbone on the same N_pretrain scans.
  4. Extract features (FP and quantized).
  5. Linear probe on all four feature sets.

Both backbones start from the same CT-CLIP_v2.pt checkpoint.
Resume logic is handled automatically: if checkpoint_latest.pt exists in the
pretrain output directory, training continues from the last completed epoch.

Usage:
    python scripts/run_mini_experiment.py
    python scripts/run_mini_experiment.py --n_pretrain 300 --epochs 30
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/transformer_maskgit")))
sys.path.insert(0, str(Path("/home/nlr950/Dir/CT-CLIP/CT_CLIP")))
sys.path.insert(0, str(Path(__file__).parent))

from models.backbone import CTViTBackbone
from models.simsiam import Projector, Predictor
from models.quantization import quantized_forward
from pretrain.augmentations import CTAugmentation
from pretrain.dataset import CTPretrainDataset
from pretrain.trainer import train as pretrain_train
from cache_all_features import AllScansDataset
import run_multi_condition as rmc

CHECKPOINT   = "/home/nlr950/Dir/CT-CLIP/checkpoints/CT-CLIP_v2.pt"
DATA_FOLDER  = "/home/nlr950/Dir/CT-CLIP/data/merlin_data"
REPORTS_FILE = "/home/nlr950/Dir/CT-CLIP/data/reports_final.xlsx"
LABELS_FILE  = "/home/nlr950/Dir/CT-CLIP/data/zero_shot_findings_disease_cls.csv"
META_FILE    = "/home/nlr950/Dir/CT-CLIP/data/metadata.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_backbone():
    return CTViTBackbone(checkpoint_path=CHECKPOINT, use_pre_vq=True)

def _build_projector(norm: str = 'bn'):
    return Projector(in_dim=512, hidden_dim=2048, out_dim=2048, norm=norm)

def _build_predictor(norm: str = 'bn'):
    return Predictor(in_dim=2048, hidden_dim=512, out_dim=2048, norm=norm)


def _extract(backbone, loader, device, w_bits=None, a_bits=None):
    backbone.eval()
    feats = []
    for x, _ in tqdm.tqdm(loader, unit="batch", leave=False):
        with torch.no_grad():
            if w_bits is not None:
                with quantized_forward([backbone], w_bits, a_bits):
                    feats.append(backbone(x.to(device)).cpu())
            else:
                feats.append(backbone(x.to(device)).cpu())
    return torch.cat(feats)


def _pretrain_cfg(mode, output_dir, epochs, num_workers, lr, save_every=0, batch_size=2,
                  freeze_epochs=0):
    cfg = {
        "mode": mode,
        "use_aux_loss": True,
        "backbone":  {"checkpoint": CHECKPOINT, "use_pre_vq": True},
        "training":  {
            "batch_size": batch_size, "lr": lr, "momentum": 0.9,
            "weight_decay": 1e-4, "epochs": epochs,
            "num_workers": num_workers,
            "save_every": save_every if save_every > 0 else epochs,
            "lr_schedule": "cosine",
            "freeze_epochs": freeze_epochs,
        },
        "projector":    {"in_dim": 512,  "hidden_dim": 2048, "out_dim": 2048},
        "predictor":    {"in_dim": 2048, "hidden_dim": 512,  "out_dim": 2048},
        "quantization": {"w_bits_min": 2, "w_bits_max": 8,
                         "a_bits_min": 4, "a_bits_max": 8},
        "output_dir": str(output_dir),
    }
    return cfg


def _cache_features(backbone, loader, device, feat_dir, fp_name, quant_name, w_bits, a_bits):
    """Extract FP and quantized features into feat_dir/{name}/feats.pt; skip if present."""
    for name, wb, ab in [(fp_name, None, None), (quant_name, w_bits, a_bits)]:
        fdir = feat_dir / name
        if (fdir / "feats.pt").exists():
            print(f"    skip {name} — already cached")
            continue
        fdir.mkdir(parents=True, exist_ok=True)
        feats = _extract(backbone, loader, device, wb, ab)
        torch.save(feats, fdir / "feats.pt")
        print(f"    {name}  {feats.shape}")


# ── probe ─────────────────────────────────────────────────────────────────────

def _run_probes(feat_dir, accessions, label_names, all_labels,
                conditions, seeds, device, probe_epochs):
    backbones = sorted(d.name for d in feat_dir.iterdir()
                       if d.is_dir() and (d / "feats.pt").exists())
    feats_all = {b: torch.load(feat_dir / b / "feats.pt", map_location="cpu")
                 for b in backbones}

    results = {}
    for cond in conditions:
        if cond not in label_names:
            print(f"[probe] skip '{cond}' — not in label file")
            continue

        cond_idx   = label_names.index(cond)
        cond_lbl   = all_labels[:, cond_idx]
        valid_mask = (cond_lbl == 0) | (cond_lbl == 1)
        n_valid    = valid_mask.sum().item()
        print(f"\n{'='*60}\nCondition: {cond}  ({n_valid} labeled)")

        tr_idx, va_idx, te_idx = rmc.condition_split(accessions, valid_mask, seed=42)
        n_tr = len(tr_idx)
        print(f"  split → train={n_tr}  val={len(va_idx)}  test={len(te_idx)}")

        ns = sorted({n_tr if x == "all" else int(x)
                     for x in ["100", "all"] if x == "all" or int(x) <= n_tr})

        tr_t, va_t, te_t = map(torch.tensor, [tr_idx, va_idx, te_idx])
        tr_lbl = cond_lbl[tr_t]
        va_lbl = cond_lbl[va_t]
        te_lbl = cond_lbl[te_t]

        results[cond] = {}
        for b in backbones:
            print(f"  --- {b} ---")
            results[cond][b] = {}
            for n in ns:
                per_seed = rmc.run_n(
                    feats_all[b][tr_t], tr_lbl,
                    feats_all[b][va_t], va_lbl,
                    feats_all[b][te_t], te_lbl,
                    n, seeds, device, probe_epochs,
                )
                results[cond][b][n] = per_seed
                mean_a = float(np.mean(per_seed["auroc"]))
                std_a  = float(np.std(per_seed["auroc"]))
                print(f"    N={n:6d}  AUROC {mean_a:.4f} ±{std_a:.4f}")

    return results, backbones


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pretrain",   type=int, default=300)
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--save_every",   type=int, default=0,
                        help="Save a numbered checkpoint every N epochs. 0 = final only.")
    parser.add_argument("--probe_epochs", type=int, default=500)
    parser.add_argument("--seeds",        type=int, default=3)
    parser.add_argument("--quant_w",      type=int, default=4)
    parser.add_argument("--quant_a",      type=int, default=4)
    parser.add_argument("--batch_size",   type=int, default=2,
                        help="Pretraining batch size. LR is auto-scaled via "
                             "0.05*batch_size/256 unless --lr is set explicitly.")
    parser.add_argument("--lr",           type=float, default=None,
                        help="Override learning rate. Default: 0.05*batch_size/256.")
    parser.add_argument("--num_workers",      type=int, default=4)
    parser.add_argument("--num_workers_eval", type=int, default=2)
    parser.add_argument("--norm",         default="ln", choices=["bn", "ln"])
    parser.add_argument("--freeze_epochs", type=int, default=0,
                        help="Freeze backbone for this many epochs, then unfreeze. 0 = never frozen.")
    parser.add_argument("--conditions",   default=rmc.DEFAULT_CONDITIONS)
    parser.add_argument("--output_dir",   default="runs/mini_experiment")
    parser.add_argument("--phase",        default="all",
                        choices=["all", "pretrain_fp", "pretrain_ssql", "probe"],
                        help="Which phase to run. Use pretrain_fp / pretrain_ssql "
                             "to run FP and SSQL as separate parallel jobs, then "
                             "'probe' once both feature caches are ready.")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out        = Path(args.output_dir)
    feat_dir   = out / "feature_cache"
    conditions = [c.strip() for c in args.conditions.split(",")]
    seeds      = list(range(args.seeds))
    w, a       = args.quant_w, args.quant_a
    lr         = args.lr if args.lr is not None else 0.05 * args.batch_size / 256

    run_fp    = args.phase in ("all", "pretrain_fp")
    run_ssql  = args.phase in ("all", "pretrain_ssql")
    run_probe = args.phase in ("all", "probe")

    print(f"=== Mini experiment  phase={args.phase}  n_pretrain={args.n_pretrain}  "
          f"epochs={args.epochs}  bs={args.batch_size}  quant=W{w}A{a}  "
          f"norm={args.norm}  lr={lr:.2e} ===")
    print(f"  output → {out}\n")

    # ── datasets (only needed when pretraining or extracting features) ────────
    if run_fp or run_ssql:
        os.environ["CT_CLIP_MAX_SAMPLES"] = str(args.n_pretrain)
        pretrain_ds = CTPretrainDataset(
            data_folder=DATA_FOLDER, reports_file=REPORTS_FILE,
            meta_file=META_FILE, augmentation=CTAugmentation(),
        )
        print(f"Pretrain dataset : {len(pretrain_ds)} scans")

        os.environ.pop("CT_CLIP_MAX_SAMPLES", None)
        eval_ds = AllScansDataset(DATA_FOLDER, REPORTS_FILE, LABELS_FILE, META_FILE)
        eval_loader = DataLoader(
            eval_ds, batch_size=4, shuffle=False,
            num_workers=args.num_workers_eval, pin_memory=True,
        )
        print(f"Eval dataset     : {len(eval_ds)} scans")

        # save shared metadata once (whichever phase runs first writes it)
        feat_dir.mkdir(parents=True, exist_ok=True)
        if not (feat_dir / "accessions.json").exists():
            with open(feat_dir / "accessions.json", "w") as f:
                json.dump(eval_ds.accessions, f)
            with open(feat_dir / "label_names.json", "w") as f:
                json.dump(eval_ds.label_names, f)
            all_labels = torch.tensor(
                np.stack([lbl for _, lbl in eval_ds.samples]), dtype=torch.float32)
            torch.save(all_labels, feat_dir / "labels.pt")

    # ── FP pretraining ───────────────────────────────────────────────────────
    if run_fp:
        print(f"\n{'='*60}\nFP pretraining  ({args.epochs} epochs)\n{'='*60}")
        pretrain_loader = DataLoader(
            pretrain_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        backbone_fp  = _build_backbone()
        projector_fp = _build_projector(args.norm)
        predictor_fp = _build_predictor(args.norm)
        pretrain_train(backbone_fp, projector_fp, predictor_fp,
                       pretrain_loader,
                       _pretrain_cfg("fp", out / "pretrain_fp",
                                     args.epochs, args.num_workers, lr,
                                     args.save_every, args.batch_size,
                                     args.freeze_epochs),
                       device)
        del pretrain_loader

        print(f"\nExtracting FP backbone features...")
        backbone_fp.freeze()
        _cache_features(backbone_fp, eval_loader, device, feat_dir,
                        "fp", f"fp_w{w}a{a}", w, a)

        del projector_fp, predictor_fp
        backbone_fp.cpu()
        torch.cuda.empty_cache()

    # ── SSQL pretraining ─────────────────────────────────────────────────────
    if run_ssql:
        print(f"\n{'='*60}\nSSQL pretraining  ({args.epochs} epochs)\n{'='*60}")
        pretrain_loader = DataLoader(
            pretrain_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        backbone_ssql  = _build_backbone()   # fresh CT-CLIP init
        projector_ssql = _build_projector(args.norm)
        predictor_ssql = _build_predictor(args.norm)
        pretrain_train(backbone_ssql, projector_ssql, predictor_ssql,
                       pretrain_loader,
                       _pretrain_cfg("ssql", out / "pretrain_ssql",
                                     args.epochs, args.num_workers, lr,
                                     args.save_every, args.batch_size,
                                     args.freeze_epochs),
                       device)
        del pretrain_loader

        print(f"\nExtracting SSQL backbone features...")
        backbone_ssql.freeze()
        _cache_features(backbone_ssql, eval_loader, device, feat_dir,
                        "ssql", f"ssql_w{w}a{a}", w, a)

        del projector_ssql, predictor_ssql
        backbone_ssql.cpu()
        torch.cuda.empty_cache()

    # ── linear probe evaluation ───────────────────────────────────────────────
    if run_probe:
        print(f"\n{'='*60}\nLinear probe evaluation\n{'='*60}")
        with open(feat_dir / "accessions.json") as f: accessions  = json.load(f)
        with open(feat_dir / "label_names.json") as f: label_names = json.load(f)
        all_labels = torch.load(feat_dir / "labels.pt", map_location="cpu")

        results, backbones = _run_probes(
            feat_dir, accessions, label_names, all_labels,
            conditions, seeds, device, args.probe_epochs,
        )

        # ── cross-condition summary ───────────────────────────────────────────
        print(f"\n{'='*60}\nCross-condition AUROC (N=all)\n{'='*60}")
        col_w = 18
        header = f"  {'Condition':<32}" + "".join(f"  {b:>{col_w}}" for b in backbones)
        print(header)
        for cond in conditions:
            if cond not in results: continue
            row = f"  {cond:<32}"
            for b in backbones:
                max_n  = max(results[cond][b].keys())
                mean_a = float(np.mean(results[cond][b][max_n]["auroc"]))
                std_a  = float(np.std(results[cond][b][max_n]["auroc"]))
                row += f"  {mean_a:.4f} ±{std_a:.4f}"
            print(row)

        def _serialise(obj):
            if isinstance(obj, dict):
                return {str(k): _serialise(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [int(x) if isinstance(x, (int, np.integer)) else float(x) for x in obj]
            return float(obj)

        out_path = out / "probe_results.json"
        with open(out_path, "w") as f:
            json.dump({"conditions": conditions, "backbones": backbones,
                       "seeds": seeds, "results": _serialise(results)}, f, indent=2)
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
