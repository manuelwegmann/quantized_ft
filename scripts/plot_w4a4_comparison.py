"""
W4A4 quantized AUROC per condition: CT-CLIP vs FP SimSiam vs SSQL SimSiam.

Data sources
------------
CT-CLIP W4A4  : runs/quant_probe/results.json        (1000 probe epochs, 5 seeds)
FP SimSiam W4A4 : runs/mini_experiment_ln/probe_results.json (500 probe epochs, 3 seeds)
SSQL SimSiam W4A4 : same file as above

NOTE: probe hyperparameters differ between CT-CLIP and the SimSiam models.
Once a unified probe run is available (all backbones probed with the same
settings), update DATA_SOURCES below to point to a single results file.

Usage:
    python scripts/plot_w4a4_comparison.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "plots"
OUT.mkdir(exist_ok=True)

# ── data sources ──────────────────────────────────────────────────────────────

def load(path):
    with open(ROOT / path) as f:
        return json.load(f)

qp    = load("runs/quant_probe/results.json")
probe = load("runs/mini_experiment_ln/probe_results.json")

CONDITIONS = qp["conditions"]   # ground-truth order
COND_LABELS = [
    "Atelectasis", "No Gallbladder", "Renal Cyst",
    "Pleural Effusion", "Cardiomegaly", "Gallstones",
]

# ── per-condition AUROC at max N ───────────────────────────────────────────────

def maxn_stats(results, backbone):
    """Return (means, stds) arrays across conditions, each at max available N."""
    means, stds = [], []
    for cond in CONDITIONS:
        cond_data = results.get(cond, {}).get(backbone, {})
        if not cond_data:
            means.append(np.nan); stds.append(0.0)
            continue
        best_n  = str(max(cond_data.keys(), key=int))
        aurocs  = cond_data[best_n]["auroc"]
        means.append(float(np.mean(aurocs)))
        stds.append(float(np.std(aurocs)))
    return np.array(means), np.array(stds)


ct_m,   ct_s   = maxn_stats(qp["results"],    "pretrained_pre_vq_w4a4")
fp_m,   fp_s   = maxn_stats(probe["results"], "fp_w4a4")
ssql_m, ssql_s = maxn_stats(probe["results"], "ssql_w4a4")

# ── plot ──────────────────────────────────────────────────────────────────────

x      = np.arange(len(CONDITIONS))
width  = 0.24
offsets = np.array([-1, 0, 1]) * width

specs = [
    (ct_m,   ct_s,   "#F46D43", "CT-CLIP W4A4"),
    (fp_m,   fp_s,   "#A6D96A", "FP SimSiam W4A4"),
    (ssql_m, ssql_s, "#C2A5CF", "SSQL SimSiam W4A4"),
]

fig, ax = plt.subplots(figsize=(10, 4.8))

for (means, stds, color, label), offset in zip(specs, offsets):
    ax.bar(x + offset, means, width,
           yerr=stds, capsize=4,
           color=color, alpha=0.88, zorder=3, label=label)

ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(COND_LABELS, rotation=25, ha="right", fontsize=10)
ax.set_ylabel("AUROC (W4A4 quantized)", fontsize=12)
ax.set_title("W4A4 quantized AUROC per condition (N = max per condition)",
             fontsize=12)
ax.set_ylim(0.35, 0.75)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.25, axis="y")

fig.tight_layout()
out_path = OUT / "w4a4_comparison.pdf"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_path}")
