"""
Three-panel comparison: full-precision AUROC, W4A4 AUROC, and % change.
Models: CT-CLIP (original), FP SimSiam, SSQL SimSiam.
Gallstones excluded (anomalous W4A4 result due to ~39 test samples).

Data sources (probe configs differ — unified re-run pending):
  CT-CLIP FP / W4A4 : runs/quant_probe/results.json        (1000 ep, 5 seeds)
  FP / SSQL SimSiam  : runs/mini_experiment_ln/probe_results.json (500 ep, 3 seeds)

Usage:
    python scripts/plot_quantization_comparison.py
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

# ── config ────────────────────────────────────────────────────────────────────

EXCLUDE   = {"gallstones"}
CONDITIONS = [c for c in [
    "atelectasis", "surgically_absent_gallbladder", "renal_cyst",
    "pleural_effusion", "cardiomegaly",
] if c not in EXCLUDE]

COND_LABELS = [
    "Atelectasis", "No Gallbladder", "Renal Cyst",
    "Pleural Effusion", "Cardiomegaly",
]

C = {
    "ct_clip":  "#2166AC",
    "fp":       "#1A9641",
    "ssql":     "#762A83",
}

# ── load data ─────────────────────────────────────────────────────────────────

def load(path):
    with open(ROOT / path) as f:
        return json.load(f)

qp    = load("runs/quant_probe/results.json")
probe = load("runs/mini_experiment_ln/probe_results.json")


def maxn_stats(results, backbone, conditions):
    means, stds = [], []
    for cond in conditions:
        cond_data = results.get(cond, {}).get(backbone, {})
        if not cond_data:
            means.append(np.nan); stds.append(0.0)
            continue
        best_n = str(max(cond_data.keys(), key=int))
        aurocs = cond_data[best_n]["auroc"]
        means.append(float(np.mean(aurocs)))
        stds.append(float(np.std(aurocs)))
    return np.array(means), np.array(stds)


ct_fp_m,   ct_fp_s   = maxn_stats(qp["results"],    "pretrained_pre_vq",      CONDITIONS)
ct_q4_m,   ct_q4_s   = maxn_stats(qp["results"],    "pretrained_pre_vq_w4a4", CONDITIONS)
fp_fp_m,   fp_fp_s   = maxn_stats(probe["results"], "fp",                     CONDITIONS)
fp_q4_m,   fp_q4_s   = maxn_stats(probe["results"], "fp_w4a4",                CONDITIONS)
ssql_fp_m, ssql_fp_s = maxn_stats(probe["results"], "ssql",                   CONDITIONS)
ssql_q4_m, ssql_q4_s = maxn_stats(probe["results"], "ssql_w4a4",              CONDITIONS)

ct_pct   = (ct_q4_m   - ct_fp_m)   / ct_fp_m   * 100
fp_pct   = (fp_q4_m   - fp_fp_m)   / fp_fp_m   * 100
ssql_pct = (ssql_q4_m - ssql_fp_m) / ssql_fp_m * 100

# ── plot ──────────────────────────────────────────────────────────────────────

x       = np.arange(len(CONDITIONS))
width   = 0.22
offsets = np.array([-1, 0, 1]) * width

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
fig.suptitle(
    "CT-CLIP vs FP SimSiam vs SSQL SimSiam — full precision and W4A4 quantized\n"
    "(N = max per condition; gallstones excluded)",
    fontsize=12, y=1.02,
)

bar_kw = dict(width=width, alpha=0.88, zorder=3)

# ── Panel A: full-precision AUROC ─────────────────────────────────────────────
ax = axes[0]
ax.bar(x + offsets[0], ct_fp_m,   color=C["ct_clip"], yerr=ct_fp_s,   capsize=3,
       label="CT-CLIP",      **bar_kw)
ax.bar(x + offsets[1], fp_fp_m,   color=C["fp"],      yerr=fp_fp_s,   capsize=3,
       label="FP SimSiam",   **bar_kw)
ax.bar(x + offsets[2], ssql_fp_m, color=C["ssql"],    yerr=ssql_fp_s, capsize=3,
       label="SSQL SimSiam", **bar_kw)
ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(COND_LABELS, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("AUROC", fontsize=11)
ax.set_ylim(0.40, 0.85)
ax.set_title("A   Full precision", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.25, axis="y")

# ── Panel B: W4A4 AUROC ───────────────────────────────────────────────────────
ax = axes[1]
ax.bar(x + offsets[0], ct_q4_m,   color=C["ct_clip"], yerr=ct_q4_s,   capsize=3,
       label="CT-CLIP W4A4",      **bar_kw)
ax.bar(x + offsets[1], fp_q4_m,   color=C["fp"],      yerr=fp_q4_s,   capsize=3,
       label="FP SimSiam W4A4",   **bar_kw)
ax.bar(x + offsets[2], ssql_q4_m, color=C["ssql"],    yerr=ssql_q4_s, capsize=3,
       label="SSQL SimSiam W4A4", **bar_kw)
ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(COND_LABELS, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("AUROC", fontsize=11)
ax.set_ylim(0.35, 0.85)
ax.set_title("B   W4A4 quantized", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.25, axis="y")

# ── Panel C: % change FP → W4A4 ───────────────────────────────────────────────
ax = axes[2]
ax.bar(x + offsets[0], ct_pct,   color=C["ct_clip"], label="CT-CLIP",      **bar_kw)
ax.bar(x + offsets[1], fp_pct,   color=C["fp"],      label="FP SimSiam",   **bar_kw)
ax.bar(x + offsets[2], ssql_pct, color=C["ssql"],    label="SSQL SimSiam", **bar_kw)
ax.axhline(0, color="gray", lw=1.0, ls="-", zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(COND_LABELS, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("AUROC change FP → W4A4 (%)", fontsize=11)
ax.set_title("C   Quantization degradation", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.25, axis="y")

fig.tight_layout()
out_path = OUT / "quantization_comparison.pdf"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_path}")
