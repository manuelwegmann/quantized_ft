#!/usr/bin/env python3
"""Generate presentation figures for CT-CLIP quantized pretraining experiments."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "runs"
OUT  = ROOT / "plots"
OUT.mkdir(exist_ok=True)

# Colorblind-friendly palette
C = {
    "ct_clip":    "#2166AC",   # blue
    "random":     "#D73027",   # red
    "cnn":        "#F4A582",   # peach
    "fp":         "#1A9641",   # green
    "fp_q":       "#A6D96A",   # light green
    "ssql":       "#762A83",   # purple
    "ssql_q":     "#C2A5CF",   # light purple
    "snap":       "#888888",
}

LABEL = {
    "pretrained_pre_vq": "CT-CLIP (pretrained)",
    "random":            "Random ViT",
    "random_cnn":        "Random CNN",
    "fp":                "FP SimSiam",
    "fp_w4a4":           "FP SimSiam W4A4",
    "ssql":              "SSQL SimSiam",
    "ssql_w4a4":         "SSQL SimSiam W4A4",
    "pretrained_pre_vq_w4a4": "CT-CLIP W4A4",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def macro_auroc(results, backbone, n):
    """Mean and std AUROC across all conditions and seeds for a given (backbone, N)."""
    n = str(n)
    vals = []
    for cond_data in results.values():
        if backbone in cond_data and n in cond_data[backbone]:
            vals.extend(cond_data[backbone][n]["auroc"])
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


def per_condition_auroc_at_maxn(results, conditions, backbone):
    """For each condition, use the largest available N (prevalence varies per condition)."""
    means, stds, ns_used = [], [], []
    for cond in conditions:
        cond_data = results.get(cond, {}).get(backbone, {})
        if not cond_data:
            means.append(np.nan); stds.append(0); ns_used.append(0)
            continue
        best_n = str(max(cond_data.keys(), key=int))
        aurocs = cond_data[best_n]["auroc"]
        means.append(float(np.mean(aurocs)))
        stds.append(float(np.std(aurocs)))
        ns_used.append(int(best_n))
    return np.array(means), np.array(stds), ns_used


# ---------------------------------------------------------------------------
# Figure 1 – Zero-shot linear probing
# ---------------------------------------------------------------------------
def make_fig1():
    mc = load(RUNS / "multi_condition/results.json")
    results    = mc["results"]
    conditions = mc["conditions"]

    cond_labels = [
        "Atelectasis", "No Gallbladder", "Renal Cyst",
        "Pleural Effusion", "Cardiomegaly", "Gallstones",
    ]

    # Reported per-condition AUROC from Blankemeier et al. (Merlin), same condition order.
    MERLIN_AUROC = np.array([0.72, 0.93, 0.61, 0.80, 0.81, 0.75])

    # (backbone_key, label, color, hardcoded_means_or_None)
    bar_specs = [
        (None,                "Merlin VLM",        "#FF6B35",   MERLIN_AUROC),
        ("pretrained_pre_vq", "CT-CLIP (pre-VQ)",  C["ct_clip"], None),
        ("pretrained",        "CT-CLIP (post-VQ)", "#5BA3D9",    None),
        ("random",            "Random ViT",         C["random"],  None),
        ("random_cnn",        "Random CNN",         C["cnn"],     None),
    ]
    n_bb    = len(bar_specs)
    x       = np.arange(len(conditions))
    width   = 0.16
    offsets = np.linspace(-(n_bb - 1) / 2, (n_bb - 1) / 2, n_bb) * width

    fig, ax = plt.subplots(figsize=(12, 5))

    for (bb, label, color, hardcoded), offset in zip(bar_specs, offsets):
        if hardcoded is not None:
            means = hardcoded
        else:
            means, _, _ = per_condition_auroc_at_maxn(results, conditions, bb)
        ax.bar(x + offset, means, width, label=label, color=color, alpha=0.88, zorder=3)

    # annotate each condition with its max N
    _, _, ns_used = per_condition_auroc_at_maxn(results, conditions, "pretrained_pre_vq")
    for xi, n in zip(x, ns_used):
        ax.text(xi, 0.365, f"N={n}", ha="center", va="bottom", fontsize=7.5, color="#555555")

    ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Zero-shot linear probing on Merlin abdominal CT (N = max per condition)",
                 fontsize=13)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(0.35, 0.98)
    ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_zeroshot.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_zeroshot")


# ---------------------------------------------------------------------------
# Figure 2 – Pretraining collapse
# ---------------------------------------------------------------------------
def make_fig2():
    diag = load(RUNS / "exp_ln_1000/diagnostics_fp.json")
    run  = diag["results"]["pretrain_fp"]
    epochs = sorted(run.keys(), key=int)
    ep     = [int(e) for e in epochs]

    uni  = [run[e]["uniformity"]     for e in epochs]
    rank = [run[e]["effective_rank"] for e in epochs]
    knn  = [run[e]["knn_auroc"]      for e in epochs]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    color = C["fp"]

    panel_specs = [
        (axes[0], uni,  "A  Uniformity",    "Uniformity",    None),
        (axes[1], rank, "B  Effective rank", "Effective rank", None),
        (axes[2], knn,  "C  kNN AUROC",      "kNN macro-AUROC", knn[0]),
    ]
    for ax, vals, title, ylabel, baseline in panel_specs:
        ax.plot(ep, vals, "o-", color=color, lw=2.2, markersize=6, zorder=3)
        if baseline is not None:
            ax.axhline(baseline, color=C["snap"], lw=0.9, ls=":",
                       label="CT-CLIP baseline", zorder=1)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        if baseline is not None:
            ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.25)

    fig.suptitle("FP SimSiam pretraining dynamics (N=1000, bs=2, LN, cosine LR)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_collapse.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_collapse")


# ---------------------------------------------------------------------------
# Figures 5a/b/c – Per-condition comparison: CT-CLIP, FP SimSiam, SSQL SimSiam
# ---------------------------------------------------------------------------
def make_fig5():
    probe = load(RUNS / "mini_experiment_ln/probe_results.json")
    quant = load(RUNS / "quant_probe/results.json")
    conditions = probe["conditions"]

    cond_labels = [
        "Atelectasis", "No Gallbladder", "Renal Cyst",
        "Pleural Effusion", "Cardiomegaly", "Gallstones",
    ]

    def maxn_mean(results, backbone):
        means, _, _ = per_condition_auroc_at_maxn(results, conditions, backbone)
        return means

    # Full-precision features
    ct_fp    = maxn_mean(quant["results"],  "pretrained_pre_vq")
    fp       = maxn_mean(probe["results"],  "fp")
    ssql     = maxn_mean(probe["results"],  "ssql")

    # W8A8 quantized (only available for CT-CLIP in current data)
    ct_w8a8  = maxn_mean(quant["results"],  "pretrained_pre_vq_w8a8")

    # W4A4 quantized
    ct_w4a4  = maxn_mean(quant["results"],  "pretrained_pre_vq_w4a4")
    fp_w4a4  = maxn_mean(probe["results"],  "fp_w4a4")
    ssql_q   = maxn_mean(probe["results"],  "ssql_w4a4")

    x = np.arange(len(conditions))

    # --- Figure 5a: Full-precision AUROC per condition ---
    width  = 0.25
    offsets_3 = np.array([-1, 0, 1]) * width
    kwargs = dict(width=width, alpha=0.88, zorder=3)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x + offsets_3[0], ct_fp, color=C["ct_clip"], label="CT-CLIP (original)", **kwargs)
    ax.bar(x + offsets_3[1], fp,    color=C["fp"],      label="FP SimSiam",         **kwargs)
    ax.bar(x + offsets_3[2], ssql,  color=C["ssql"],    label="SSQL SimSiam",       **kwargs)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_ylim(0.4, 0.85)
    ax.set_title("Full-precision AUROC per condition (N = max per condition)", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig5a_fp_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5a_fp_comparison")

    # --- Figure 5b: W8A8 and W4A4 AUROC per condition ---
    # CT-CLIP has both W8A8 and W4A4; SimSiam models have W4A4 only (W8A8 pending).
    width  = 0.18
    offsets_4 = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    kwargs4 = dict(width=width, alpha=0.88, zorder=3)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x + offsets_4[0], ct_w8a8, color="#4393C3",    label="CT-CLIP W8A8",       **kwargs4)
    ax.bar(x + offsets_4[1], ct_w4a4, color="#F46D43",    label="CT-CLIP W4A4",       **kwargs4)
    ax.bar(x + offsets_4[2], fp_w4a4, color=C["fp_q"],    label="FP SimSiam W4A4",    **kwargs4)
    ax.bar(x + offsets_4[3], ssql_q,  color=C["ssql_q"],  label="SSQL SimSiam W4A4",  **kwargs4)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_ylim(0.4, 0.82)
    ax.set_title("Quantized AUROC per condition — W8A8 and W4A4 (N = max per condition)",
                 fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig5b_quantized_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5b_quantized_comparison")

    # --- Figure 5c: % AUROC change due to W4A4 quantization ---
    pct_ct   = (ct_w4a4 - ct_fp) / ct_fp * 100
    pct_fp   = (fp_w4a4 - fp)    / fp    * 100
    pct_ssql = (ssql_q  - ssql)  / ssql  * 100

    width  = 0.25
    kwargs = dict(width=width, alpha=0.88, zorder=3)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x + offsets_3[0], pct_ct,   color=C["ct_clip"], label="CT-CLIP (original)", **kwargs)
    ax.bar(x + offsets_3[1], pct_fp,   color=C["fp"],      label="FP SimSiam",         **kwargs)
    ax.bar(x + offsets_3[2], pct_ssql, color=C["ssql"],    label="SSQL SimSiam",       **kwargs)
    ax.axhline(0, color="gray", lw=1.0, ls="-", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("AUROC change due to W4A4 (%)", fontsize=12)
    ax.set_title("W4A4 quantization degradation per condition", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig5c_quant_change.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5c_quant_change")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig5()
    print(f"\nAll figures saved to {OUT}/")
