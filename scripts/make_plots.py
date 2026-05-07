#!/usr/bin/env python3
"""Generate presentation figures for CT-CLIP quantized pretraining experiments."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    bar_specs = [
        ("pretrained_pre_vq", "CT-CLIP (pre-VQ)",  C["ct_clip"]),
        ("pretrained",        "CT-CLIP (post-VQ)", "#5BA3D9"),
        ("random",            "Random ViT",         C["random"]),
        ("random_cnn",        "Random CNN",         C["cnn"]),
    ]
    n_bb    = len(bar_specs)
    x       = np.arange(len(conditions))
    width   = 0.20
    offsets = np.linspace(-(n_bb - 1) / 2, (n_bb - 1) / 2, n_bb) * width

    fig, ax = plt.subplots(figsize=(11, 5))

    for (bb, label, color), offset in zip(bar_specs, offsets):
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
    ax.set_ylim(0.35, 0.95)
    ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_zeroshot.{ext}", dpi=150, bbox_inches="tight")
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
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_collapse.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_collapse")


# ---------------------------------------------------------------------------
# Figure 3 – SSQL vs FP quantization retention
# ---------------------------------------------------------------------------
def make_fig3():
    probe = load(RUNS / "mini_experiment_ln/probe_results.json")
    quant = load(RUNS / "quant_probe/results.json")
    conditions = probe["conditions"]  # same 6 conditions in both files

    def maxn_auroc(results, backbone):
        """Macro-mean across all conditions, each at its own max N."""
        m, s, _ = per_condition_auroc_at_maxn(results, conditions, backbone)
        valid = m[~np.isnan(m)]
        return float(np.mean(valid)), float(np.std(valid))

    # Each entry: (label, FP mean, FP std, W4A4 mean, W4A4 std, color_fp, color_q)
    rows = []

    ct_fp_m,   ct_fp_s   = maxn_auroc(quant["results"],  "pretrained_pre_vq")
    ct_q_m,    ct_q_s    = maxn_auroc(quant["results"],  "pretrained_pre_vq_w4a4")
    rows.append(("CT-CLIP\n(original)",      ct_fp_m,  ct_fp_s,  ct_q_m,  ct_q_s,
                 C["ct_clip"], "#92C5DE"))

    fp_fp_m,   fp_fp_s   = maxn_auroc(probe["results"], "fp")
    fp_q_m,    fp_q_s    = maxn_auroc(probe["results"], "fp_w4a4")
    rows.append(("FP SimSiam\n(N=300, 40ep)", fp_fp_m, fp_fp_s, fp_q_m, fp_q_s,
                 C["fp"], C["fp_q"]))

    ssql_fp_m, ssql_fp_s = maxn_auroc(probe["results"], "ssql")
    ssql_q_m,  ssql_q_s  = maxn_auroc(probe["results"], "ssql_w4a4")
    rows.append(("SSQL SimSiam\n(N=300, 40ep)", ssql_fp_m, ssql_fp_s, ssql_q_m, ssql_q_s,
                 C["ssql"], C["ssql_q"]))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x      = np.arange(len(rows))
    width  = 0.32

    for i, (label, fp_m, fp_s, q_m, q_s, col_fp, col_q) in enumerate(rows):
        ax.bar(x[i] - width / 2, fp_m, width, color=col_fp, yerr=fp_s,
               capsize=4, alpha=0.9, zorder=3, label="Full precision" if i == 0 else "")
        ax.bar(x[i] + width / 2, q_m,  width, color=col_q,  yerr=q_s,
               capsize=4, alpha=0.9, zorder=3, hatch="///",
               label="W4A4 quantized" if i == 0 else "")

        # annotate drop
        if not (np.isnan(fp_m) or np.isnan(q_m)):
            drop      = (fp_m - q_m) * 100
            retention = q_m / fp_m * 100
            ax.annotate(
                f"−{drop:.1f} pp\n({retention:.0f}% ret.)",
                xy=(x[i] + width / 2, q_m),
                xytext=(x[i] + width / 2, q_m - 0.055),
                ha="center", fontsize=9, color="#333333",
            )

    ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=11)
    ax.set_ylabel("Macro-mean AUROC (6 conditions, each at max N)", fontsize=11)
    ax.set_title("Quantization robustness: SSQL vs FP SimSiam (W4A4)", fontsize=13)
    ax.set_ylim(0.38, 0.75)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3_ssql_vs_fp.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_ssql_vs_fp")


# ---------------------------------------------------------------------------
# Figure 4 – Quantization degradation curve (original CT-CLIP)
# ---------------------------------------------------------------------------
def make_fig4():
    quant      = load(RUNS / "quant_probe/results.json")
    conditions = quant["conditions"]

    bb_specs = [
        ("pretrained_pre_vq",      "FP",   C["ct_clip"], "o-"),
        ("pretrained_pre_vq_w8a8", "W8A8", "#4393C3",    "s-"),
        ("pretrained_pre_vq_w4a8", "W4A8", "#74ADD1",    "^-"),
        ("pretrained_pre_vq_w4a4", "W4A4", "#F46D43",    "D-"),
        ("pretrained_pre_vq_w2a4", "W2A4", "#D73027",    "v-"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- Panel A: macro-mean AUROC at N=100 and N=300 (consistent across conditions) ---
    ax = axes[0]
    ns_consistent = [100, 300]
    for bb, label, color, style in bb_specs:
        means = [macro_auroc(quant["results"], bb, n)[0] for n in ns_consistent]
        stds  = [macro_auroc(quant["results"], bb, n)[1] for n in ns_consistent]
        means, stds = np.array(means), np.array(stds)
        ax.plot(ns_consistent, means, style, color=color, label=label,
                lw=2.2, markersize=7, zorder=3)
        ax.fill_between(ns_consistent, means - stds, means + stds,
                        color=color, alpha=0.10)

    ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(ns_consistent)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Training samples N", fontsize=12)
    ax.set_ylabel("Macro-mean AUROC (6 conditions)", fontsize=12)
    ax.set_title("A  Learning curves by quantization level", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(0.45, 0.75)
    ax.grid(True, alpha=0.25)

    # --- Panel B: degradation at max N per condition ---
    ax = axes[1]
    means_by_bb = []
    stds_by_bb  = []
    labels_bb   = []
    for bb, label, color, _ in bb_specs:
        m, s, _ = per_condition_auroc_at_maxn(quant["results"], conditions, bb)
        means_by_bb.append(float(np.mean(m[~np.isnan(m)])))
        stds_by_bb.append(float(np.std(m[~np.isnan(m)])))
        labels_bb.append(label)

    colors_bb = [s[2] for s in bb_specs]
    x = np.arange(len(bb_specs))
    ax.bar(x, means_by_bb, color=colors_bb, yerr=stds_by_bb, capsize=4,
           alpha=0.85, zorder=3, width=0.55)
    # annotate drop from FP
    fp_val = means_by_bb[0]
    for i in range(1, len(means_by_bb)):
        drop = (fp_val - means_by_bb[i]) * 100
        ax.annotate(f"−{drop:.1f}pp",
                    xy=(x[i], means_by_bb[i]),
                    xytext=(x[i], means_by_bb[i] - 0.025),
                    ha="center", fontsize=9, color="#333333")

    ax.axhline(fp_val, color=C["ct_clip"], lw=1, ls="--", alpha=0.5, label="FP baseline")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bb, fontsize=11)
    ax.set_ylabel("Macro-mean AUROC (6 conditions, max N)", fontsize=11)
    ax.set_title("B  Degradation vs FP (max N per condition)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim(0.38, 0.70)
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle("CT-CLIP post-training quantization (PTQ) sensitivity", fontsize=13, y=1.01)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4_quant_degradation.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_quant_degradation")


# ---------------------------------------------------------------------------
# Figures 5a/b/c – Per-condition comparison: FP vs SSQL (from job 3213)
# ---------------------------------------------------------------------------
def make_fig5():
    probe = load(RUNS / "mini_experiment_ln/probe_results.json")
    conditions = probe["conditions"]

    cond_labels = [
        "Atelectasis", "No Gallbladder", "Renal Cyst",
        "Pleural Effusion", "Cardiomegaly", "Gallstones",
    ]

    # Read per-condition AUROC at max N (no error bars as requested)
    def maxn_mean(results, backbone):
        means, _, _ = per_condition_auroc_at_maxn(results, conditions, backbone)
        return means

    fp       = maxn_mean(probe["results"], "fp")
    fp_w4a4  = maxn_mean(probe["results"], "fp_w4a4")
    ssql     = maxn_mean(probe["results"], "ssql")
    ssql_q   = maxn_mean(probe["results"], "ssql_w4a4")

    x      = np.arange(len(conditions))
    width  = 0.35
    kwargs = dict(width=width, alpha=0.88, zorder=3)

    # --- Figure 5a: Full-precision AUROC per condition ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, fp,   color=C["fp"],   label="FP SimSiam",   **kwargs)
    ax.bar(x + width / 2, ssql, color=C["ssql"], label="SSQL SimSiam", **kwargs)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_ylim(0.4, 0.82)
    ax.set_title("FP SimSiam vs SSQL SimSiam — full precision (N = max per condition)",
                 fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5a_fp_comparison.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5a_fp_comparison")

    # --- Figure 5b: W4A4 AUROC per condition ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, fp_w4a4, color=C["fp_q"],   label="FP SimSiam W4A4",   **kwargs)
    ax.bar(x + width / 2, ssql_q,  color=C["ssql_q"], label="SSQL SimSiam W4A4", **kwargs)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_ylim(0.4, 0.82)
    ax.set_title("FP SimSiam vs SSQL SimSiam — W4A4 quantized (N = max per condition)",
                 fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5b_w4a4_comparison.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5b_w4a4_comparison")

    # --- Figure 5c: % AUROC change due to W4A4 quantization ---
    pct_fp   = (fp_w4a4 - fp)   / fp   * 100
    pct_ssql = (ssql_q  - ssql) / ssql * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, pct_fp,   color=C["fp"],   label="FP SimSiam",   **kwargs)
    ax.bar(x + width / 2, pct_ssql, color=C["ssql"], label="SSQL SimSiam", **kwargs)
    ax.axhline(0, color="gray", lw=1.0, ls="-", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("AUROC change due to W4A4 (%)", fontsize=12)
    ax.set_title("Quantization degradation per condition: FP SimSiam vs SSQL SimSiam",
                 fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5c_quant_change.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5c_quant_change")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    print(f"\nAll figures saved to {OUT}/")
