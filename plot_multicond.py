import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_PATH = "runs/multi_condition/results.json"
OUTPUT_PATH  = "runs/multi_condition/auroc_multicond.pdf"

MODEL_LABELS = {
    "pretrained":        "ViT pretrained",
    "pretrained_pre_vq": "ViT pre-VQ",
    "random":            "ViT random",
    "random_cnn":        "CNN random",
}
COLORS = {
    "pretrained":        "#1f77b4",
    "pretrained_pre_vq": "#ff7f0e",
    "random":            "#2ca02c",
    "random_cnn":        "#9467bd",
}
MARKERS = {
    "pretrained":        "o",
    "pretrained_pre_vq": "s",
    "random":            "^",
    "random_cnn":        "D",
}

COND_LABELS = {
    "atelectasis":                 "Atelectasis",
    "surgically_absent_gallbladder":"Absent Gallbladder",
    "renal_cyst":                  "Renal Cyst",
    "pleural_effusion":            "Pleural Effusion",
    "cardiomegaly":                "Cardiomegaly",
    "gallstones":                  "Gallstones",
}

with open(RESULTS_PATH) as f:
    data = json.load(f)

conditions = data["conditions"]
backbones  = data["backbones"]
results    = data["results"]

fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
axes = axes.flatten()

for ax, cond in zip(axes, conditions):
    cond_data = results[cond]
    for bb in backbones:
        bb_data = cond_data[bb]
        ns, means, stds = [], [], []
        for n_str, entry in sorted(bb_data.items(), key=lambda x: int(x[0])):
            aurocs = entry["auroc"]
            ns.append(int(n_str))
            means.append(np.mean(aurocs))
            stds.append(np.std(aurocs, ddof=1))
        ns = np.array(ns)
        means = np.array(means)
        stds  = np.array(stds)
        ax.plot(ns, means, marker=MARKERS[bb], color=COLORS[bb],
                label=MODEL_LABELS[bb], linewidth=1.6, markersize=5)
        ax.fill_between(ns, means - stds, means + stds,
                        color=COLORS[bb], alpha=0.15)

    # collect all N values for this condition (from the first backbone)
    first_bb = list(cond_data.values())[0]
    all_ns = sorted(int(k) for k in first_bb.keys())

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_title(COND_LABELS[cond], fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(all_ns)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis="x", which="both", labelsize=8, rotation=45)
    ax.set_xlim(all_ns[0] * 0.6, all_ns[-1] * 1.6)
    ax.set_xlabel("Training samples (N)", fontsize=9)
    ax.set_ylabel("AUROC", fontsize=9)
    ax.set_ylim(0.42, 0.82)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.7)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))

fig.suptitle("Linear probe AUROC — 6 conditions × 4 backbones\n(shaded band = ±1 SD over 5 seeds)",
             fontsize=12)

plt.savefig(OUTPUT_PATH, bbox_inches="tight")
print(f"Saved → {OUTPUT_PATH}")
plt.close()
