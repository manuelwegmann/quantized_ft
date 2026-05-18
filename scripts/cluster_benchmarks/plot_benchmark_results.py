"""Bar plots comparing GPU benchmark results across clusters."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

labels = [
    "DIKU\n1x A100",
    "Gefion\n1x H100",
    "Gefion\n2x H100",
    "Gefion\n4x H100",
    "Gefion\n8x H100",
]
elapsed   = [1542.0, 609.0, 308.0, 155.0, 77.0]
throughput = [1.95,   4.93,  9.74, 19.41, 38.86]

colors = ["#4878CF"] + ["#E07B39"] * 4  # blue for DIKU, orange for Gefion

x = np.arange(len(labels))
bar_width = 0.5

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Cluster GPU Benchmark: DIKU (A100) vs Gefion (H100)\n"
    "SimSiam forward+backward, synthetic CT volumes, 3000 samples",
    fontsize=12,
)

# --- Throughput ---
ax = axes[0]
bars = ax.bar(x, throughput, width=bar_width, color=colors, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Throughput (samples / sec)")
ax.set_title("Throughput")
for bar, val in zip(bars, throughput):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)

# --- Total elapsed time ---
ax = axes[1]
bars = ax.bar(x, elapsed, width=bar_width, color=colors, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Total elapsed time (s)")
ax.set_title("Total Elapsed Time")
for bar, val in zip(bars, elapsed):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f"{val:.0f}s", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
out_path = RESULTS_DIR / "benchmark_cluster_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
