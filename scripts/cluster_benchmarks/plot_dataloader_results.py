"""Bar plots comparing dataloader benchmark results across clusters."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

workers = [1, 4, 8, 16]

diku_ms  = [8286, 2019, 1109, 777]
gefion_ms = [11573, 2940, 1536, 820]

diku_sps  = [0.241, 0.990, 1.803, 2.572]
gefion_sps = [0.173, 0.680, 1.302, 2.438]

x = np.arange(len(workers))
bar_width = 0.35

DIKU_COLOR   = "#4878CF"
GEFION_COLOR = "#E07B39"

PIPELINE = (
    "Pipeline: NIfTI load → resize (480×480×240) → CTAugmentation → 2 views/scan\n"
    "500 scans · batch=2 · 5 warmup + 50 timed batches · CPU/storage only (no GPU)"
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Dataloader Benchmark: DIKU (local disk) vs Gefion (network storage)\n{PIPELINE}",
    fontsize=10,
)

# --- Throughput ---
ax = axes[0]
b1 = ax.bar(x - bar_width / 2, diku_sps,   bar_width, label="DIKU (A100 node)",   color=DIKU_COLOR,   edgecolor="white")
b2 = ax.bar(x + bar_width / 2, gefion_sps, bar_width, label="Gefion (H100 node)", color=GEFION_COLOR, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"{w} workers" for w in workers])
ax.set_ylabel("Throughput (samples / sec)")
ax.set_title("Throughput")
ax.legend()
for bar, val in zip(b1, diku_sps):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)
for bar, val in zip(b2, gefion_sps):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

# --- ms/batch ---
ax = axes[1]
b1 = ax.bar(x - bar_width / 2, diku_ms,   bar_width, label="DIKU (A100 node)",   color=DIKU_COLOR,   edgecolor="white")
b2 = ax.bar(x + bar_width / 2, gefion_ms, bar_width, label="Gefion (H100 node)", color=GEFION_COLOR, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"{w} workers" for w in workers])
ax.set_ylabel("ms / batch")
ax.set_title("Batch Latency")
ax.legend()
for bar, val in zip(b1, diku_ms):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{val}", ha="center", va="bottom", fontsize=8)
for bar, val in zip(b2, gefion_ms):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{val}", ha="center", va="bottom", fontsize=8)

fig.tight_layout()
out_path = RESULTS_DIR / "benchmark_dataloader_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
