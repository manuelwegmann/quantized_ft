"""
Visualize augmentation strength — one PDF page per scan.

Each page shows 4 rows (Original / View 1 / View 2 / |V1−V2|) × 3 columns
(Axial / Coronal / Sagittal mid-slices).  Rows are colour-coded and labelled
so the role of each image is unambiguous at a glance.

Run from the project root (CPU-only):
    python scripts/visualize_augmentations.py [--n_scans 3] [--out plots/aug_vis.pdf]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import pandas as pd

from pretrain.dataset import _nii_to_tensor
from pretrain.augmentations import CTAugmentation


# ── colour scheme for the four row types ────────────────────────────────────
ROW_META = [
    # (label text,            bg colour,  cmap)
    ("ORIGINAL",              "#e8e8e8",  "gray"),
    ("VIEW 1\n(augmented)",   "#c6dbef",  "gray"),
    ("VIEW 2\n(augmented)",   "#c7e9c0",  "gray"),
]
ORIENTATIONS = ["Axial\n(mid-depth)", "Coronal\n(mid-height)", "Sagittal\n(mid-width)"]
AUG_PARAMS = (
    "Augmentation pipeline: "
    "RandomCrop3D(ratio=0.85)  →  RandomFlip3D(p=0.5 per axis)  →  "
    "IntensityJitter(scale ∈ [0.9, 1.1],  shift ∈ [−0.05, 0.05])  →  "
    "GaussianNoise(std=0.01)"
)


def mid_slices(vol):
    """(1, D, H, W) → list of three 2-D numpy arrays: axial, coronal, sagittal."""
    v = vol[0].numpy()
    D, H, W = v.shape
    return [v[D // 2], v[:, H // 2, :], v[:, :, W // 2]]


def subject_id(path: Path) -> str:
    """Strip .nii.gz (or .nii) to get a clean ID string."""
    name = path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def render_page(pdf, scan_idx, path, original, view1, view2):
    vols = [original, view1, view2]

    vmin = float(original.min())
    vmax = float(original.max())

    sid = subject_id(path)
    slices_per_vol = [mid_slices(v) for v in vols]

    fig = plt.figure(figsize=(13, 11))
    fig.patch.set_facecolor("white")

    # Title (top) and augmentation params (bottom footer)
    fig.text(0.5, 0.980, f"Subject: {sid}",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.018, AUG_PARAMS,
             ha="center", va="bottom", fontsize=7.5, color="#444444",
             style="italic")

    # GridSpec: narrow label col + 3 image cols
    gs = GridSpec(
        3, 4,
        figure=fig,
        left=0.14, right=0.97,
        top=0.93, bottom=0.06,
        wspace=0.06, hspace=0.32,
        width_ratios=[0.22, 1, 1, 1],
    )

    for row, (label, bg, cmap) in enumerate(ROW_META):
        # ── row label panel ──────────────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.set_facecolor(bg)
        ax_lbl.text(0.5, 0.5, label,
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    transform=ax_lbl.transAxes, linespacing=1.4)
        for spine in ax_lbl.spines.values():
            spine.set_edgecolor("#aaaaaa")
        ax_lbl.set_xticks([])
        ax_lbl.set_yticks([])

        # ── three slice panels ───────────────────────────────────────────────
        clim = (vmin, vmax)
        for col, (sl, orient) in enumerate(zip(slices_per_vol[row], ORIENTATIONS)):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(sl, cmap=cmap, vmin=clim[0], vmax=clim[1],
                      aspect="auto", interpolation="nearest")
            if row == 0:
                ax.set_title(orient, fontsize=9, pad=4)
            ax.axis("off")

            # thin coloured border matching the row colour
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(bg)
                spine.set_linewidth(2)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  → page {scan_idx + 1}: {sid}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_scans", type=int, default=3)
    parser.add_argument("--out", default="plots/aug_vis.pdf")
    parser.add_argument("--data_folder",
                        default="/home/nlr950/Dir/CT-CLIP/data/merlin_data")
    parser.add_argument("--meta_file",
                        default="/home/nlr950/Dir/CT-CLIP/data/metadata.csv")
    args = parser.parse_args()

    aug = CTAugmentation()

    meta = pd.read_csv(args.meta_file)
    if "VolumeName" in meta.columns:
        meta = meta.set_index("VolumeName")

    paths = sorted(Path(args.data_folder).glob("*.nii.gz"))[: args.n_scans]
    if not paths:
        raise FileNotFoundError(f"No .nii.gz files found in {args.data_folder}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(paths)} scan(s) → {out}")
    with PdfPages(out) as pdf:
        for scan_idx, path in enumerate(paths):
            name = path.name
            meta_row = meta.loc[name].to_dict() if name in meta.index else {}
            print(f"[{scan_idx + 1}/{len(paths)}] loading {name} ...", flush=True)

            original = _nii_to_tensor(str(path), meta_row)
            view1    = aug(original.clone())
            view2    = aug(original.clone())
            render_page(pdf, scan_idx, path, original, view1, view2)

    print(f"\nDone — {out}")


if __name__ == "__main__":
    main()
