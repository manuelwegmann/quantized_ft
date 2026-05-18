"""
Data loading throughput benchmark for quantized_ft.

Measures how fast the full preprocessing pipeline (NIfTI load → resize →
augment) runs from disk, sweeping over different num_workers values.
No GPU needed — this is a pure CPU / storage benchmark.

Usage:
    python scripts/benchmark_dataloader.py --data_dir /path/to/merlin_data

Example (Gefion):
    python scripts/benchmark_dataloader.py \\
        --data_dir /dcai/users/wegman/data/merlin_dataset/merlinabdominalctdataset/merlin_data
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # noqa: E402

from pretrain.augmentations import CTAugmentation
from pretrain.dataset import _nii_to_tensor


class _RawCTDataset(Dataset):
    """Minimal dataset: glob all .nii.gz files, no reports/metadata required."""

    def __init__(self, data_dir: str, max_files: int, augmentation):
        root = Path(data_dir)
        files = sorted(root.rglob("*.nii.gz"))[:max_files]
        if not files:
            raise FileNotFoundError(f"No .nii.gz files found under {data_dir}")
        self.files = files
        self.augmentation = augmentation
        print(f"[dataloader_bench] found {len(self.files)} scans under {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tensor = _nii_to_tensor(str(self.files[idx]), {})
        view1 = self.augmentation(tensor.clone())
        view2 = self.augmentation(tensor.clone())
        return view1, view2


def _run_one(data_dir, num_workers, batch_size, warmup_batches, timed_batches, max_files):
    aug = CTAugmentation()
    dataset = _RawCTDataset(data_dir, max_files=max_files, augmentation=aug)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    it = iter(loader)

    # Warmup — let workers spin up and prefetch
    print(f"  [workers={num_workers}] warming up ({warmup_batches} batches)...")
    for _ in range(warmup_batches):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)

    # Timed run
    print(f"  [workers={num_workers}] timing {timed_batches} batches...")
    t0 = time.perf_counter()
    loaded = 0
    for _ in range(timed_batches):
        try:
            x1, x2 = next(it)
        except StopIteration:
            it = iter(loader)
            x1, x2 = next(it)
        loaded += x1.shape[0]
    elapsed = time.perf_counter() - t0

    samples_per_sec = loaded / elapsed
    ms_per_batch = elapsed / timed_batches * 1000
    return elapsed, ms_per_batch, samples_per_sec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Directory containing .nii.gz files")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_files", type=int, default=50,
                   help="Fixed number of scans to use — keep identical across clusters for a fair comparison")
    p.add_argument("--warmup_batches", type=int, default=2, help="Batches to discard before timing")
    p.add_argument("--timed_batches", type=int, default=5, help="Batches to time per num_workers setting")
    p.add_argument(
        "--num_workers", type=int, nargs="+", default=[1, 4, 8, 16],
        help="List of num_workers values to sweep (e.g. --num_workers 1 4 8 16)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[dataloader_bench] batch_size={args.batch_size}  "
          f"warmup={args.warmup_batches}  timed={args.timed_batches}")
    print(f"[dataloader_bench] sweeping num_workers: {args.num_workers}")
    print()

    results = []
    for nw in args.num_workers:
        elapsed, ms_per_batch, sps = _run_one(
            args.data_dir, nw, args.batch_size,
            args.warmup_batches, args.timed_batches, args.max_files,
        )
        results.append((nw, ms_per_batch, sps))
        print(f"  workers={nw:2d}  {ms_per_batch:7.0f} ms/batch  {sps:.3f} samples/sec")
        print()

    print("=" * 52)
    print(f"  {'workers':>8}  {'ms/batch':>10}  {'samples/sec':>13}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*13}")
    for nw, ms, sps in results:
        print(f"  {nw:>8}  {ms:>10.0f}  {sps:>13.3f}")
    best_nw, best_ms, best_sps = max(results, key=lambda r: r[2])
    print(f"\n  Best: num_workers={best_nw}  →  {best_sps:.3f} samples/sec  ({best_ms:.0f} ms/batch)")
    print("=" * 52)


if __name__ == "__main__":
    main()
