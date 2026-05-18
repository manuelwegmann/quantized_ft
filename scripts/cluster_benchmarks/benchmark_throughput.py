"""
GPU throughput benchmark for quantized_ft FP SimSiam pretraining.
Uses synthetic CT volumes (random tensors) — no real data or checkpoint needed.

Single GPU:
    python scripts/benchmark_throughput.py

Multi-GPU via torchrun:
    torchrun --nproc_per_node=4 scripts/benchmark_throughput.py

Results are printed on rank 0 only.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # registers CT-CLIP on sys.path (needs CT_CLIP_ROOT set)

from models.backbone import CTViTBackbone
from models.simsiam import Projector, Predictor
from pretrain.loss import negative_cosine_similarity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=2, help="Samples per GPU per step")
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--steps", type=int, default=20, help="Timed steps")
    p.add_argument("--depth", type=int, default=240, help="CT depth dimension D")
    return p.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_rank0 = local_rank == 0

    if is_rank0:
        print(f"[benchmark] GPUs={world_size}  device={torch.cuda.get_device_name(local_rank)}")
        print(f"[benchmark] batch/GPU={args.batch_size}  depth={args.depth}  "
              f"warmup={args.warmup_steps}  steps={args.steps}")

    # Random init — no checkpoint needed for a speed benchmark
    backbone  = CTViTBackbone(checkpoint_path=None).to(device)
    projector = Projector(in_dim=512, hidden_dim=2048, out_dim=2048).to(device)
    predictor = Predictor(in_dim=2048, hidden_dim=512, out_dim=2048).to(device)

    if world_size > 1:
        backbone  = DDP(backbone,  device_ids=[local_rank])
        projector = DDP(projector, device_ids=[local_rank])
        predictor = DDP(predictor, device_ids=[local_rank])

    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(projector.parameters()) + list(predictor.parameters()),
        lr=0.05, momentum=0.9,
    )

    x_shape = (args.batch_size, 1, args.depth, 480, 480)
    if is_rank0:
        gb = args.batch_size * args.depth * 480 * 480 * 4 / 1e9
        print(f"[benchmark] synthetic input shape: {x_shape}  (~{gb:.1f} GB/batch fp32)")

    def run_step():
        x1 = torch.randn(x_shape, device=device)
        x2 = torch.randn(x_shape, device=device)
        z1 = projector(backbone(x1))
        z2 = projector(backbone(x2))
        p1 = predictor(z1)
        p2 = predictor(z2)
        loss = (
            negative_cosine_similarity(p1, z2.detach())
            + negative_cosine_similarity(p2, z1.detach())
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if is_rank0:
        print(f"[benchmark] warming up ({args.warmup_steps} steps)...")
    for _ in range(args.warmup_steps):
        run_step()
    torch.cuda.synchronize(device)

    if is_rank0:
        print(f"[benchmark] timing {args.steps} steps...")
    t0 = time.perf_counter()
    for _ in range(args.steps):
        run_step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    if is_rank0:
        ms_per_step  = elapsed / args.steps * 1000
        total_samples = args.batch_size * world_size * args.steps
        throughput   = total_samples / elapsed
        print()
        print("=" * 52)
        print(f"  GPU type       : {torch.cuda.get_device_name(local_rank)}")
        print(f"  GPU count      : {world_size}")
        print(f"  Batch / GPU    : {args.batch_size}")
        print(f"  Steps timed    : {args.steps}")
        print(f"  Total elapsed  : {elapsed:.1f} s")
        print(f"  ms / step      : {ms_per_step:.0f} ms")
        print(f"  Throughput     : {throughput:.3f} samples/sec  (all GPUs)")
        print(f"  Per-GPU        : {throughput / world_size:.3f} samples/sec")
        print("=" * 52)
        print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
