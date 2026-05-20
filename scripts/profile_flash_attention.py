"""
Memory profiling: standard attention vs F.scaled_dot_product_attention (Flash).

Runs the CTViT backbone forward pass at increasing batch sizes under two
attention implementations and reports peak GPU memory. The goal is to find
the largest batch size that fits within GPU memory with the Flash variant,
which determines whether BatchNorm becomes viable for SimSiam pretraining.

Usage:
    python scripts/profile_flash_attention.py
    python scripts/profile_flash_attention.py --batch_sizes 1,2,4,8,16,32
"""

import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import paths  # noqa: E402
from models.backbone import CTViTBackbone
from paths import CHECKPOINT


# ── FlashAttention patch ──────────────────────────────────────────────────────

def _flash_forward(self, x, mask=None, context=None, attn_bias=None):
    """
    Drop-in replacement for Attention.forward using F.scaled_dot_product_attention.
    Preserves the QK-norm + null_kv + attn_bias logic of the original.
    """
    from transformer_maskgit.attention import exists, default, l2norm

    batch = x.shape[0]
    if exists(context):
        context = self.context_norm(context)
    kv_input = default(context, x)
    x = self.norm(x)

    q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

    nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b=batch, r=2).unbind(dim=-2)
    k = torch.cat((nk, k), dim=-2)
    v = torch.cat((nv, v), dim=-2)

    q, k = map(l2norm, (q, k))
    q = q * self.q_scale
    k = k * self.k_scale

    # Convert additive attn_bias to attn_mask (pad for null_kv tokens)
    attn_mask = None
    if exists(attn_bias):
        attn_mask = F.pad(attn_bias, (self.num_null_kv, 0), value=0.0)

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=float(self.scale),
    )

    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)


def patch_flash(backbone: CTViTBackbone):
    """Replace all Attention.forward methods with the Flash variant."""
    from transformer_maskgit.attention import Attention
    for module in backbone.modules():
        if isinstance(module, Attention):
            module.forward = types.MethodType(_flash_forward, module)


# ── profiling ─────────────────────────────────────────────────────────────────

def _measure(backbone, projector, predictor, batch_size, device, dtype):
    """
    Peak memory in MB for one SimSiam FP training step (forward + backward).
    Two views are passed through the backbone simultaneously, matching the
    actual memory profile of pretraining.
    """
    backbone.train()
    projector.train()
    predictor.train()

    shape = (batch_size, 1, 240, 480, 480)
    x1 = torch.randn(shape, dtype=dtype, device=device)
    x2 = torch.randn(shape, dtype=dtype, device=device)

    # zero any leftover gradients
    for m in (backbone, projector, predictor):
        for p in m.parameters():
            p.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        z1 = projector(backbone(x1))
        z2 = projector(backbone(x2))
        p1 = predictor(z1)
        p2 = predictor(z2)
        loss = (
            -F.cosine_similarity(p1, z2.detach()).mean()
            - F.cosine_similarity(p2, z1.detach()).mean()
        )
        loss.backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / 1024**2
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def profile(backbone, projector, predictor, batch_sizes, device, dtype, label):
    print(f"\n{'─'*60}")
    print(f"  {label}  (dtype={dtype})")
    print(f"{'─'*60}")
    print(f"  {'batch':>6}  {'peak_MB':>10}  {'peak_GB':>9}  {'status':>8}")
    results = {}
    for bs in batch_sizes:
        mem_mb = _measure(backbone, projector, predictor, bs, device, dtype)
        if mem_mb is None:
            print(f"  {bs:>6}  {'OOM':>10}  {'OOM':>9}  {'OOM':>8}")
            results[bs] = None
            break
        else:
            mem_gb = mem_mb / 1024
            status = "OK" if mem_mb < 38_000 else "tight" if mem_mb < 45_000 else "needs80G"
            print(f"  {bs:>6}  {mem_mb:>10.0f}  {mem_gb:>9.2f}  {status:>8}")
            results[bs] = mem_mb
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8,16")
    parser.add_argument("--dtype",       type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    dtype_map   = {"float32": torch.float32,
                   "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype       = dtype_map[args.dtype]
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("No CUDA device found — exiting.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU : {gpu_name}  ({gpu_mem:.1f} GB total)")
    print(f"PyTorch : {torch.__version__}")
    sdpa_backends = torch.backends.cuda.flash_sdp_enabled(), \
                    torch.backends.cuda.mem_efficient_sdp_enabled(), \
                    torch.backends.cuda.math_sdp_enabled()
    print(f"SDPA backends — flash: {sdpa_backends[0]}  "
          f"mem_efficient: {sdpa_backends[1]}  math: {sdpa_backends[2]}")

    print(f"\nLoading backbone + SimSiam projector/predictor...")
    from models.simsiam import Projector, Predictor
    backbone  = CTViTBackbone(checkpoint_path=str(CHECKPOINT), use_pre_vq=True).to(device)
    projector = Projector(in_dim=512, hidden_dim=2048, out_dim=2048, norm="ln").to(device)
    predictor = Predictor(in_dim=2048, hidden_dim=512, out_dim=2048, norm="ln").to(device)

    print("  Simulating FP SimSiam training step (2 views, forward + backward)\n")

    # ── standard attention ────────────────────────────────────────────────
    std_results = profile(backbone, projector, predictor, batch_sizes, device, dtype,
                          "Standard attention — SimSiam training step")

    # ── flash attention ───────────────────────────────────────────────────
    patch_flash(backbone)
    flash_results = profile(backbone, projector, predictor, batch_sizes, device, dtype,
                            "Flash attention (SDPA) — SimSiam training step")

    # ── comparison ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Comparison (dtype={args.dtype})")
    print(f"{'='*60}")
    print(f"  {'batch':>6}  {'std_GB':>8}  {'flash_GB':>10}  {'saving_GB':>10}  {'saving_%':>9}")
    for bs in batch_sizes:
        s = std_results.get(bs)
        f = flash_results.get(bs)
        if s is None and f is None:
            print(f"  {bs:>6}  {'OOM':>8}  {'OOM':>10}")
        elif s is None:
            print(f"  {bs:>6}  {'OOM':>8}  {f/1024:>10.2f}")
        elif f is None:
            print(f"  {bs:>6}  {s/1024:>8.2f}  {'OOM':>10}")
        else:
            saving_gb  = (s - f) / 1024
            saving_pct = 100 * (s - f) / s
            print(f"  {bs:>6}  {s/1024:>8.2f}  {f/1024:>10.2f}  "
                  f"{saving_gb:>10.2f}  {saving_pct:>8.1f}%")

    # ── verdict ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Verdict")
    print(f"{'='*60}")
    max_std_bs   = max((bs for bs, m in std_results.items()   if m is not None), default=0)
    max_flash_bs = max((bs for bs, m in flash_results.items() if m is not None), default=0)
    print(f"  Max batch size — standard : {max_std_bs}")
    print(f"  Max batch size — flash    : {max_flash_bs}")
    if max_flash_bs >= 8:
        print(f"  BatchNorm viability      : YES (bs={max_flash_bs} >= 8)")
    elif max_flash_bs >= 4:
        print(f"  BatchNorm viability      : MARGINAL (bs={max_flash_bs}, prefer >= 8)")
    else:
        print(f"  BatchNorm viability      : NO (bs={max_flash_bs} too small)")


if __name__ == "__main__":
    main()