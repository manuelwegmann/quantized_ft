"""
Memory profiling: float32 vs bfloat16 autocast for SimSiam training.

Runs a full SimSiam FP training step (2 views, forward + backward) at
increasing batch sizes under three configurations:
  1. float32          — current baseline
  2. bfloat16 autocast — mixed precision (bfloat16 activations, float32 params)
  3. bfloat16 + Flash  — combined (both changes together)

bfloat16 is preferred over float16 for training because it preserves the
dynamic range of float32 (same exponent bits), making gradient underflow
far less likely without needing a GradScaler.

Usage:
    python scripts/profile_mixed_precision.py
    python scripts/profile_mixed_precision.py --batch_sizes 1,2,4,8
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
from models.simsiam import Projector, Predictor
from pretrain.loss import negative_cosine_similarity
from paths import CHECKPOINT


# ── Flash Attention patch (reused from profile_flash_attention.py) ────────────

def _flash_forward(self, x, mask=None, context=None, attn_bias=None):
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
    q, k = map(lambda t: F.normalize(t, dim=-1), (q, k))
    q = q * self.q_scale
    k = k * self.k_scale
    attn_mask = None
    if exists(attn_bias):
        attn_mask = F.pad(attn_bias, (self.num_null_kv, 0), value=0.0)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=float(self.scale),
    )
    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)


def _patch_flash(backbone):
    from transformer_maskgit.attention import Attention
    for module in backbone.modules():
        if isinstance(module, Attention):
            module.forward = types.MethodType(_flash_forward, module)


# ── measurement ───────────────────────────────────────────────────────────────

def _measure(backbone, projector, predictor, batch_size, device, amp_dtype=None):
    """
    One SimSiam FP step: 2-view forward + backward.
    amp_dtype: None = float32, torch.bfloat16 = mixed precision.
    Returns peak memory in MB, or None on OOM.
    """
    backbone.train()
    projector.train()
    predictor.train()

    x1 = torch.randn(batch_size, 1, 240, 480, 480, device=device)
    x2 = torch.randn(batch_size, 1, 240, 480, 480, device=device)

    for m in (backbone, projector, predictor):
        for p in m.parameters():
            p.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        if amp_dtype:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                z1 = projector(backbone(x1))
                z2 = projector(backbone(x2))
                p1 = predictor(z1)
                p2 = predictor(z2)
                loss = (
                    negative_cosine_similarity(p1, z2.detach())
                    + negative_cosine_similarity(p2, z1.detach())
                )
        else:
            z1 = projector(backbone(x1))
            z2 = projector(backbone(x2))
            p1 = predictor(z1)
            p2 = predictor(z2)
            loss = (
                negative_cosine_similarity(p1, z2.detach())
                + negative_cosine_similarity(p2, z1.detach())
            )

        loss.backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / 1024**2

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


# ── profiling ─────────────────────────────────────────────────────────────────

def profile(backbone, projector, predictor, batch_sizes, device, amp_dtype, label):
    dtype_str = str(amp_dtype).replace("torch.", "") if amp_dtype else "float32"
    print(f"\n{'─'*64}")
    print(f"  {label}  ({dtype_str})")
    print(f"{'─'*64}")
    print(f"  {'batch':>6}  {'peak_MB':>10}  {'peak_GB':>9}  {'status':>10}")
    results = {}
    for bs in batch_sizes:
        mem = _measure(backbone, projector, predictor, bs, device, amp_dtype)
        if mem is None:
            print(f"  {bs:>6}  {'OOM':>10}  {'OOM':>9}  {'OOM':>10}")
            results[bs] = None
            break
        gb     = mem / 1024
        status = "OK" if mem < 38_000 else "tight(40G)" if mem < 45_000 else "needs80G"
        print(f"  {bs:>6}  {mem:>10.0f}  {gb:>9.2f}  {status:>10}")
        results[bs] = mem
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", default="1,2,4,8")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("No CUDA — exiting.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_gb   = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU     : {gpu_name}  ({gpu_gb:.1f} GB)")
    print(f"PyTorch : {torch.__version__}")

    print("\nLoading backbone + SimSiam projector/predictor...")
    backbone  = CTViTBackbone(checkpoint_path=str(CHECKPOINT), use_pre_vq=True).to(device)
    projector = Projector(in_dim=512, hidden_dim=2048, out_dim=2048, norm="ln").to(device)
    predictor = Predictor(in_dim=2048, hidden_dim=512, out_dim=2048, norm="ln").to(device)
    print("  2-view SimSiam FP step (forward + backward)\n")

    # ── three configurations ──────────────────────────────────────────────
    fp32_res   = profile(backbone, projector, predictor, batch_sizes, device,
                         None,               "1. float32 baseline")

    bf16_res   = profile(backbone, projector, predictor, batch_sizes, device,
                         torch.bfloat16,     "2. bfloat16 autocast")

    _patch_flash(backbone)
    bf16f_res  = profile(backbone, projector, predictor, batch_sizes, device,
                         torch.bfloat16,     "3. bfloat16 autocast + Flash attention")

    # ── comparison table ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Comparison")
    print(f"{'='*72}")
    print(f"  {'bs':>4}  {'fp32_GB':>9}  {'bf16_GB':>9}  {'bf16+fa_GB':>11}"
          f"  {'bf16_save%':>11}  {'bf16+fa_save%':>14}")
    for bs in batch_sizes:
        fp  = fp32_res.get(bs)
        bf  = bf16_res.get(bs)
        bff = bf16f_res.get(bs)
        def fmt(v): return f"{v/1024:>9.2f}" if v else f"{'OOM':>9}"
        def pct(v): return f"{100*(fp-v)/fp:>10.1f}%" if (v and fp) else f"{'n/a':>10}"
        print(f"  {bs:>4}  {fmt(fp)}  {fmt(bf)}  {fmt(bff):>11}"
              f"  {pct(bf):>11}  {pct(bff):>14}")

    # ── verdict ───────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Verdict")
    print(f"{'='*72}")
    for label, res in [("float32", fp32_res), ("bfloat16", bf16_res),
                       ("bfloat16+flash", bf16f_res)]:
        max_bs = max((bs for bs, m in res.items() if m is not None), default=0)
        bn_ok  = "YES" if max_bs >= 8 else "MARGINAL" if max_bs >= 4 else "NO"
        print(f"  {label:<18} max_bs={max_bs}  BatchNorm viable: {bn_ok}")


if __name__ == "__main__":
    main()