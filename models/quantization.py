"""
SSQL-style quantization utilities.

FakeQuantize applies a symmetric uniform quantizer with straight-through
estimator (STE) so gradients flow unchanged through the quantization step.

quantized_forward() is a context manager that temporarily monkey-patches
all nn.Linear modules inside a list of nn.Modules so their weights and
input activations are fake-quantized for that forward pass only.

Usage:
    w_bits, a_bits = sample_bits()
    with quantized_forward([backbone, projector], w_bits, a_bits):
        z_q = projector(backbone(x))
    # Weights are fully restored after the context exits.
"""

import random
from contextlib import contextmanager
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core quantizer (STE)
# ---------------------------------------------------------------------------

class _FakeQuantize(torch.autograd.Function):
    """Symmetric min-max uniform quantizer with straight-through estimator."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        x_min = x.detach().min()
        x_max = x.detach().max()
        scale = (x_max - x_min).clamp(min=1e-8) / (2 ** n_bits - 1)
        x_q = torch.round((x - x_min) / scale) * scale + x_min
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # STE: pass gradient unchanged


def fake_quantize(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    return _FakeQuantize.apply(x, n_bits)


# ---------------------------------------------------------------------------
# Bit-width sampler
# ---------------------------------------------------------------------------

def sample_bits(
    w_bits_range: Tuple[int, int] = (2, 8),
    a_bits_range: Tuple[int, int] = (4, 8),
) -> Tuple[int, int]:
    """Sample random bit-widths for weights and activations independently."""
    w_bits = random.randint(*w_bits_range)
    a_bits = random.randint(*a_bits_range)
    return w_bits, a_bits


# ---------------------------------------------------------------------------
# Context manager: quantized forward pass
# ---------------------------------------------------------------------------

@contextmanager
def quantized_forward(
    modules: List[nn.Module],
    w_bits: int,
    a_bits: int,
):
    """
    Temporarily replace the forward() of all nn.Linear layers inside
    `modules` so that weights are fake-quantized to w_bits and input
    activations to a_bits.  Restores original forwards on exit.

    Only nn.Linear is patched; BatchNorm and LayerNorm are left in full
    precision (standard QAT practice).  The VQ codebook in CTViT is a
    discrete lookup and is not affected.
    """
    patched: List[Tuple[nn.Linear, object]] = []

    for module in modules:
        for layer in module.modules():
            if not isinstance(layer, nn.Linear):
                continue

            orig_forward = layer.forward

            # Capture layer/bit-widths via default-arg binding to avoid
            # closure issues in the loop.
            def make_q_forward(lyr, wb, ab):
                def q_forward(x):
                    q_w = fake_quantize(lyr.weight, wb)
                    q_x = fake_quantize(x, ab)
                    return F.linear(q_x, q_w, lyr.bias)
                return q_forward

            layer.forward = make_q_forward(layer, w_bits, a_bits)
            patched.append((layer, orig_forward))

    try:
        yield
    finally:
        for layer, orig in patched:
            layer.forward = orig
