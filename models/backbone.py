import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

_CT_CLIP_ROOT = Path("/home/nlr950/Dir/CT-CLIP")
for _p in [
    str(_CT_CLIP_ROOT / "transformer_maskgit"),
    str(_CT_CLIP_ROOT / "CT_CLIP"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformer_maskgit.ctvit import CTViT  # noqa: E402

EMBED_DIM = 512

_CTVIT_KWARGS = dict(
    dim=512,
    codebook_size=8192,
    image_size=480,
    patch_size=20,
    temporal_patch_size=10,
    spatial_depth=4,
    temporal_depth=4,
    dim_head=32,
    heads=8,
)


class CTViTBackbone(nn.Module):
    """
    CTViT encoder initialized from CT-CLIP_v2.pt (vision weights only).

    Input:  (B, 1, D, H, W)  normalized CT volume
    Output: (B, 512)          global mean-pooled token embedding

    Args:
        checkpoint_path: path to CT-CLIP_v2.pt
        use_pre_vq: if True, bypass the VQ codebook step and use pre-VQ
                    transformer tokens; default False (post-VQ, faithful
                    to the pretrained representation)
    """

    def __init__(self, checkpoint_path: Optional[str] = None, use_pre_vq: bool = False):
        super().__init__()
        self.use_pre_vq = use_pre_vq
        self.ctvit = CTViT(**_CTVIT_KWARGS)
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        else:
            print("[backbone] no checkpoint — using random initialisation")

    def _load_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # CT-CLIP_v2.pt is a full CTCLIP state dict; image encoder lives
        # under 'visual_transformer.*' keys.
        img_state = {
            k[len("visual_transformer."):]: v
            for k, v in ckpt.items()
            if k.startswith("visual_transformer.")
        }
        if not img_state:
            # Checkpoint already contains only CTViT weights
            img_state = ckpt
        missing, unexpected = self.ctvit.load_state_dict(img_state, strict=False)
        print(f"[backbone] loaded {len(img_state)} tensors from {checkpoint_path}")
        if missing:
            print(f"[backbone] missing  ({len(missing)}): {missing[:3]}")
        if unexpected:
            print(f"[backbone] unexpected ({len(unexpected)}): {unexpected[:3]}")

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers (used by downstream linear probe)
    # ------------------------------------------------------------------

    def freeze(self):
        for p in self.ctvit.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.ctvit.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, D, H, W)
        Returns: (B, 512)
        """
        assert tuple(x.shape[3:]) == self.ctvit.image_size, \
            f"Expected spatial dims (H, W) {self.ctvit.image_size}, got {tuple(x.shape[3:])}"
        if self.use_pre_vq:
            # Patch embed → spatial+temporal transformers, skip VQ codebook
            tokens = self.ctvit.to_patch_emb(x)   # (B, t, h, w, dim)
            tokens = self.ctvit.encode(tokens)      # (B, t, h, w, dim)
        else:
            # Full forward up to and including VQ step (STE gradients flow)
            tokens = self.ctvit(x, return_encoded_tokens=True)  # (B, t, h, w, 512)

        return tokens.mean(dim=(1, 2, 3))  # (B, 512)
