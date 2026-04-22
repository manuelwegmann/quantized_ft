"""
3D CT augmentations operating on tensors of shape (1, D, H, W).
All transforms are in-place-safe (they return new tensors).
"""

import random

import torch
import torch.nn.functional as F


class RandomFlip3D:
    """Randomly flip along any subset of the spatial axes."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, D, H, W); dims 1,2,3 are spatial
        for dim in [1, 2, 3]:
            if random.random() < self.p:
                x = x.flip(dim)
        return x


class RandomCrop3D:
    """
    Randomly crop a sub-volume then resize back to the original shape.
    crop_ratio controls the minimum fraction of each axis retained.
    """

    def __init__(self, crop_ratio: float = 0.85):
        self.crop_ratio = crop_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, d, h, w = x.shape
        cd = max(1, int(d * self.crop_ratio))
        ch = max(1, int(h * self.crop_ratio))
        cw = max(1, int(w * self.crop_ratio))
        sd = random.randint(0, d - cd)
        sh = random.randint(0, h - ch)
        sw = random.randint(0, w - cw)
        cropped = x[:, sd: sd + cd, sh: sh + ch, sw: sw + cw]
        # Restore original shape with trilinear interpolation
        cropped = F.interpolate(
            cropped.unsqueeze(0),
            size=(d, h, w),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        return cropped


class IntensityJitter:
    """
    Random multiplicative scale + additive shift on HU-normalised values.
    Values are in [-1, 1] after preprocessing, so shifts are small.
    """

    def __init__(
        self,
        scale_range: tuple = (0.9, 1.1),
        shift_range: tuple = (-0.05, 0.05),
    ):
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = random.uniform(*self.scale_range)
        shift = random.uniform(*self.shift_range)
        return x * scale + shift


class GaussianNoise:
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class CTAugmentation:
    """Compose a standard augmentation pipeline for CT SSL pretraining."""

    def __init__(
        self,
        crop_ratio: float = 0.85,
        flip_p: float = 0.5,
        intensity_jitter: bool = True,
        gaussian_noise: bool = True,
        noise_std: float = 0.01,
    ):
        self.transforms = [
            RandomCrop3D(crop_ratio=crop_ratio),
            RandomFlip3D(p=flip_p),
        ]
        if intensity_jitter:
            self.transforms.append(IntensityJitter())
        if gaussian_noise:
            self.transforms.append(GaussianNoise(std=noise_std))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x
