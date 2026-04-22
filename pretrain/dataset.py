"""
CT pretraining dataset.

Returns two independently augmented views of each scan — no labels needed.
Scan discovery and NIfTI preprocessing are adapted from the battle-tested
CTReportDatasetinfer in the CT-CLIP zero-shot inference project.
"""

import os
from functools import partial
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import Dataset

from pretrain.augmentations import CTAugmentation


# ---------------------------------------------------------------------------
# Preprocessing (shared logic from data_inference_nii.py)
# ---------------------------------------------------------------------------

_TARGET_SPACING = (1.5, 0.75, 0.75)   # (z, x, y) in mm
_TARGET_SHAPE   = (480, 480, 240)       # (H, W, D)
_HU_RANGE       = (-1000.0, 1000.0)


def _resize_volume(tensor: torch.Tensor, current: tuple, target: tuple) -> np.ndarray:
    """Trilinear resize of a (1, 1, D, H, W) tensor to match target spacing."""
    orig = tensor.shape[2:]
    factors = [current[i] / target[i] for i in range(len(orig))]
    new_shape = [int(orig[i] * factors[i]) for i in range(len(orig))]
    return F.interpolate(
        tensor, size=new_shape, mode="trilinear", align_corners=False
    ).cpu().numpy()


def _nii_to_tensor(path: str, meta_row: dict) -> torch.Tensor:
    """
    Load a NIfTI file and return a preprocessed (1, D, H, W) float32 tensor.
    Values are in [-1, 1] (HU / 1000 after clipping).
    """
    nii = nib.load(path)
    img = nii.get_fdata()

    header = nii.header
    zooms = header.get_zooms()[:3]
    slope_inter = header.get_slope_inter()

    slope     = meta_row.get("RescaleSlope",
                              slope_inter[0] if slope_inter[0] is not None else 1.0)
    intercept = meta_row.get("RescaleIntercept",
                              slope_inter[1] if slope_inter[1] is not None else 0.0)

    if "XYSpacing" in meta_row:
        xy_spacing = float(str(meta_row["XYSpacing"])[1:][:-2].split(",")[0])
    else:
        xy_spacing = float(zooms[0])

    z_spacing = float(meta_row["ZSpacing"]) if "ZSpacing" in meta_row else float(zooms[2])

    img = slope * img + intercept
    img = np.clip(img, *_HU_RANGE)
    img = img.transpose(2, 0, 1)   # (D, H, W)

    t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    img = _resize_volume(t, (z_spacing, xy_spacing, xy_spacing), _TARGET_SPACING)[0][0]
    img = np.transpose(img, (1, 2, 0))  # (H, W, D)
    img = (img / 1000.0).astype(np.float32)

    tensor = torch.tensor(img)  # (H, W, D)
    dh, dw, dd = _TARGET_SHAPE
    h, w, d = tensor.shape

    h_start = max((h - dh) // 2, 0)
    w_start = max((w - dw) // 2, 0)
    d_start = max((d - dd) // 2, 0)
    tensor = tensor[h_start: h_start + dh, w_start: w_start + dw, d_start: d_start + dd]

    pad_h0 = (dh - tensor.size(0)) // 2
    pad_h1 = dh - tensor.size(0) - pad_h0
    pad_w0 = (dw - tensor.size(1)) // 2
    pad_w1 = dw - tensor.size(1) - pad_w0
    pad_d0 = (dd - tensor.size(2)) // 2
    pad_d1 = dd - tensor.size(2) - pad_d0
    tensor = F.pad(tensor, (pad_d0, pad_d1, pad_w0, pad_w1, pad_h0, pad_h1), value=-1)

    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
    return tensor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _normalize_name(name) -> Optional[str]:
    if pd.isna(name):
        return None
    name = str(name).strip()
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name or None


def _first_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


class CTPretrainDataset(Dataset):
    """
    Yields (view1, view2) pairs of independently augmented CT volumes.

    Parameters
    ----------
    data_folder   : directory containing *.nii.gz files (flat or nested)
    reports_file  : XLSX or CSV with accession IDs; used to whitelist valid scans
    meta_file     : optional CSV with per-scan spacing / rescale metadata
    augmentation  : callable applied independently to each view; defaults to
                    CTAugmentation with standard parameters
    """

    def __init__(
        self,
        data_folder: str,
        reports_file: str,
        meta_file: Optional[str] = None,
        augmentation=None,
    ):
        self.data_folder = data_folder
        self.augmentation = augmentation or CTAugmentation()

        max_env = os.environ.get("CT_CLIP_MAX_SAMPLES")
        self.max_samples = int(max_env) if max_env else None

        self._valid_accessions = self._load_accessions(reports_file)
        self._meta = self._load_meta(meta_file)
        self.samples = self._prepare_samples()

    # ------------------------------------------------------------------

    def _load_accessions(self, reports_file: str) -> set:
        suffix = Path(reports_file).suffix.lower()
        df = pd.read_excel(reports_file) if suffix in {".xlsx", ".xls"} else pd.read_csv(reports_file)
        id_col = _first_column(df, ["VolumeName", "study id", "StudyInstanceUID"])
        if id_col is None:
            raise ValueError("Reports file must contain 'VolumeName' or 'study id'.")
        return {_normalize_name(v) for v in df[id_col] if _normalize_name(v) is not None}

    def _load_meta(self, meta_file: Optional[str]) -> dict:
        if meta_file is None:
            return {}
        df = pd.read_csv(meta_file)
        id_col = _first_column(df, ["VolumeName", "study id", "StudyInstanceUID"])
        if id_col is None:
            return {}
        return {
            _normalize_name(row[id_col]): row.to_dict()
            for _, row in df.iterrows()
            if _normalize_name(row[id_col]) is not None
        }

    def _prepare_samples(self) -> list:
        root = Path(self.data_folder)
        samples = []
        for nii_file in tqdm.tqdm(sorted(root.rglob("*.nii.gz")), desc="indexing scans"):
            name = _normalize_name(nii_file.name)
            if name not in self._valid_accessions:
                continue
            samples.append(str(nii_file))
            if self.max_samples and len(samples) >= self.max_samples:
                break
        return samples

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]
        name = _normalize_name(Path(path).name)
        meta_row = self._meta.get(name, {})
        tensor = _nii_to_tensor(path, meta_row)   # (1, D, H, W)
        view1 = self.augmentation(tensor.clone())
        view2 = self.augmentation(tensor.clone())
        return view1, view2
