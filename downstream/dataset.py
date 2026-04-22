"""
Merlin labeled dataset for downstream classification evaluation.

Produces a fixed 70 / 15 / 15 train / val / test split, reproducibly
seeded per accession ID (not per file index) so the same scan always
lands in the same split regardless of scan discovery order.

Label file: data/zero_shot_findings_disease_cls.csv  (30 binary columns)
"""

import os
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import Dataset

# Reuse the preprocessing helpers from the pretrain dataset
from pretrain.dataset import _nii_to_tensor, _normalize_name, _first_column


class MerlinDataset(Dataset):
    """
    Parameters
    ----------
    data_folder  : directory containing *.nii.gz files
    reports_file : XLSX or CSV that maps accession IDs to scan text
                   (used only to whitelist accessions that have reports)
    labels_file  : CSV with accession IDs + 30 binary label columns
    meta_file    : optional CSV with per-scan spacing metadata
    split        : 'train' | 'val' | 'test'
    seed         : RNG seed for the fixed split (default 42)
    """

    SPLIT_RATIOS = (0.70, 0.15, 0.15)

    def __init__(
        self,
        data_folder: str,
        reports_file: str,
        labels_file: str,
        meta_file: Optional[str] = None,
        split: str = "train",
        seed: int = 42,
    ):
        assert split in {"train", "val", "test"}
        self.split = split

        max_env = os.environ.get("CT_CLIP_MAX_SAMPLES")
        self.max_samples = int(max_env) if max_env else None

        self._valid_accessions = self._load_report_accessions(reports_file)
        self._label_lookup, self.label_names = self._load_labels(labels_file)
        self._meta = self._load_meta(meta_file)

        all_samples = self._discover_samples(data_folder)
        self.samples = self._split(all_samples, split, seed)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_report_accessions(self, reports_file: str) -> set:
        suffix = Path(reports_file).suffix.lower()
        df = (
            pd.read_excel(reports_file)
            if suffix in {".xlsx", ".xls"}
            else pd.read_csv(reports_file)
        )
        id_col = _first_column(df, ["VolumeName", "study id", "StudyInstanceUID"])
        if id_col is None:
            raise ValueError("Reports file must contain 'VolumeName' or 'study id'.")
        return {_normalize_name(v) for v in df[id_col] if _normalize_name(v) is not None}

    def _load_labels(self, labels_file: str):
        df = pd.read_csv(labels_file)
        id_col = _first_column(df, ["VolumeName", "study id", "StudyInstanceUID"])
        if id_col is None:
            raise ValueError("Labels file must contain 'VolumeName' or 'study id'.")
        label_cols = [c for c in df.columns if c != id_col]
        lookup = {}
        for _, row in df.iterrows():
            name = _normalize_name(row[id_col])
            if name is None:
                continue
            lookup[name] = pd.to_numeric(row[label_cols], errors="coerce").to_numpy(
                dtype=np.float32
            )
        return lookup, label_cols

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

    def _discover_samples(self, data_folder: str) -> list:
        """Return sorted list of (path, label_array) for all valid scans."""
        root = Path(data_folder)
        samples = []
        for nii_file in tqdm.tqdm(sorted(root.rglob("*.nii.gz")), desc="indexing scans"):
            name = _normalize_name(nii_file.name)
            if name not in self._valid_accessions:
                continue
            if name not in self._label_lookup:
                continue
            samples.append((str(nii_file), self._label_lookup[name]))
            if self.max_samples and len(samples) >= self.max_samples:
                break
        return samples

    # ------------------------------------------------------------------
    # Deterministic split by accession name
    # ------------------------------------------------------------------

    def _split(self, all_samples: list, split: str, seed: int) -> list:
        accessions = sorted({_normalize_name(Path(p).name) for p, _ in all_samples})
        rng = np.random.RandomState(seed)
        rng.shuffle(accessions)

        n = len(accessions)
        n_train = int(self.SPLIT_RATIOS[0] * n)
        n_val   = int(self.SPLIT_RATIOS[1] * n)

        if split == "train":
            keep = set(accessions[:n_train])
        elif split == "val":
            keep = set(accessions[n_train: n_train + n_val])
        else:
            keep = set(accessions[n_train + n_val:])

        return [(p, lbl) for p, lbl in all_samples if _normalize_name(Path(p).name) in keep]

    # ------------------------------------------------------------------

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, labels = self.samples[index]
        name = _normalize_name(Path(path).name)
        meta_row = self._meta.get(name, {})
        tensor = _nii_to_tensor(path, meta_row)                 # (1, D, H, W)
        return tensor, torch.tensor(labels, dtype=torch.float32)
