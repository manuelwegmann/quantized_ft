"""Central path configuration for quantized_ft.

Set the CT_CLIP_ROOT environment variable to point at your CT-CLIP repository.
Defaults to ../CT-CLIP relative to this file (sibling directory convention).

    export CT_CLIP_ROOT=/path/to/CT-CLIP
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

_ct_clip_env = os.environ.get("CT_CLIP_ROOT")
CT_CLIP_ROOT = Path(_ct_clip_env) if _ct_clip_env else PROJECT_ROOT.parent / "CT-CLIP"

# Ensure env vars are set so os.path.expandvars works in YAML loaders
os.environ.setdefault("CT_CLIP_ROOT", str(CT_CLIP_ROOT))
os.environ.setdefault("QFT_ROOT", str(PROJECT_ROOT))

for _p in [
    str(CT_CLIP_ROOT / "transformer_maskgit"),
    str(CT_CLIP_ROOT / "CT_CLIP"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHECKPOINT   = CT_CLIP_ROOT / "checkpoints" / "CT-CLIP_v2.pt"
DATA_FOLDER  = CT_CLIP_ROOT / "data" / "merlin_data"
REPORTS_FILE = CT_CLIP_ROOT / "data" / "reports_final.xlsx"
LABELS_FILE  = CT_CLIP_ROOT / "data" / "zero_shot_findings_disease_cls.csv"
META_FILE    = CT_CLIP_ROOT / "data" / "metadata.csv"
