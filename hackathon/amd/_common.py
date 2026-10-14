"""Study/metadata selection shared by the AMD CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from mri_core.rsna_inference_engine import InferenceEngineError
from mri_core.rsna_integration import discover_available_studies, load_rsna_metadata_safe
from mri_core.rsna_knee_dataset import RSNAKneeMetadata, load_rsna_metadata


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True, help="RSNA knee data dir (train.csv, raw/train_series, ...)")
    parser.add_argument("--dicom-root", help="DICOM root if not <data-root>/raw")
    parser.add_argument("--study-uid", help="StudyInstanceUID; default: first fully downloaded labeled study")


def load_metadata_and_study(data_root: str, dicom_root: Optional[str], study_uid: Optional[str]) -> tuple[RSNAKneeMetadata, str]:
    root = Path(data_root)
    metadata = load_rsna_metadata(root, dicom_root=dicom_root) if dicom_root else load_rsna_metadata_safe(root)
    if study_uid is None:
        available = discover_available_studies(root, metadata)
        if not available:
            raise InferenceEngineError(f"No fully downloaded labeled study under {root}")
        study_uid = available[0]
    return metadata, study_uid
