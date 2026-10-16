"""Environment/.env-driven MRI data_root and checkpoint path resolution for the AMD app.

Reuses `hackathon.amd.llm_config.load_dotenv_if_present()` -- the same `.env` loading
mechanism already used for the optional LLM layer -- rather than introducing a second,
independent configuration system. Precedence is identical to that module: the real process
environment wins, then an optional local `.env` file fills in anything still unset, then
these resolve to `None` (not a path) when neither is set.

Cross-platform by construction: both variables are read as plain strings and wrapped in
`pathlib.Path`, which renders OS-native separators -- a Windows value like
`D:\\Work\\...\\rsna-knee` and a Linux value like `/app/data/rsna-knee` both work with no
OS-specific branching anywhere in this module or its caller.

All checks here are deliberately cheap: file/directory existence, a file extension, and
whether one small metadata CSV is present -- never parsing that CSV, never loading DICOM
series, and never loading checkpoint weights. That work happens only once a user actually
requests inference, in hackathon/amd/streamlit_app.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .llm_config import load_dotenv_if_present

EXPECTED_CHECKPOINT_SUFFIX = ".ckpt"
REQUIRED_METADATA_FILENAME = "train.csv"


@dataclass(frozen=True)
class DataPathsConfig:
    data_root: Optional[Path]
    checkpoint: Optional[Path]
    dotenv_loaded: bool

    def to_dict(self) -> dict:
        return {
            "data_root": str(self.data_root) if self.data_root else None,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "dotenv_loaded": self.dotenv_loaded,
        }


def resolve_data_paths(*, use_dotenv: bool = True, dotenv_path: Optional[Path] = None) -> DataPathsConfig:
    """Resolve AMD_KAGGLE_RSNA_ROOT / AMD_KAGGLE_CHECKPOINT.

    Reads the real process environment first; `load_dotenv_if_present()` only fills in a
    variable that is still unset there (it never overrides one already present), so a real
    environment variable -- e.g. a Railway Variable -- always wins over `.env`.
    """
    dotenv_loaded = load_dotenv_if_present(dotenv_path) if use_dotenv else False
    data_root = os.environ.get("AMD_KAGGLE_RSNA_ROOT") or None
    checkpoint = os.environ.get("AMD_KAGGLE_CHECKPOINT") or None
    return DataPathsConfig(
        data_root=Path(data_root) if data_root else None,
        checkpoint=Path(checkpoint) if checkpoint else None,
        dotenv_loaded=dotenv_loaded,
    )


def validate_data_root(data_root: Path) -> List[str]:
    """Cheap, structural checks only. Returns human-readable problem strings; an empty list
    means the directory looks usable. Never reads train.csv's contents, only its presence."""
    problems: List[str] = []
    if not data_root.is_dir():
        problems.append(f"Data directory not found: {data_root}")
        return problems
    metadata_path = data_root / REQUIRED_METADATA_FILENAME
    if not metadata_path.is_file():
        problems.append(
            f"Required dataset metadata not found: {metadata_path} "
            f"(expected alongside raw/train_series/... under {data_root})."
        )
    return problems


def validate_checkpoint(checkpoint: Path) -> List[str]:
    """Cheap, structural checks only -- never loads or reads the checkpoint's weights."""
    problems: List[str] = []
    if not checkpoint.is_file():
        problems.append(f"Checkpoint file not found: {checkpoint}")
        return problems
    if checkpoint.suffix.lower() != EXPECTED_CHECKPOINT_SUFFIX:
        problems.append(
            f"Checkpoint does not have the expected {EXPECTED_CHECKPOINT_SUFFIX!r} extension: {checkpoint}"
        )
    return problems


def validate_data_paths(data_root: Path, checkpoint: Path) -> List[str]:
    """Convenience wrapper: every problem with either path, in one list."""
    return validate_data_root(data_root) + validate_checkpoint(checkpoint)
