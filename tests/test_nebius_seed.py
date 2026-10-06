"""Smoke test for MRI Evidence Council — Nebius × NVIDIA hackathon."""

import os
from pathlib import Path


def test_mri_core_exists():
    """Verify mri_core/ directory exists."""
    mri_core_path = Path(__file__).parent.parent / "mri_core"
    assert mri_core_path.exists(), "mri_core/ directory not found"
    assert mri_core_path.is_dir(), "mri_core/ is not a directory"


def test_nebius_readme_exists():
    """Verify hackathon/nebius_nvidia/README.md exists."""
    readme_path = (
        Path(__file__).parent.parent
        / "hackathon"
        / "nebius_nvidia"
        / "README.md"
    )
    assert readme_path.exists(), "hackathon/nebius_nvidia/README.md not found"


def test_nebius_readme_contains_evidence_council():
    """Verify README contains 'MRI Evidence Council'."""
    readme_path = (
        Path(__file__).parent.parent
        / "hackathon"
        / "nebius_nvidia"
        / "README.md"
    )
    with open(readme_path, "r") as f:
        content = f.read()
    assert (
        "MRI Evidence Council" in content
    ), "README does not contain 'MRI Evidence Council'"
