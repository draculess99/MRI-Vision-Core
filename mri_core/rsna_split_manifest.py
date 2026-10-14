"""Reproducible, auditable study-level 5-fold split manifests for RSNA knee training.

Wraps the existing split_labeled_studies_cv() (mri_core/rsna_knee_train.py) -- which is
left unmodified -- and records exactly which study UIDs landed in each fold, their label
distributions, the seed/splitter, and the library versions used, so a benchmark number
(e.g. the historical 0.6078 mean macro-AUROC) can be checked for structural reproducibility
without ever claiming the AUROC itself was independently re-measured.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

import numpy as np
import torch

from .rsna_knee_dataset import TARGET_COLUMNS
from .rsna_knee_train import split_labeled_studies_cv


class SplitIntegrityError(Exception):
    """A generated CV manifest violates study-level disjointness or coverage."""


def _package_version(name: str) -> str:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return "not installed"


def library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scikit_learn": _package_version("scikit-learn"),
        "iterative_stratification": _package_version("iterative-stratification"),
    }


def _label_counts(frame) -> dict:
    return {target: int(frame[target].sum()) for target in TARGET_COLUMNS}


def build_rsna_cv_manifest(metadata, *, n_splits: int, seed: int, preprocessing_config: dict | None = None) -> dict:
    """Build the full manifest: per-fold study UIDs (sorted), label distributions, and provenance.

    Does not train anything; only calls the existing deterministic splitter once per fold.
    """
    folds = []
    for fold in range(n_splits):
        train_frame, valid_frame = split_labeled_studies_cv(metadata, n_splits=n_splits, fold=fold, seed=seed)
        folds.append({
            "fold": fold,
            "train_study_ids": sorted(train_frame["StudyInstanceUID"].astype(str).tolist()),
            "valid_study_ids": sorted(valid_frame["StudyInstanceUID"].astype(str).tolist()),
            "train_size": len(train_frame),
            "valid_size": len(valid_frame),
            "train_label_counts": _label_counts(train_frame),
            "valid_label_counts": _label_counts(valid_frame),
        })

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_splits": n_splits,
        "seed": seed,
        "splitter": "MultilabelStratifiedKFold",
        "splitter_kwargs": {"shuffle": True, "random_state": seed},
        "target_columns": list(TARGET_COLUMNS),
        "total_labeled_studies": int(len(metadata.complete_label_train)),
        "preprocessing_config": preprocessing_config or {},
        "library_versions": library_versions(),
        "folds": folds,
        "benchmark_reproducibility_note": (
            "This manifest reproduces the study-level partition structure (sizes, "
            "disjointness, coverage) deterministically from train.csv + the given seed. "
            "It does NOT independently re-measure any reported AUROC; a macro-AUROC value "
            "from a prior run must still be re-trained to be verified."
        ),
    }
    return manifest


def verify_cv_manifest_integrity(manifest: dict) -> None:
    """Raise SplitIntegrityError if any fold violates study-level disjointness or coverage."""
    all_valid_sets = [set(fold["valid_study_ids"]) for fold in manifest["folds"]]
    total_labeled = manifest["total_labeled_studies"]

    for fold in manifest["folds"]:
        train_set = set(fold["train_study_ids"])
        valid_set = set(fold["valid_study_ids"])
        overlap = train_set & valid_set
        if overlap:
            raise SplitIntegrityError(
                f"Fold {fold['fold']}: {len(overlap)} studies appear in both train and valid: "
                f"{sorted(overlap)[:5]}..."
            )
        if len(train_set) != fold["train_size"] or len(valid_set) != fold["valid_size"]:
            raise SplitIntegrityError(f"Fold {fold['fold']}: duplicate study IDs within a split")

    for i in range(len(all_valid_sets)):
        for j in range(i + 1, len(all_valid_sets)):
            cross_overlap = all_valid_sets[i] & all_valid_sets[j]
            if cross_overlap:
                raise SplitIntegrityError(
                    f"Fold {i} and fold {j} validation sets overlap: {len(cross_overlap)} studies"
                )

    union = set().union(*all_valid_sets) if all_valid_sets else set()
    if len(union) != total_labeled:
        raise SplitIntegrityError(
            f"Validation folds cover {len(union)} studies, expected exactly {total_labeled} "
            "(every labeled study should appear in exactly one fold's validation set)"
        )


def write_cv_manifest(manifest: dict, path: Path) -> None:
    """Atomic write (temp file + rename), matching the project's checkpoint-write pattern."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
