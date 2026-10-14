"""Reusable, non-diagnostic RSNA knee MRI inference engine.

Wraps the existing, tested inference primitives (run_inference / run_ensemble_inference in
rsna_knee_inference.py / rsna_knee_ensemble.py) with a single structured result type,
checkpoint provenance (SHA-256 + epoch + fold), latency measurement, and an explicit
"unable_to_evaluate" status for missing/unreliable inputs -- instead of each caller (a
Streamlit app, a hackathon branch, a test) re-implementing checkpoint loading, error
handling, or provenance tracking on its own.

This module performs no training and introduces no new model code: RSNAKneeCNN, the
DICOM/series loading, and the preprocessing pipeline are all reused exactly as-is.

Non-diagnostic notice: every probability this module returns is a raw sigmoid output from
a research model trained on 46-48 studies per fold. It is not a calibrated confidence score,
not a clinical finding, and not a diagnosis. See NON_DIAGNOSTIC_DISCLAIMER below.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .rsna_knee_dataset import TARGET_COLUMNS, RSNAKneeMetadata
from .rsna_knee_inference import (
    ArchitectureMismatchError,
    CheckpointNotFoundError,
    MalformedCheckpointError,
    RSNAInferenceError,
    StudyUnavailableError,
    TargetMismatchError,
    run_inference,
)
from .rsna_knee_ensemble import run_ensemble_inference

NON_DIAGNOSTIC_DISCLAIMER = (
    "Research prototype output. These are raw model probabilities, not calibrated "
    "confidence scores, not clinical findings, and not a diagnosis. Not for clinical "
    "decision-making. See the deterministic quality report (mri_core.decision) for the "
    "separate, model-independent image-quality assessment."
)

STATUS_OK = "ok"
STATUS_UNABLE_TO_EVALUATE = "unable_to_evaluate"
STATUS_ERROR = "error"


class InferenceEngineError(Exception):
    """Raised for configuration-level failures (bad checkpoints, architecture mismatch) --
    as opposed to STATUS_UNABLE_TO_EVALUATE, which is returned (not raised) for input-side
    problems like an incomplete local study."""


def sha256_of_file(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelProvenance:
    """Identifies exactly which trained weights produced a result."""

    checkpoint_path: str
    checkpoint_sha256: str
    epoch: Optional[int]
    fold: Optional[int]
    target_columns: tuple

    def to_dict(self) -> dict:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "epoch": self.epoch,
            "fold": self.fold,
            "target_columns": list(self.target_columns),
        }


@dataclass(frozen=True)
class StudyInferenceResult:
    """The single structured result type every caller (Streamlit app, hackathon branch,
    test) should use instead of calling run_inference/run_ensemble_inference directly."""

    study_uid: str
    status: str  # STATUS_OK | STATUS_UNABLE_TO_EVALUATE | STATUS_ERROR
    probabilities: Optional[Dict[str, float]]
    is_ensemble: bool
    model_provenance: List[ModelProvenance]
    latency_seconds: float
    warnings: List[str] = field(default_factory=list)
    review_required: bool = False
    disclaimer: str = NON_DIAGNOSTIC_DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "study_uid": self.study_uid,
            "status": self.status,
            "probabilities": self.probabilities,
            "is_ensemble": self.is_ensemble,
            "model_provenance": [p.to_dict() for p in self.model_provenance],
            "latency_seconds": self.latency_seconds,
            "warnings": list(self.warnings),
            "review_required": self.review_required,
            "disclaimer": self.disclaimer,
        }


def _load_provenance(checkpoint_path: Path) -> ModelProvenance:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise InferenceEngineError(f"No checkpoint file at {checkpoint_path}")
    try:
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise InferenceEngineError(f"Failed to read checkpoint {checkpoint_path}: {exc}") from exc
    return ModelProvenance(
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=sha256_of_file(checkpoint_path),
        epoch=data.get("epoch") if isinstance(data, dict) else None,
        fold=(data.get("config", {}) or {}).get("cv_fold") if isinstance(data, dict) else None,
        target_columns=tuple(data.get("target_columns", TARGET_COLUMNS)) if isinstance(data, dict) else TARGET_COLUMNS,
    )


def run_single_model_inference(
    metadata: RSNAKneeMetadata,
    checkpoint_path: Path | str,
    study_uid: str,
    *,
    device: str = "cpu",
) -> StudyInferenceResult:
    """Run one trusted checkpoint on one study. Never raises for input-side problems --
    those come back as STATUS_UNABLE_TO_EVALUATE. Raises InferenceEngineError only for
    checkpoint/configuration problems, which are not about this particular study's input."""
    checkpoint_path = Path(checkpoint_path)
    provenance = _load_provenance(checkpoint_path)

    started = time.perf_counter()
    try:
        probabilities = run_inference(metadata, checkpoint_path, study_uid, device=device)
    except StudyUnavailableError as exc:
        return StudyInferenceResult(
            study_uid=str(study_uid), status=STATUS_UNABLE_TO_EVALUATE, probabilities=None,
            is_ensemble=False, model_provenance=[provenance],
            latency_seconds=time.perf_counter() - started,
            warnings=[f"Study unavailable: {exc}"], review_required=True,
        )
    except (CheckpointNotFoundError, MalformedCheckpointError, TargetMismatchError, ArchitectureMismatchError) as exc:
        raise InferenceEngineError(f"Checkpoint configuration problem: {exc}") from exc
    except RSNAInferenceError as exc:
        raise InferenceEngineError(f"Unexpected inference failure: {exc}") from exc

    _validate_probabilities(probabilities)
    return StudyInferenceResult(
        study_uid=str(study_uid), status=STATUS_OK, probabilities=probabilities,
        is_ensemble=False, model_provenance=[provenance],
        latency_seconds=time.perf_counter() - started, warnings=[], review_required=False,
    )


def run_ensemble_model_inference(
    metadata: RSNAKneeMetadata,
    fold_checkpoint_paths: Sequence[Path | str],
    study_uid: str,
    *,
    device: str = "cpu",
) -> StudyInferenceResult:
    """Run the full 5-fold ensemble (arithmetic mean of sigmoid probabilities) on one study.

    No partial ensemble is ever returned: if any fold's checkpoint is missing, malformed,
    or incompatible, that is a configuration problem (InferenceEngineError), not a
    per-study input problem.
    """
    fold_checkpoint_paths = [Path(p) for p in fold_checkpoint_paths]
    provenances = [_load_provenance(p) for p in fold_checkpoint_paths]

    started = time.perf_counter()
    try:
        probabilities = run_ensemble_inference(metadata, fold_checkpoint_paths, study_uid, device=device)
    except StudyUnavailableError as exc:
        return StudyInferenceResult(
            study_uid=str(study_uid), status=STATUS_UNABLE_TO_EVALUATE, probabilities=None,
            is_ensemble=True, model_provenance=provenances,
            latency_seconds=time.perf_counter() - started,
            warnings=[f"Study unavailable: {exc}"], review_required=True,
        )
    except (CheckpointNotFoundError, MalformedCheckpointError, TargetMismatchError, ArchitectureMismatchError) as exc:
        raise InferenceEngineError(f"Ensemble checkpoint configuration problem: {exc}") from exc
    except RSNAInferenceError as exc:
        raise InferenceEngineError(f"Unexpected ensemble inference failure: {exc}") from exc

    _validate_probabilities(probabilities)
    return StudyInferenceResult(
        study_uid=str(study_uid), status=STATUS_OK, probabilities=probabilities,
        is_ensemble=True, model_provenance=provenances,
        latency_seconds=time.perf_counter() - started, warnings=[], review_required=False,
    )


def _validate_probabilities(probabilities: Dict[str, float]) -> None:
    if list(probabilities.keys()) != list(TARGET_COLUMNS):
        raise InferenceEngineError(
            f"Probability keys/order {list(probabilities.keys())!r} do not match "
            f"TARGET_COLUMNS {list(TARGET_COLUMNS)!r}"
        )
    for target, value in probabilities.items():
        if not (isinstance(value, float) and value == value and value not in (float("inf"), float("-inf"))):
            raise InferenceEngineError(f"probabilities[{target!r}] is not a finite float: {value!r}")
        if not (0.0 <= value <= 1.0):
            raise InferenceEngineError(f"probabilities[{target!r}] out of [0.0, 1.0]: {value!r}")


@dataclass(frozen=True)
class EnsembleCompatibilityReport:
    """Static, study-independent check that five fold checkpoints are mutually compatible --
    does not run any inference or require a local DICOM study."""

    fold_checkpoint_paths: tuple
    target_columns_match: bool
    configs_match: bool
    mismatches: tuple
    provenances: tuple

    @property
    def compatible(self) -> bool:
        return self.target_columns_match and self.configs_match and not self.mismatches

    def to_dict(self) -> dict:
        return {
            "fold_checkpoint_paths": list(self.fold_checkpoint_paths),
            "target_columns_match": self.target_columns_match,
            "configs_match": self.configs_match,
            "mismatches": list(self.mismatches),
            "compatible": self.compatible,
            "provenances": [p.to_dict() for p in self.provenances],
        }


def verify_ensemble_compatibility(fold_checkpoint_paths: Sequence[Path | str]) -> EnsembleCompatibilityReport:
    """Check architecture/target-order/preprocessing-config compatibility across all fold
    checkpoints, without touching any study data. Use before running any ensemble inference
    to get a clear report rather than a mid-inference exception."""
    paths = [Path(p) for p in fold_checkpoint_paths]
    mismatches = []
    provenances = []
    target_sets, config_sets = [], []

    for index, path in enumerate(paths, start=1):
        if not path.is_file():
            mismatches.append(f"Fold {index}: checkpoint file not found at {path}")
            continue
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            mismatches.append(f"Fold {index}: failed to load: {exc}")
            continue
        provenances.append(_load_provenance(path))
        target_sets.append(tuple(data.get("target_columns", ())))

        config = data.get("config", {}) or {}
        normalized = {
            "num_planes": config.get("num_planes") or len(config.get("planes", [])) or None,
            "dropout": config.get("dropout"),
            "image_size": config.get("image_size"),
        }
        config_sets.append(normalized)

    target_columns_match = len(set(target_sets)) <= 1 and bool(target_sets)
    if not target_columns_match and target_sets:
        mismatches.append(f"Target columns differ across folds: {set(target_sets)}")

    configs_match = True
    if config_sets:
        reference = config_sets[0]
        for index, config in enumerate(config_sets[1:], start=2):
            for key in ("num_planes", "dropout"):
                if config.get(key) != reference.get(key):
                    configs_match = False
                    mismatches.append(f"Fold {index}: config[{key}]={config.get(key)!r} != fold 1's {reference.get(key)!r}")

    return EnsembleCompatibilityReport(
        fold_checkpoint_paths=tuple(str(p) for p in paths),
        target_columns_match=target_columns_match,
        configs_match=configs_match,
        mismatches=tuple(mismatches),
        provenances=tuple(provenances),
    )
