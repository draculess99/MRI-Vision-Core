"""Shared data contracts for the specialist-critic-consensus evidence-review layer.

Every downstream module (specialist, critic, consensus, review_gate, llm_adapter,
orchestrator) reads an EvidenceBundle and never mutates it -- it is a deeply immutable
snapshot built once from an existing mri_core.rsna_inference_engine.StudyInferenceResult.
This module never re-implements inference, checkpoint loading, or DICOM handling; it
only restates what the inference engine already produced in a form this package can
reason about alongside a deterministic quality-gate status.

Immutability note: a frozen dataclass only prevents reassigning its *attributes* -- a
plain dict or list stored in one of those attributes remains fully mutable in place
(`bundle.probabilities["ACL"] = 0.99` would silently succeed on a naive frozen
dataclass). Every nested structure here is therefore deep-frozen with
types.MappingProxyType (for mappings) and tuples (for sequences) via _deep_freeze(),
so any attempted mutation anywhere in the structure raises TypeError instead of
silently succeeding and leaving the bundle's own fingerprint stale.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from mri_core.rsna_inference_engine import StudyInferenceResult
from mri_core.rsna_knee_dataset import TARGET_COLUMNS


class QualityGateStatus(str, Enum):
    """Deterministic image/pipeline quality state, distinct from inference_status.

    NOT_ASSESSED is a first-class value, not a stand-in for "passed" -- a caller that
    never ran mri_core.decision.generate_decision_report for this study must report
    NOT_ASSESSED, never OK.
    """

    NOT_ASSESSED = "NOT_ASSESSED"
    OK = "OK"
    REVIEW = "REVIEW"
    INVALID = "INVALID"


class WorkflowStatus(str, Enum):
    """Research-workflow classification produced by the consensus layer.

    These are workflow statuses about evidence completeness, not clinical findings and
    not a diagnosis of any kind.
    """

    EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW = "EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW"
    MORE_EVIDENCE_REQUIRED = "MORE_EVIDENCE_REQUIRED"
    INFERENCE_UNAVAILABLE = "INFERENCE_UNAVAILABLE"
    INCONSISTENT_EVIDENCE = "INCONSISTENT_EVIDENCE"


class ReviewState(str, Enum):
    """Simulated human-review workflow state. No actual clinician approval is implied
    by any value here."""

    PENDING_REVIEW = "PENDING_REVIEW"
    MORE_EVIDENCE_REQUESTED = "MORE_EVIDENCE_REQUESTED"
    REVIEWED = "REVIEWED"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


DEFAULT_LIMITATIONS: Tuple[str, ...] = (
    "Research prototype; not a diagnostic device and not clinically validated.",
    "Trained on at most 58 fully labeled RSNA knee studies (46-48 per cross-validation fold).",
    "Probabilities are uncalibrated sigmoid outputs, not confidence scores.",
)

# ----------------------------------------------------------------------- deep freezing


def _deep_freeze(value: Any) -> Any:
    """Recursively convert dicts to read-only MappingProxyType and lists to tuples.

    Scalars (str, int, float, bool, None) and enums pass through unchanged. Applied to
    every nested structure stored on an EvidenceBundle so that mutation anywhere in the
    tree raises TypeError rather than silently succeeding.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _deep_jsonable(value: Any) -> Any:
    """Inverse of _deep_freeze for JSON/fingerprint purposes, with explicit handling
    of non-finite floats -- this never relies on json.dumps's own (non-standard)
    implicit NaN/Infinity encoding. NaN/Infinity/-Infinity become explicit string
    sentinels so the resulting structure is strict-JSON-safe and the sentinel is
    visible in any serialized output rather than silently passed through.
    """
    if isinstance(value, MappingProxyType):
        return {k: _deep_jsonable(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _deep_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_deep_jsonable(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Enum):
        return value.value
    return value


# ------------------------------------------------------------------------- the bundle


@dataclass(frozen=True)
class EvidenceBundle:
    """The single input contract this entire package consumes.

    Built only via build_evidence_bundle() from a real StudyInferenceResult -- there is
    no supported way to hand-assemble one with invented probabilities, findings, or
    provenance. Every nested field (probabilities, checkpoint_provenance) is deep-frozen
    via _deep_freeze(): probabilities is a types.MappingProxyType (read-only mapping,
    insertion-order preserved so TARGET_COLUMNS order survives), and
    checkpoint_provenance is a tuple of MappingProxyType snapshots, each with its own
    nested fields (e.g. target_columns) frozen too. Attempting to mutate any of these
    raises TypeError immediately -- there is no path by which the bundle's fingerprint
    can go stale through mutation, because mutation itself is rejected.
    """

    study_uid: str
    inference_status: str  # mri_core.rsna_inference_engine.STATUS_OK / _UNABLE_TO_EVALUATE / _ERROR
    probabilities: Optional[Mapping[str, float]]  # MappingProxyType; exactly the 12 TARGET_COLUMNS, in order, or None
    is_ensemble: bool
    model_count: int
    checkpoint_provenance: Tuple[Mapping[str, Any], ...]  # tuple of MappingProxyType snapshots, order-preserved
    latency_seconds: float
    quality_gate_status: QualityGateStatus
    warnings: Tuple[str, ...]
    evidence_references: Tuple[str, ...]
    limitations: Tuple[str, ...]
    disclaimer: str
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingerprint", self._compute_fingerprint())

    def _compute_fingerprint(self) -> str:
        """Canonical, validated evidence only: every field is passed through
        _deep_jsonable() (so nested MappingProxyType/tuple/non-finite-float values are
        resolved to a strict-JSON-safe form) before hashing, with allow_nan=False as a
        defensive check that nothing non-finite could slip through unconverted.
        """
        payload = {
            "study_uid": self.study_uid,
            "inference_status": self.inference_status,
            "probabilities": _deep_jsonable(self.probabilities) if self.probabilities is not None else None,
            "is_ensemble": self.is_ensemble,
            "checkpoint_provenance": _deep_jsonable(self.checkpoint_provenance),
            "quality_gate_status": self.quality_gate_status.value,
        }
        canonical = json.dumps(payload, sort_keys=True, allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "study_uid": self.study_uid,
            "inference_status": self.inference_status,
            "probabilities": _deep_jsonable(self.probabilities) if self.probabilities is not None else None,
            "is_ensemble": self.is_ensemble,
            "model_count": self.model_count,
            "checkpoint_provenance": _deep_jsonable(self.checkpoint_provenance),
            "latency_seconds": self.latency_seconds,
            "quality_gate_status": self.quality_gate_status.value,
            "warnings": list(self.warnings),
            "evidence_references": list(self.evidence_references),
            "limitations": list(self.limitations),
            "disclaimer": self.disclaimer,
            "fingerprint": self.fingerprint,
        }


class EvidenceBundleError(ValueError):
    """Raised when a StudyInferenceResult cannot be faithfully represented as a bundle."""


def build_evidence_bundle(
    result: StudyInferenceResult,
    *,
    quality_gate_status: QualityGateStatus = QualityGateStatus.NOT_ASSESSED,
    evidence_references: Sequence[str] = (),
    extra_limitations: Sequence[str] = (),
) -> EvidenceBundle:
    """Build an EvidenceBundle from a real inference-engine result.

    quality_gate_status defaults to NOT_ASSESSED -- callers must explicitly pass the
    real mri_core.decision.QualityStatus-derived value when a deterministic quality
    assessment was actually run; omitting it is never silently treated as "OK".

    This function copies result.probabilities into a brand-new dict (never the
    caller's own dict object) before deep-freezing it, so even if the caller continues
    to hold and mutate their original StudyInferenceResult.probabilities afterward,
    the bundle is unaffected -- see test_original_study_inference_result_unaffected_
    by_later_mutation.
    """
    probabilities = result.probabilities
    if probabilities is not None and list(probabilities.keys()) != list(TARGET_COLUMNS):
        raise EvidenceBundleError(
            f"probabilities must contain exactly the 12 TARGET_COLUMNS in order; "
            f"got {list(probabilities.keys())!r}"
        )
    frozen_probabilities = _deep_freeze(dict(probabilities)) if probabilities is not None else None
    frozen_provenance = _deep_freeze(tuple(p.to_dict() for p in result.model_provenance))
    return EvidenceBundle(
        study_uid=str(result.study_uid),
        inference_status=result.status,
        probabilities=frozen_probabilities,
        is_ensemble=bool(result.is_ensemble),
        model_count=len(result.model_provenance),
        checkpoint_provenance=frozen_provenance,
        latency_seconds=float(result.latency_seconds),
        quality_gate_status=quality_gate_status,
        warnings=tuple(result.warnings),
        evidence_references=tuple(evidence_references),
        limitations=tuple(DEFAULT_LIMITATIONS) + tuple(extra_limitations),
        disclaimer=result.disclaimer,
    )


def utc_timestamp() -> float:
    """Single place to obtain a timestamp, so tests can monkeypatch it if ever needed."""
    return time.time()
