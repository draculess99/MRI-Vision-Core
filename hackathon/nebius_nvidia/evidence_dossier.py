"""Deterministic, non-diagnostic MRI Evidence Dossier adapter.

Wraps the existing mri_core pipeline output (read-only) into a typed dossier
that always requires a human decision. No network, no LLM, no diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from mri_core.pipeline import process_mri_image

LOW_CONTRAST_STD = 10.0
FOREGROUND_MIN_PCT = 1.0
FOREGROUND_MAX_PCT = 90.0
MIN_DIMENSION = 32
DELTA_REVIEW_THRESHOLDS = {
    "foreground_percentage": 5.0,
    "mean_intensity": 15.0,
    "std_deviation": 10.0,
}

NON_DIAGNOSTIC_NOTICE = (
    "Non-diagnostic evidence summary. Not a clinical diagnosis or treatment "
    "recommendation. A qualified human reviewer must evaluate the source images."
)


class EvidenceStatus(str, Enum):
    CONSISTENT = "CONSISTENT"
    REVIEW = "REVIEW"
    UNCERTAIN = "UNCERTAIN"


class ApprovalState(str, Enum):
    PENDING = "PENDING_HUMAN_APPROVAL"
    APPROVED = "APPROVED_BY_HUMAN"
    REJECTED = "REJECTED_BY_HUMAN"


@dataclass(frozen=True)
class QualityEvidence:
    width: int
    height: int
    mean_intensity: float
    std_deviation: float
    foreground_percentage: float
    flags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ChangeGuardEvidence:
    status: EvidenceStatus
    deltas: Mapping[str, float] = field(default_factory=dict)
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class HumanApproval:
    state: ApprovalState = ApprovalState.PENDING
    reviewer: Optional[str] = None
    comment: Optional[str] = None


@dataclass(frozen=True)
class EvidenceDossier:
    quality: QualityEvidence
    changeguard: Optional[ChangeGuardEvidence]
    status: EvidenceStatus
    review_notes: Tuple[str, ...]
    approval: HumanApproval = field(default_factory=HumanApproval)
    notice: str = NON_DIAGNOSTIC_NOTICE

    @property
    def requires_human_approval(self) -> bool:
        return True

    def decide(self, approve: bool, reviewer: str, comment: str = "") -> "EvidenceDossier":
        if not reviewer or not reviewer.strip():
            raise ValueError("A named human reviewer is required to decide.")
        state = ApprovalState.APPROVED if approve else ApprovalState.REJECTED
        return replace(
            self, approval=HumanApproval(state, reviewer.strip(), comment or None)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "requires_human_approval": self.requires_human_approval,
            "approval": {
                "state": self.approval.state.value,
                "reviewer": self.approval.reviewer,
                "comment": self.approval.comment,
            },
            "quality": {
                "width": self.quality.width,
                "height": self.quality.height,
                "mean_intensity": self.quality.mean_intensity,
                "std_deviation": self.quality.std_deviation,
                "foreground_percentage": self.quality.foreground_percentage,
                "flags": list(self.quality.flags),
            },
            "changeguard": None
            if self.changeguard is None
            else {
                "status": self.changeguard.status.value,
                "deltas": dict(self.changeguard.deltas),
                "reasons": list(self.changeguard.reasons),
            },
            "review_notes": list(self.review_notes),
            "notice": self.notice,
        }


def quality_from_features(features: Mapping[str, Any]) -> QualityEvidence:
    """Derive quality evidence from an mri_core ``extract_features`` dict."""
    width, height = int(features["image_width"]), int(features["image_height"])
    std = float(features["std_deviation"])
    fg = float(features["foreground_percentage"])
    flags = []
    if min(width, height) < MIN_DIMENSION:
        flags.append("small_image")
    if std < LOW_CONTRAST_STD:
        flags.append("low_contrast")
    if fg < FOREGROUND_MIN_PCT:
        flags.append("foreground_nearly_empty")
    elif fg > FOREGROUND_MAX_PCT:
        flags.append("foreground_nearly_full")
    return QualityEvidence(
        width=width,
        height=height,
        mean_intensity=float(features["mean_intensity"]),
        std_deviation=std,
        foreground_percentage=fg,
        flags=tuple(flags),
    )


def compare_features(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> ChangeGuardEvidence:
    """Deterministic baseline-vs-current comparison over mri_core features."""
    deltas = {
        key: float(current[key]) - float(baseline[key])
        for key in DELTA_REVIEW_THRESHOLDS
    }
    reasons = tuple(
        f"{key} changed by {deltas[key]:+.2f} (threshold {limit})"
        for key, limit in DELTA_REVIEW_THRESHOLDS.items()
        if abs(deltas[key]) > limit
    )
    status = EvidenceStatus.REVIEW if reasons else EvidenceStatus.CONSISTENT
    return ChangeGuardEvidence(status=status, deltas=deltas, reasons=reasons)


def build_dossier(
    features: Mapping[str, Any],
    changeguard: Optional[ChangeGuardEvidence] = None,
) -> EvidenceDossier:
    """Assemble a dossier from mri_core features and optional change evidence."""
    quality = quality_from_features(features)
    notes = [NON_DIAGNOSTIC_NOTICE]
    status = EvidenceStatus.CONSISTENT

    if quality.flags:
        status = EvidenceStatus.UNCERTAIN
        notes.append("Image quality flags raised: " + ", ".join(quality.flags) + ".")
    if changeguard is None:
        notes.append("No ChangeGuard comparison available; change evidence not assessed.")
    else:
        if changeguard.status is not EvidenceStatus.CONSISTENT:
            notes.append(
                f"ChangeGuard status {changeguard.status.value}: human review needed."
            )
            if status is EvidenceStatus.CONSISTENT:
                status = changeguard.status
            elif changeguard.status is EvidenceStatus.UNCERTAIN:
                status = EvidenceStatus.UNCERTAIN
        notes.extend(changeguard.reasons)
    notes.append("Human approval is required before this dossier is used.")

    return EvidenceDossier(
        quality=quality,
        changeguard=changeguard,
        status=status,
        review_notes=tuple(notes),
    )


def build_dossier_from_images(
    image: np.ndarray,
    baseline_image: Optional[np.ndarray] = None,
    segmentation_method: str = "otsu",
) -> EvidenceDossier:
    """Run the existing mri_core pipeline (unmodified) and build a dossier."""
    current = process_mri_image(image, segmentation_method)["features"]
    changeguard = None
    if baseline_image is not None:
        base = process_mri_image(baseline_image, segmentation_method)["features"]
        changeguard = compare_features(base, current)
    return build_dossier(current, changeguard)
