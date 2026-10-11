"""Deterministic Alexa+ tool layer for MRI evidence analysis.

Three core tools exposed to voice or UI:
- analyze_mri_evidence: Analyze a demonstration case
- explain_review_status: Explain findings in natural language
- request_human_approval: Request approval from a named human reviewer

No network calls, no LLM, no diagnosis. Deterministic only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ReviewStatus(str, Enum):
    """Evidence review status."""

    CONSISTENT = "CONSISTENT"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    UNCERTAIN = "UNCERTAIN"


class ApprovalState(str, Enum):
    """Human approval state."""

    PENDING = "PENDING_HUMAN_APPROVAL"
    APPROVED = "APPROVED_BY_HUMAN"


@dataclass(frozen=True)
class MRIEvidence:
    """Synthetic MRI evidence record."""

    case_id: str
    baseline_date: str
    current_date: str
    quality_score: float
    foreground_percentage: float
    mean_intensity: float
    status: ReviewStatus
    change_description: str


@dataclass(frozen=True)
class ReviewExplanation:
    """Natural language explanation of review status."""

    status: ReviewStatus
    summary: str
    quality_notes: list[str]
    change_notes: list[str]
    requires_human_approval: bool


@dataclass(frozen=True)
class ApprovalRequest:
    """Request for human approval."""

    case_id: str
    approval_state: ApprovalState
    reviewer_name: Optional[str] = None
    comment: Optional[str] = None


SYNTHETIC_CASES = {
    "case-001-consistent": MRIEvidence(
        case_id="case-001-consistent",
        baseline_date="2025-06-15",
        current_date="2025-10-15",
        quality_score=0.95,
        foreground_percentage=45.2,
        mean_intensity=128.5,
        status=ReviewStatus.CONSISTENT,
        change_description="Baseline matches current image in measured evidence fields.",
    ),
    "case-002-change": MRIEvidence(
        case_id="case-002-change",
        baseline_date="2025-06-15",
        current_date="2025-10-15",
        quality_score=0.92,
        foreground_percentage=62.1,
        mean_intensity=145.3,
        status=ReviewStatus.NEEDS_REVIEW,
        change_description="Foreground changed by 16.9%, mean intensity by 16.8. Review needed.",
    ),
    "case-003-uncertain": MRIEvidence(
        case_id="case-003-uncertain",
        baseline_date="2025-06-15",
        current_date="2025-10-15",
        quality_score=0.68,
        foreground_percentage=8.5,
        mean_intensity=85.2,
        status=ReviewStatus.UNCERTAIN,
        change_description="Image quality is low; insufficient information for confident assessment.",
    ),
}

NON_DIAGNOSTIC_NOTICE = (
    "Non-diagnostic evidence summary. Not a clinical diagnosis or treatment "
    "recommendation. A qualified human reviewer must evaluate the source images."
)


def analyze_mri_evidence(case_id: str) -> MRIEvidence:
    """Analyze a demonstration MRI case.

    Args:
        case_id: One of case-001-consistent, case-002-change, case-003-uncertain

    Returns:
        Synthetic MRI evidence record

    Raises:
        ValueError: If case_id is not in the demonstration set
    """
    if case_id not in SYNTHETIC_CASES:
        raise ValueError(
            f"Case {case_id} not found. Available: {', '.join(SYNTHETIC_CASES.keys())}"
        )
    return SYNTHETIC_CASES[case_id]


def explain_review_status(evidence: MRIEvidence) -> ReviewExplanation:
    """Explain the review status in natural language.

    Args:
        evidence: MRI evidence from analyze_mri_evidence

    Returns:
        Natural language explanation of findings and status
    """
    quality_notes = []
    if evidence.quality_score > 0.90:
        quality_notes.append("Image quality is good.")
    elif evidence.quality_score > 0.70:
        quality_notes.append("Image quality is acceptable but has some noise.")
    else:
        quality_notes.append("Image quality is low and limits confidence.")

    if evidence.foreground_percentage < 1.0:
        quality_notes.append("Foreground is nearly empty.")
    elif evidence.foreground_percentage > 90:
        quality_notes.append("Foreground fills most of the image.")

    change_notes = []
    if evidence.status == ReviewStatus.CONSISTENT:
        summary = "Current image is consistent with baseline."
        change_notes.append("Measured evidence fields show no significant changes.")
    elif evidence.status == ReviewStatus.NEEDS_REVIEW:
        summary = "Current image shows notable differences from baseline."
        change_notes.append(evidence.change_description)
    else:
        summary = "Evidence assessment is uncertain due to image quality."
        change_notes.append(evidence.change_description)

    return ReviewExplanation(
        status=evidence.status,
        summary=summary,
        quality_notes=quality_notes,
        change_notes=change_notes,
        requires_human_approval=True,
    )


def request_human_approval(
    case_id: str, reviewer_name: str, comment: str = ""
) -> ApprovalRequest:
    """Request approval from a named human reviewer.

    The copilot cannot approve cases. Only a named human reviewer can
    transition approval state from PENDING to APPROVED.

    Args:
        case_id: Case identifier
        reviewer_name: Name of the human reviewer (required, non-empty)
        comment: Optional reviewer comment

    Returns:
        Approval request record (always in PENDING state until human acts)

    Raises:
        ValueError: If reviewer_name is empty or whitespace-only
    """
    if not reviewer_name or not reviewer_name.strip():
        raise ValueError("A named human reviewer is required.")

    return ApprovalRequest(
        case_id=case_id,
        approval_state=ApprovalState.PENDING,
        reviewer_name=reviewer_name.strip(),
        comment=comment or None,
    )
