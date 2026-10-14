"""Consensus layer: combines specialist summaries and critic findings into one
research-workflow status, through documented, deterministic rules -- never by
averaging agent opinions into a fictional medical confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from mri_core.rsna_inference_engine import STATUS_ERROR, STATUS_OK, STATUS_UNABLE_TO_EVALUATE

from .contracts import EvidenceBundle, QualityGateStatus, ReviewState, Severity, WorkflowStatus
from .critic import INTERNAL_INCONSISTENCY_CODES, CriticFinding
from .specialist import SpecialistSummary


@dataclass(frozen=True)
class ConsensusResult:
    workflow_status: WorkflowStatus
    specialist_summaries: Tuple[SpecialistSummary, ...]
    critic_findings: Tuple[CriticFinding, ...]
    reasons: Tuple[str, ...]
    required_next_steps: Tuple[str, ...]
    human_review_status: ReviewState

    def to_dict(self) -> dict:
        return {
            "workflow_status": self.workflow_status.value,
            "specialist_summaries": [s.to_dict() for s in self.specialist_summaries],
            "critic_findings": [f.to_dict() for f in self.critic_findings],
            "reasons": list(self.reasons),
            "required_next_steps": list(self.required_next_steps),
            "human_review_status": self.human_review_status.value,
        }


def build_consensus(
    bundle: EvidenceBundle,
    specialist_summaries: Tuple[SpecialistSummary, ...],
    critic_findings: Tuple[CriticFinding, ...],
) -> ConsensusResult:
    """Required precedence (first matching rule wins; this ordering is exhaustive and
    each condition below is mutually exclusive of the ones above it):

    1. Status OK with missing predictions -> INCONSISTENT_EVIDENCE.
       (critic code STATUS_PROBABILITY_MISMATCH, ERROR severity)
    2. Non-OK status with predictions present -> INCONSISTENT_EVIDENCE.
       (critic code STATUS_PROBABILITY_MISMATCH, WARNING severity -- still routed here
       because it is an internal-inconsistency *code*, regardless of severity)
    3. Invalid values, label order, or missing provenance -> INCONSISTENT_EVIDENCE.
       (critic codes NON_FINITE_PROBABILITY, PROBABILITY_OUT_OF_RANGE,
       LABEL_ORDER_MISMATCH, MISSING_PROVENANCE, MISSING_PROVENANCE_HASH,
       UNEXPECTED_ENSEMBLE_SIZE)
       Rules 1-3 are all implemented as one check: "any critic finding whose code is
       in INTERNAL_INCONSISTENCY_CODES" -- checked first, because a self-contradictory
       engine result is a different, more concerning problem than plain unavailability.
    4. Explicit unavailable/error status with no predictions -> INFERENCE_UNAVAILABLE.
       (inference_status is STATUS_UNABLE_TO_EVALUATE/STATUS_ERROR, or probabilities is
       None, and none of rules 1-3 already matched)
    5. Valid inference (status OK, predictions present, structurally consistent) with
       quality_gate_status NOT_ASSESSED or REVIEW -> MORE_EVIDENCE_REQUIRED. The same
       branch also covers quality_gate_status INVALID and any remaining non-fatal
       critic warning -- an invalid-quality image needs a different/better image, which
       is "more evidence," not an internal software inconsistency.
    6. Valid inference with complete supporting evidence and quality_gate_status OK ->
       EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW.
    """
    errors = tuple(f for f in critic_findings if f.severity == Severity.ERROR)
    warnings_ = tuple(f for f in critic_findings if f.severity == Severity.WARNING)
    inconsistency_findings = tuple(f for f in critic_findings if f.code in INTERNAL_INCONSISTENCY_CODES)

    if inconsistency_findings:
        status = WorkflowStatus.INCONSISTENT_EVIDENCE
        reasons = tuple(f.message for f in inconsistency_findings)
        next_steps = ("Investigate the inference engine's output before any further research use.",)

    elif bundle.inference_status in (STATUS_UNABLE_TO_EVALUATE, STATUS_ERROR) or bundle.probabilities is None:
        status = WorkflowStatus.INFERENCE_UNAVAILABLE
        reasons = (f"inference_status={bundle.inference_status!r}; no usable probabilities for this study.",)
        next_steps = ("Obtain a complete, compatible local study and re-run inference.",)

    elif errors or warnings_ or bundle.quality_gate_status != QualityGateStatus.OK:
        status = WorkflowStatus.MORE_EVIDENCE_REQUIRED
        remaining = errors + warnings_
        reasons = tuple(f.message for f in remaining) if remaining else (
            f"quality_gate_status={bundle.quality_gate_status.value} (expected OK).",
        )
        next_steps = (
            "Run (or re-run) the deterministic quality assessment and resolve any flagged "
            "warnings before treating this study's evidence as complete.",
        )

    else:
        status = WorkflowStatus.EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW
        reasons = ("All critic checks passed and quality_gate_status is OK.",)
        next_steps = ("Proceed to the simulated human-review workflow.",)

    return ConsensusResult(
        workflow_status=status,
        specialist_summaries=specialist_summaries,
        critic_findings=critic_findings,
        reasons=reasons,
        required_next_steps=next_steps,
        human_review_status=ReviewState.PENDING_REVIEW,
    )
