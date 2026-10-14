"""Top-level entry point tying specialist, critic, consensus, and the review gate
together around one mri_core.rsna_inference_engine.StudyInferenceResult.

This is the one function other code (OpenCV/Nebius/AMD branches, CLI demo, tests)
should call -- it does not duplicate any inference, DICOM, or quality-assessment logic;
it only orchestrates the existing engine's result through this package's layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mri_core.rsna_inference_engine import StudyInferenceResult

from .consensus import ConsensusResult, build_consensus
from .contracts import EvidenceBundle, QualityGateStatus, build_evidence_bundle
from .critic import CriticFinding, review as critic_review
from .review_gate import ReviewGate
from .specialist import SpecialistSummary, summarize_by_group


@dataclass
class WorkflowResult:
    """Everything produced for one study: the immutable evidence/consensus trail, plus
    a fresh, mutable ReviewGate scoped to that evidence's fingerprint."""

    bundle: EvidenceBundle
    specialist_summaries: tuple[SpecialistSummary, ...]
    critic_findings: tuple[CriticFinding, ...]
    consensus: ConsensusResult
    review_gate: ReviewGate

    def to_dict(self) -> dict:
        return {
            "bundle": self.bundle.to_dict(),
            "specialist_summaries": [s.to_dict() for s in self.specialist_summaries],
            "critic_findings": [f.to_dict() for f in self.critic_findings],
            "consensus": self.consensus.to_dict(),
            "review_gate": {
                "state": self.review_gate.state.value,
                "evidence_fingerprint": self.review_gate.evidence_fingerprint,
                "history": [e.to_dict() for e in self.review_gate.history],
            },
        }


def process_inference_result(
    result: StudyInferenceResult,
    *,
    quality_gate_status: QualityGateStatus = QualityGateStatus.NOT_ASSESSED,
    evidence_references: Sequence[str] = (),
) -> WorkflowResult:
    """Build an EvidenceBundle from result, run it through specialist/critic/consensus,
    and attach a fresh ReviewGate. Deterministic and side-effect-free except for the
    ReviewGate's own internal timestamp-bearing state, which starts at PENDING_REVIEW
    with an empty history.
    """
    bundle = build_evidence_bundle(
        result, quality_gate_status=quality_gate_status, evidence_references=evidence_references
    )
    specialists = summarize_by_group(bundle)
    findings = critic_review(bundle)
    consensus = build_consensus(bundle, specialists, findings)
    gate = ReviewGate(bundle.fingerprint)
    return WorkflowResult(
        bundle=bundle,
        specialist_summaries=specialists,
        critic_findings=findings,
        consensus=consensus,
        review_gate=gate,
    )
