"""Data contracts for multi-agent report labeling (frozen, versioned, fingerprinted)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class LabelState(str, Enum):
    """Exactly one of four mutually exclusive states per target."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"
    NOT_MENTIONED = "not_mentioned"


class VerificationStatus(str, Enum):
    """Verification result status."""
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class AdjudicationConsensus(str, Enum):
    """Reconciliation outcome."""
    UNANIMOUS = "unanimous"
    RESOLVED_VIA_EXTRACTION = "resolved_via_extraction"
    RESOLVED_VIA_VERIFICATION = "resolved_via_verification"
    RESOLVED_VIA_ESCALATION = "resolved_via_escalation"


@dataclass(frozen=True)
class Finding:
    """Single target finding with evidence span and metadata."""
    target: str
    state: LabelState
    confidence: float
    evidence_span: Optional[str] = None
    evidence_section: Optional[str] = None
    location_specificity: Optional[str] = None
    hedging_flags: tuple = ()
    reason: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class ExtractionResult:
    """Output from Agent 1 (Report Extractor)."""
    study_uid: str
    timestamp: str
    model_id: str
    prompt_version: str
    language_detected: str
    findings: tuple  # tuple of Finding objects
    processing_notes: tuple = ()

    def to_dict(self) -> dict:
        return {
            "study_uid": self.study_uid,
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "language_detected": self.language_detected,
            "findings": [
                {
                    "target": f.target,
                    "state": f.state.value,
                    "confidence": f.confidence,
                    "evidence_span": f.evidence_span,
                    "evidence_section": f.evidence_section,
                    "location_specificity": f.location_specificity,
                    "hedging_flags": list(f.hedging_flags),
                    "reason": f.reason,
                }
                for f in self.findings
            ],
            "processing_notes": list(self.processing_notes),
        }


@dataclass(frozen=True)
class VerificationFinding:
    """Per-target verification result."""
    target: str
    extractor_state: LabelState
    verification_status: VerificationStatus
    supporting_text: Optional[str] = None
    is_explicit: bool = False
    negation_scope: Optional[str] = None
    historical_context: bool = False
    severity_qualifier: Optional[str] = None
    issue: Optional[str] = None
    alternative_finding: Optional[str] = None
    confidence: float = 1.0


@dataclass(frozen=True)
class VerificationResult:
    """Output from Agent 2 (Evidence Verifier)."""
    study_uid: str
    verification_timestamp: str
    verified_by_model: str
    findings: tuple  # tuple of VerificationFinding objects

    def to_dict(self) -> dict:
        supported = sum(1 for f in self.findings if f.verification_status == VerificationStatus.SUPPORTED)
        unsupported = sum(1 for f in self.findings if f.verification_status == VerificationStatus.UNSUPPORTED)
        contradicted = sum(1 for f in self.findings if f.verification_status == VerificationStatus.CONTRADICTED)

        return {
            "study_uid": self.study_uid,
            "verification_timestamp": self.verification_timestamp,
            "verified_by_model": self.verified_by_model,
            "findings": [
                {
                    "target": f.target,
                    "extractor_state": f.extractor_state.value,
                    "verification_status": f.verification_status.value,
                    "supporting_text": f.supporting_text,
                    "is_explicit": f.is_explicit,
                    "negation_scope": f.negation_scope,
                    "historical_context": f.historical_context,
                    "severity_qualifier": f.severity_qualifier,
                    "issue": f.issue,
                    "alternative_finding": f.alternative_finding,
                    "confidence": f.confidence,
                }
                for f in self.findings
            ],
            "summary": {
                "supported_count": supported,
                "unsupported_count": unsupported,
                "contradicted_count": contradicted,
            },
        }


@dataclass(frozen=True)
class AdjudicatedFinding:
    """Final reconciled label for one target."""
    target: str
    final_state: LabelState
    consensus: AdjudicationConsensus
    extractor_state: LabelState
    verifier_status: VerificationStatus
    evidence_span: Optional[str]
    confidence: float
    human_review_required: bool
    review_reason: Optional[str] = None
    resolution_rule: Optional[str] = None

    def __post_init__(self):
        if self.final_state == LabelState.NOT_MENTIONED and self.evidence_span is not None:
            raise ValueError("NOT_MENTIONED cannot have evidence_span")


@dataclass(frozen=True)
class AdjudicationResult:
    """Output from Agent 3 (Adjudicator)."""
    study_uid: str
    adjudication_timestamp: str
    schema_version: str
    final_labels: tuple  # tuple of AdjudicatedFinding objects
    escalated_count: int
    escalation_reasons: tuple

    def to_dict(self) -> dict:
        return {
            "study_uid": self.study_uid,
            "adjudication_timestamp": self.adjudication_timestamp,
            "schema_version": self.schema_version,
            "final_labels": [
                {
                    "target": f.target,
                    "final_state": f.final_state.value,
                    "consensus": f.consensus.value,
                    "extractor_state": f.extractor_state.value,
                    "verifier_status": f.verifier_status.value,
                    "evidence_span": f.evidence_span,
                    "confidence": f.confidence,
                    "human_review_required": f.human_review_required,
                    "review_reason": f.review_reason,
                    "resolution_rule": f.resolution_rule,
                }
                for f in self.final_labels
            ],
            "escalated_count": self.escalated_count,
            "escalation_reasons": list(self.escalation_reasons),
        }


def extract_from_report(report_text: str, study_uid: str = None) -> ExtractionResult:
    """Extract findings from report. Implemented in extractor.py."""
    from .extractor import extract_from_report_rule_based
    return extract_from_report_rule_based(report_text, study_uid)


def verify_extraction(report_text: str, extraction: ExtractionResult) -> VerificationResult:
    """Verify extracted findings. Implemented in verifier.py."""
    from .verifier import verify_extraction as verify_impl
    return verify_impl(report_text, extraction)


def adjudicate(
    report_text: str,
    extraction: ExtractionResult,
    verification: VerificationResult,
) -> AdjudicationResult:
    """Reconcile extraction and verification. Implemented in adjudicator.py."""
    from .adjudicator import adjudicate as adjudicate_impl
    return adjudicate_impl(report_text, extraction, verification)
