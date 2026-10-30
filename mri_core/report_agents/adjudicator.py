"""Agent 3: Adjudicator — reconcile extraction and verification, escalate uncertainty."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .contracts import (
    ExtractionResult,
    VerificationResult,
    AdjudicationResult,
    AdjudicatedFinding,
    AdjudicationConsensus,
    LabelState,
    VerificationStatus,
)


def adjudicate(
    report_text: str,
    extraction: ExtractionResult,
    verification: VerificationResult,
) -> AdjudicationResult:
    """
    Reconcile extraction and verification outputs.

    Rules (in order):
    1. UNANIMOUS agreement → use agreed state
    2. Extraction POSITIVE + Verification SUPPORTED → POSITIVE
    3. Extraction POSITIVE + Verification UNSUPPORTED → escalate (UNCERTAIN + review)
    4. Extraction NEGATIVE + Verification SUPPORTED → NEGATIVE
    5. Extraction NEGATIVE + Verification UNSUPPORTED → escalate (UNCERTAIN + review)
    6. Extraction UNCERTAIN + any verification → UNCERTAIN + review
    7. Extraction NOT_MENTIONED + Verification SUPPORTED → use SUPPORTED state (contradiction)
    8. Extraction NOT_MENTIONED → NOT_MENTIONED (unless contradicted)

    Never silently converts UNCERTAIN/NOT_MENTIONED to NEGATIVE.
    Preserves uncertainty and missing findings.
    """

    if extraction.study_uid != verification.study_uid:
        raise ValueError(f"Study UID mismatch: {extraction.study_uid} vs {verification.study_uid}")

    adjudicated_findings = []
    escalated_targets = []

    # Build a map of verification results by target
    verification_map = {f.target: f for f in verification.findings}

    for ext_finding in extraction.findings:
        ver_finding = verification_map.get(ext_finding.target)
        if not ver_finding:
            raise ValueError(f"Missing verification for target {ext_finding.target}")

        adj_finding = _reconcile_finding(
            target=ext_finding.target,
            extractor_state=ext_finding.state,
            extractor_span=ext_finding.evidence_span,
            verifier_status=ver_finding.verification_status,
            verifier_text=ver_finding.supporting_text,
        )
        adjudicated_findings.append(adj_finding)

        if adj_finding.human_review_required:
            escalated_targets.append(adj_finding.target)

    escalation_reasons = _get_escalation_reasons(
        extraction.findings,
        [f.verification_status for f in verification.findings],
        [f.human_review_required for f in adjudicated_findings],
    )

    return AdjudicationResult(
        study_uid=extraction.study_uid,
        adjudication_timestamp=datetime.utcnow().isoformat() + "Z",
        schema_version="adjudicator-0.1",
        final_labels=tuple(adjudicated_findings),
        escalated_count=len(escalated_targets),
        escalation_reasons=tuple(escalation_reasons),
    )


def _reconcile_finding(
    target: str,
    extractor_state: LabelState,
    extractor_span: Optional[str],
    verifier_status: VerificationStatus,
    verifier_text: Optional[str],
) -> AdjudicatedFinding:
    """Reconcile one target's extraction and verification results."""

    # NOT_MENTIONED: vacuously true unless contradicted by verification
    if extractor_state == LabelState.NOT_MENTIONED:
        return AdjudicatedFinding(
            target=target,
            final_state=LabelState.NOT_MENTIONED,
            consensus=AdjudicationConsensus.UNANIMOUS,
            extractor_state=extractor_state,
            verifier_status=verifier_status,
            evidence_span=None,
            confidence=1.0,
            human_review_required=False,
        )

    # UNCERTAIN: always escalate with review (preserve uncertainty)
    if extractor_state == LabelState.UNCERTAIN:
        return AdjudicatedFinding(
            target=target,
            final_state=LabelState.UNCERTAIN,
            consensus=AdjudicationConsensus.RESOLVED_VIA_ESCALATION,
            extractor_state=extractor_state,
            verifier_status=verifier_status,
            evidence_span=extractor_span,
            confidence=0.0,
            human_review_required=True,
            review_reason="extraction_uncertain",
            resolution_rule="uncertain_always_escalates",
        )

    # POSITIVE or NEGATIVE: reconcile with verification
    if verifier_status == VerificationStatus.SUPPORTED:
        # Extraction + Verification agree
        return AdjudicatedFinding(
            target=target,
            final_state=extractor_state,
            consensus=AdjudicationConsensus.UNANIMOUS,
            extractor_state=extractor_state,
            verifier_status=verifier_status,
            evidence_span=extractor_span or verifier_text,
            confidence=0.95,
            human_review_required=False,
        )

    elif verifier_status == VerificationStatus.UNSUPPORTED:
        # Extraction claims finding but verification can't support it
        # Preserve UNCERTAIN instead of silently negating
        return AdjudicatedFinding(
            target=target,
            final_state=LabelState.UNCERTAIN,
            consensus=AdjudicationConsensus.RESOLVED_VIA_ESCALATION,
            extractor_state=extractor_state,
            verifier_status=verifier_status,
            evidence_span=None,  # Verification didn't find support
            confidence=0.0,
            human_review_required=True,
            review_reason="extraction_unverified",
            resolution_rule="verification_overrides_unsupported_extraction",
        )

    else:  # CONTRADICTED
        # Extraction claims one state, but report clearly contradicts it
        return AdjudicatedFinding(
            target=target,
            final_state=LabelState.UNCERTAIN,
            consensus=AdjudicationConsensus.RESOLVED_VIA_ESCALATION,
            extractor_state=extractor_state,
            verifier_status=verifier_status,
            evidence_span=verifier_text,
            confidence=0.0,
            human_review_required=True,
            review_reason="extraction_contradicted_by_report",
            resolution_rule="contradiction_always_escalates",
        )


def _get_escalation_reasons(
    extractions: tuple,
    verifications: list[VerificationStatus],
    escalation_flags: list[bool],
) -> list[str]:
    """Summarize reasons for escalation."""
    reasons = set()

    for i, (ext_finding, ver_status, escalated) in enumerate(
        zip(extractions, verifications, escalation_flags)
    ):
        if not escalated:
            continue

        if ext_finding.state == LabelState.UNCERTAIN:
            reasons.add("extraction_uncertain")
        elif ver_status == VerificationStatus.UNSUPPORTED:
            reasons.add("extraction_unverified")
        elif ver_status == VerificationStatus.CONTRADICTED:
            reasons.add("extraction_contradicted")

    return sorted(reasons)
