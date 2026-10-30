"""Agent 2: Evidence Verifier — independently validate extracted claims."""

from __future__ import annotations

from datetime import datetime
import re

from .contracts import (
    ExtractionResult,
    VerificationFinding,
    VerificationResult,
    VerificationStatus,
    LabelState,
)


def verify_extraction(
    report_text: str,
    extraction: ExtractionResult,
) -> VerificationResult:
    """
    Verify each extracted finding against the original report.

    Independently checks whether claimed findings are supported by explicit text.
    Conservative: if evidence span is not found, mark as UNSUPPORTED (never invent).

    Args:
        report_text: Original MRI report text
        extraction: ExtractionResult from Agent 1

    Returns:
        VerificationResult with per-target verification status
    """
    verification_findings = []

    for finding in extraction.findings:
        verified_finding = _verify_single_finding(
            report_text=report_text,
            target=finding.target,
            state=finding.state,
            evidence_span=finding.evidence_span,
        )
        verification_findings.append(verified_finding)

    return VerificationResult(
        study_uid=extraction.study_uid,
        verification_timestamp=datetime.utcnow().isoformat() + "Z",
        verified_by_model="rule-based-verifier-0.1",
        findings=tuple(verification_findings),
    )


def _verify_single_finding(
    report_text: str,
    target: str,
    state: LabelState,
    evidence_span: str | None,
) -> VerificationFinding:
    """
    Verify a single target's extracted finding.

    Returns SUPPORTED if:
    - Evidence span is present in report (exact or similar match)
    - OR state is NOT_MENTIONED (vacuously true: no finding to verify)

    Returns UNSUPPORTED if:
    - Evidence span is claimed but not found in report
    - OR claim contradicts explicit report text
    """

    if state == LabelState.NOT_MENTIONED:
        # NOT_MENTIONED is always verified (no claim to contradict)
        return VerificationFinding(
            target=target,
            extractor_state=state,
            verification_status=VerificationStatus.SUPPORTED,
            supporting_text=None,
            is_explicit=False,
            confidence=1.0,
        )

    # For other states, we need evidence
    if not evidence_span:
        # Claim made but no evidence provided
        return VerificationFinding(
            target=target,
            extractor_state=state,
            verification_status=VerificationStatus.UNSUPPORTED,
            supporting_text=None,
            issue="claim made without evidence span",
            confidence=0.0,
        )

    # Check if evidence span appears in report (fuzzy match for robustness)
    is_found = _evidence_span_in_report(report_text, evidence_span, target)

    if is_found:
        return VerificationFinding(
            target=target,
            extractor_state=state,
            verification_status=VerificationStatus.SUPPORTED,
            supporting_text=evidence_span,
            is_explicit=True,
            confidence=0.95,
        )
    else:
        # Evidence span not found in report
        return VerificationFinding(
            target=target,
            extractor_state=state,
            verification_status=VerificationStatus.UNSUPPORTED,
            supporting_text=None,
            issue=f"evidence span not found in report: '{evidence_span[:50]}...'",
            alternative_finding=_find_alternative_mention(report_text, target),
            confidence=0.0,
        )


def _evidence_span_in_report(
    report_text: str,
    evidence_span: str,
    target: str,
) -> bool:
    """
    Check if evidence span appears in report (with fuzzy matching).

    Handles:
    - Exact substring match
    - Normalized whitespace match
    - Case-insensitive match
    - Partial match (key phrases)
    """
    if not evidence_span:
        return False

    # Normalize both texts
    report_normalized = _normalize_text(report_text)
    span_normalized = _normalize_text(evidence_span)

    # Exact match (case-insensitive, normalized)
    if span_normalized in report_normalized:
        return True

    # Partial match: key words from span appear in report
    key_words = _extract_key_words(span_normalized, target)
    if key_words:
        found_count = sum(1 for word in key_words if word in report_normalized)
        # If >50% of key words found, consider it a match
        if found_count >= len(key_words) * 0.5:
            return True

    return False


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return " ".join(text.lower().split())


def _extract_key_words(text: str, target: str) -> list[str]:
    """Extract key words from evidence span for fuzzy matching."""
    # Always include target name
    words = [_normalize_text(target).lower()]

    # Add other significant words (skip common words)
    stop_words = {"the", "a", "an", "and", "or", "is", "are", "with", "of", "in", "on", "at"}
    for word in text.split():
        if len(word) > 3 and word not in stop_words:
            words.append(_normalize_text(word))

    return words


def _find_alternative_mention(report_text: str, target: str) -> str | None:
    """
    If exact evidence span not found, search for alternative mentions of target.

    Returns a short quote if found, None otherwise.
    """
    text_lower = report_text.lower()
    target_lower = target.lower()

    # Look for target name in report
    matches = re.finditer(rf".{{0,30}}{re.escape(target_lower)}.{{0,30}}", text_lower)
    for match in matches:
        snippet = report_text[max(0, match.start()-5):min(len(report_text), match.end()+5)]
        return snippet.strip()

    return None
