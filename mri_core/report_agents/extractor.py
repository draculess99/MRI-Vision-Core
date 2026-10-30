"""Agent 1: Report Extractor — extract findings, evidence spans, and hedging flags."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional
import re

from .contracts import (
    Finding,
    ExtractionResult,
    LabelState,
)
from mri_core.report_label_rules import RuleBasedExtractor
from mri_core.rsna_knee_dataset import TARGET_COLUMNS


def extract_from_report_rule_based(
    report_text: str,
    study_uid: Optional[str] = None,
    method: str = "extractor-rule-based-0.1",
) -> ExtractionResult:
    """
    Extract findings using rule-based logic.

    Reuses existing report_label_rules.py extractor for deterministic baseline.
    Conservative: unclear cases become UNCERTAIN, not NEGATIVE.

    Args:
        report_text: Raw MRI report text (English or Spanish)
        study_uid: Optional study identifier
        method: Extraction method identifier

    Returns:
        ExtractionResult with structured findings
    """
    # Use existing rule-based extractor
    extractor = RuleBasedExtractor(profile="0.2.2")
    try:
        report_labels = extractor.extract(report_text, study_uid=study_uid)
    except Exception as e:
        # If extraction fails, return all NOT_MENTIONED
        report_labels = None

    findings = []

    if report_labels and report_labels.supported:
        for label in report_labels.labels:
            # Focus on 3 targets only (Phase 19D)
            if label.target not in ("ACL", "Medial Meniscus", "Lateral Meniscus"):
                continue

            state = LabelState(label.state.value)
            confidence = _state_to_confidence(state)

            finding = Finding(
                target=label.target,
                state=state,
                confidence=confidence,
                evidence_span=label.evidence,
                evidence_section=label.section,
                location_specificity=_estimate_location_specificity(label.evidence) if label.evidence else None,
                hedging_flags=tuple(_extract_hedging_flags(label.evidence) if label.evidence else []),
                reason=label.reason,
            )
            findings.append(finding)

    # Ensure all 3 targets are present
    found_targets = {f.target for f in findings}
    for target in ("ACL", "Medial Meniscus", "Lateral Meniscus"):
        if target not in found_targets:
            findings.append(Finding(
                target=target,
                state=LabelState.NOT_MENTIONED,
                confidence=1.0,
                evidence_span=None,
                reason="extraction_failed_or_unsupported" if report_labels is None else "not_mentioned_in_report",
            ))

    return ExtractionResult(
        study_uid=study_uid or report_labels.study_uid if report_labels else "unknown",
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_id="rule-based-report-label-rules-0.2.2",
        prompt_version=method,
        language_detected=report_labels.language if report_labels else "unknown",
        findings=tuple(findings),
        processing_notes=tuple([f"rule_based_extraction"]),
    )


def _state_to_confidence(state: LabelState) -> float:
    """Map label state to extraction confidence (rule-based)."""
    if state in (LabelState.POSITIVE, LabelState.NEGATIVE):
        return 0.90  # High confidence for decided states
    elif state == LabelState.UNCERTAIN:
        return 0.60  # Low confidence for uncertain
    else:  # NOT_MENTIONED
        return 1.0   # High confidence in absence


def _estimate_location_specificity(evidence_text: str) -> Optional[str]:
    """Estimate whether location is specific (high/medium/low)."""
    if not evidence_text:
        return None

    text_lower = evidence_text.lower()

    # High specificity: explicit side or horn
    if any(x in text_lower for x in ["posterior horn", "anterior horn", "body", "root", "left", "right", "medial", "lateral"]):
        return "high"

    # Medium specificity: general location
    if any(x in text_lower for x in ["meniscus", "meniscal", "acl", "ligament"]):
        return "medium"

    # Low specificity: vague
    return "low"


def _extract_hedging_flags(evidence_text: str) -> list[str]:
    """Extract hedging/uncertainty markers from evidence text."""
    flags = []
    text_lower = evidence_text.lower()

    hedging_patterns = {
        "possible": r"\bpossible\b",
        "likely": r"\blikely\b",
        "suspected": r"\bsuspected\b",
        "cannot_exclude": r"\b(?:cannot|can't|unable to)\s+(?:exclude|rule out)\b",
        "rule_out": r"\b(?:r/o|rule\s+out)\b",
        "uncertain": r"\b(?:uncertain|unclear)\b",
        "suggests": r"\bsuggests?\b",
        "may_represent": r"\bmay\s+(?:represent|be)\b",
        "degenerative": r"\bdegenerative\b",
    }

    for flag_name, pattern in hedging_patterns.items():
        if re.search(pattern, text_lower):
            flags.append(flag_name)

    return flags
