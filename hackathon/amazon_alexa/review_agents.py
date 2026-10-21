"""Deterministic review-support agents for MRI evidence analysis.

Provides immutable review support packets that synthesize evidence and
reviewer-focused checks. No network calls, no LLM, deterministic only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hackathon.amazon_alexa.tools import MRIEvidence, ReviewExplanation, ReviewStatus


@dataclass(frozen=True)
class ReviewSupportPacket:
    """Immutable review support packet synthesizing evidence and checks."""

    evidence_summary: str
    """Plain-language summary of the case evidence."""

    critic_findings: list[str]
    """Deterministic critic findings and flags based on evidence."""

    reviewer_checks: list[str]
    """Reviewer-focused checks and verification steps."""

    approval_required: bool = field(init=False, default=True)
    """Always True: the copilot cannot approve cases. Non-configurable invariant."""


def generate_review_support(
    evidence: MRIEvidence, explanation: ReviewExplanation
) -> ReviewSupportPacket:
    """Generate a deterministic review support packet.

    Args:
        evidence: MRI evidence record from analyze_mri_evidence
        explanation: Natural language explanation from explain_review_status

    Returns:
        Immutable review support packet for reviewer guidance
    """
    # Evidence summary: synthesize key facts
    evidence_summary = (
        f"**Case {evidence.case_id}** — "
        f"Quality {evidence.quality_score:.2f}, "
        f"Foreground {evidence.foreground_percentage:.1f}%. "
        f"Status: {evidence.status.value}. "
        f"{explanation.summary}"
    )

    # Critic findings: deterministic checks on the evidence
    critic_findings = []

    # Quality-based findings
    if evidence.quality_score < 0.70:
        critic_findings.append(
            f"⚠️ Low image quality ({evidence.quality_score:.2f}): "
            "Assessment confidence is limited. Original images must be reviewed."
        )
    elif evidence.quality_score < 0.85:
        critic_findings.append(
            f"📌 Acceptable image quality ({evidence.quality_score:.2f}) "
            "with some noise or artifacts. Verify findings in source images."
        )
    else:
        critic_findings.append(
            f"✓ Good image quality ({evidence.quality_score:.2f}). "
            "Evidence metrics are reliable."
        )

    # Foreground coverage findings
    if evidence.foreground_percentage < 10:
        critic_findings.append(
            f"⚠️ Very low foreground coverage ({evidence.foreground_percentage:.1f}%). "
            "Region of interest may be missing or misaligned."
        )
    elif evidence.foreground_percentage > 80:
        critic_findings.append(
            f"📌 High foreground coverage ({evidence.foreground_percentage:.1f}%). "
            "Verify that background was correctly excluded."
        )

    # Status-based findings
    if evidence.status == ReviewStatus.CONSISTENT:
        critic_findings.append(
            "No significant measured changes detected between baseline and current."
        )
    elif evidence.status == ReviewStatus.NEEDS_REVIEW:
        critic_findings.append(
            f"Notable change detected: {evidence.change_description}"
        )
    else:  # UNCERTAIN
        critic_findings.append(
            f"Assessment uncertain: {evidence.change_description}"
        )

    # Reviewer-focused checks: actionable verification steps
    reviewer_checks = [
        "✓ Compare source images side-by-side for visual confirmation.",
        "✓ Verify that the foreground region was correctly identified and measured.",
        "✓ Check image alignment between baseline and current acquisition.",
        "✓ Verify the relevance and magnitude of any measured changes.",
        "✓ Document reviewer name and any findings in the approval record.",
    ]

    # Add status-specific checks
    if evidence.status == ReviewStatus.CONSISTENT:
        reviewer_checks.insert(
            0, "✓ Confirm that consistency findings match visual inspection."
        )
    elif evidence.status == ReviewStatus.NEEDS_REVIEW:
        reviewer_checks.insert(
            0,
            "✓ Investigate the cause and significance of the measured change.",
        )
    else:  # UNCERTAIN
        reviewer_checks.insert(
            0, "✓ If quality permits, use alternative measurements or methods."
        )

    return ReviewSupportPacket(
        evidence_summary=evidence_summary,
        critic_findings=critic_findings,
        reviewer_checks=reviewer_checks,
    )
