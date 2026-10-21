"""Tests for deterministic review-support agents."""

import pytest

from hackathon.amazon_alexa.review_agents import (
    ReviewSupportPacket,
    generate_review_support,
)
from hackathon.amazon_alexa.tools import (
    MRIEvidence,
    ReviewExplanation,
    ReviewStatus,
    analyze_mri_evidence,
    explain_review_status,
)


class TestReviewSupportPacket:
    """Test ReviewSupportPacket dataclass."""

    def test_packet_is_immutable(self):
        """ReviewSupportPacket is frozen and immutable."""
        packet = ReviewSupportPacket(
            evidence_summary="Test",
            critic_findings=["Finding 1"],
            reviewer_checks=["Check 1"],
        )

        # Verify frozen by attempting to mutate
        with pytest.raises(AttributeError):
            packet.evidence_summary = "Modified"

    def test_packet_has_approval_required_always_true(self):
        """approval_required is always True and non-configurable."""
        packet = ReviewSupportPacket(
            evidence_summary="Test",
            critic_findings=[],
            reviewer_checks=[],
        )
        assert packet.approval_required is True

    def test_packet_approval_required_cannot_be_disabled_by_caller(self):
        """approval_required cannot be set by callers; it is always True."""
        # Attempting to pass approval_required to __init__ raises TypeError
        with pytest.raises(TypeError):
            ReviewSupportPacket(
                evidence_summary="Test",
                critic_findings=[],
                reviewer_checks=[],
                approval_required=False,
            )


class TestGenerateReviewSupport:
    """Test generate_review_support function."""

    def test_packet_generation_case_001_consistent(self):
        """Generate packet for case-001-consistent (no change)."""
        evidence = analyze_mri_evidence("case-001-consistent")
        explanation = explain_review_status(evidence)

        packet = generate_review_support(evidence, explanation)

        # Verify packet structure
        assert isinstance(packet, ReviewSupportPacket)
        assert packet.evidence_summary
        assert packet.critic_findings
        assert packet.reviewer_checks
        assert packet.approval_required is True

        # Verify summary includes case ID and status
        assert "case-001-consistent" in packet.evidence_summary
        assert "CONSISTENT" in packet.evidence_summary

        # Verify critic findings mention no change
        summary_text = " ".join(packet.critic_findings).lower()
        assert "consistent" in summary_text or "no significant" in summary_text

    def test_packet_generation_case_002_change(self):
        """Generate packet for case-002-change (needs review)."""
        evidence = analyze_mri_evidence("case-002-change")
        explanation = explain_review_status(evidence)

        packet = generate_review_support(evidence, explanation)

        assert "case-002-change" in packet.evidence_summary
        assert "NEEDS_REVIEW" in packet.evidence_summary

        # Verify critic findings mention the change
        summary_text = " ".join(packet.critic_findings)
        assert "change" in summary_text.lower()

    def test_packet_generation_case_003_uncertain(self):
        """Generate packet for case-003-uncertain (low quality)."""
        evidence = analyze_mri_evidence("case-003-uncertain")
        explanation = explain_review_status(evidence)

        packet = generate_review_support(evidence, explanation)

        assert "case-003-uncertain" in packet.evidence_summary
        assert "UNCERTAIN" in packet.evidence_summary

        # Verify low quality is flagged
        summary_text = " ".join(packet.critic_findings).lower()
        assert "low" in summary_text or "quality" in summary_text or "uncertain" in summary_text

    def test_packet_always_requires_approval(self):
        """All packets always require human approval."""
        for case_id in ["case-001-consistent", "case-002-change", "case-003-uncertain"]:
            evidence = analyze_mri_evidence(case_id)
            explanation = explain_review_status(evidence)
            packet = generate_review_support(evidence, explanation)

            assert packet.approval_required is True

    def test_packet_has_reviewer_checks(self):
        """All packets include reviewer checks."""
        evidence = analyze_mri_evidence("case-001-consistent")
        explanation = explain_review_status(evidence)

        packet = generate_review_support(evidence, explanation)

        assert len(packet.reviewer_checks) > 0
        # All checks should start with ✓ or similar
        for check in packet.reviewer_checks:
            assert check.strip()  # Non-empty

    def test_quality_score_affects_critic_findings(self):
        """Quality score is reflected in critic findings."""
        # Good quality evidence
        evidence_good = analyze_mri_evidence("case-001-consistent")  # 0.95 quality
        explanation_good = explain_review_status(evidence_good)
        packet_good = generate_review_support(evidence_good, explanation_good)

        # Low quality evidence
        evidence_low = analyze_mri_evidence("case-003-uncertain")  # 0.68 quality
        explanation_low = explain_review_status(evidence_low)
        packet_low = generate_review_support(evidence_low, explanation_low)

        # Check that quality differences are reflected
        good_text = " ".join(packet_good.critic_findings).lower()
        low_text = " ".join(packet_low.critic_findings).lower()

        assert ("good" in good_text or "reliable" in good_text)
        assert ("low" in low_text or "limited" in low_text)

    def test_foreground_coverage_affects_critic_findings(self):
        """Foreground coverage is reflected in critic findings."""
        evidence = analyze_mri_evidence("case-002-change")  # 62.1% foreground
        explanation = explain_review_status(evidence)
        packet = generate_review_support(evidence, explanation)

        # Check that foreground is mentioned
        findings_text = " ".join(packet.critic_findings)
        assert "foreground" in findings_text.lower()

    def test_packet_evidence_summary_format(self):
        """Evidence summary includes case ID, quality, foreground, and status."""
        evidence = analyze_mri_evidence("case-002-change")
        explanation = explain_review_status(evidence)

        packet = generate_review_support(evidence, explanation)

        summary = packet.evidence_summary
        # Check for key components
        assert "case-002-change" in summary
        assert "Quality" in summary
        assert "Foreground" in summary
        assert "NEEDS_REVIEW" in summary

    def test_consistent_case_checks_are_appropriate(self):
        """Case-001-consistent generates appropriate checks."""
        evidence = analyze_mri_evidence("case-001-consistent")
        explanation = explain_review_status(evidence)
        packet = generate_review_support(evidence, explanation)

        # Should have a consistency-focused check
        checks_text = " ".join(packet.reviewer_checks).lower()
        assert "consistency" in checks_text or "consistent" in checks_text

    def test_change_case_checks_are_appropriate(self):
        """Case-002-change generates appropriate checks."""
        evidence = analyze_mri_evidence("case-002-change")
        explanation = explain_review_status(evidence)
        packet = generate_review_support(evidence, explanation)

        # Should have a change investigation check
        checks_text = " ".join(packet.reviewer_checks).lower()
        assert "change" in checks_text or "investigate" in checks_text

    def test_uncertain_case_checks_are_appropriate(self):
        """Case-003-uncertain generates appropriate checks."""
        evidence = analyze_mri_evidence("case-003-uncertain")
        explanation = explain_review_status(evidence)
        packet = generate_review_support(evidence, explanation)

        # Should suggest alternative methods due to quality
        checks_text = " ".join(packet.reviewer_checks).lower()
        assert (
            "alternative" in checks_text
            or "quality" in checks_text
            or "measurement" in checks_text
        )

    def test_custom_evidence_object(self):
        """generate_review_support works with custom evidence objects."""
        custom_evidence = MRIEvidence(
            case_id="custom-test",
            baseline_date="2025-01-01",
            current_date="2025-01-15",
            quality_score=0.85,
            foreground_percentage=50.0,
            mean_intensity=100.0,
            status=ReviewStatus.NEEDS_REVIEW,
            change_description="Custom change detected.",
        )
        custom_explanation = ReviewExplanation(
            status=ReviewStatus.NEEDS_REVIEW,
            summary="Custom case requires review.",
            quality_notes=["Custom quality note."],
            change_notes=["Custom change note."],
            requires_human_approval=True,
        )

        packet = generate_review_support(custom_evidence, custom_explanation)

        assert "custom-test" in packet.evidence_summary
        assert len(packet.critic_findings) > 0
        assert len(packet.reviewer_checks) > 0

    def test_generation_does_not_alter_source_evidence(self):
        """Generating a packet does not modify the source evidence or explanation."""
        evidence = analyze_mri_evidence("case-002-change")
        explanation = explain_review_status(evidence)

        # Capture original state
        original_case_id = evidence.case_id
        original_status = evidence.status
        original_quality = evidence.quality_score

        # Generate packet
        packet = generate_review_support(evidence, explanation)

        # Verify source is unchanged
        assert evidence.case_id == original_case_id
        assert evidence.status == original_status
        assert evidence.quality_score == original_quality

        # Packet should reference the same case ID
        assert original_case_id in packet.evidence_summary

    def test_no_packet_contains_approved_by_human_state(self):
        """No generated packet ever contains or references APPROVED_BY_HUMAN."""
        for case_id in ["case-001-consistent", "case-002-change", "case-003-uncertain"]:
            evidence = analyze_mri_evidence(case_id)
            explanation = explain_review_status(evidence)
            packet = generate_review_support(evidence, explanation)

            # Combine all text from packet
            all_text = (
                packet.evidence_summary
                + " "
                + " ".join(packet.critic_findings)
                + " "
                + " ".join(packet.reviewer_checks)
            )

            # Must not contain any reference to automated approval
            assert "APPROVED_BY_HUMAN" not in all_text
            assert "approved" not in all_text.lower() or "cannot" in all_text.lower()
