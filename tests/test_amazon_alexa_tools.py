"""Tests for Amazon Alexa+ MRI Evidence Copilot tool layer.

Tests the three core deterministic tools:
- analyze_mri_evidence
- explain_review_status
- request_human_approval
"""

import pytest

from hackathon.amazon_alexa.tools import (
    ReviewStatus,
    analyze_mri_evidence,
    explain_review_status,
    request_human_approval,
)


class TestAnalyzeMRIEvidence:
    """Test analyze_mri_evidence tool."""

    def test_loads_consistent_case(self):
        evidence = analyze_mri_evidence("case-001-consistent")
        assert evidence.case_id == "case-001-consistent"
        assert evidence.status == ReviewStatus.CONSISTENT
        assert evidence.quality_score > 0.90
        assert "no change" not in evidence.change_description.lower()

    def test_loads_change_case(self):
        evidence = analyze_mri_evidence("case-002-change")
        assert evidence.case_id == "case-002-change"
        assert evidence.status == ReviewStatus.NEEDS_REVIEW
        assert evidence.quality_score > 0.85

    def test_loads_uncertain_case(self):
        evidence = analyze_mri_evidence("case-003-uncertain")
        assert evidence.case_id == "case-003-uncertain"
        assert evidence.status == ReviewStatus.UNCERTAIN
        assert evidence.quality_score < 0.75

    def test_raises_on_invalid_case(self):
        with pytest.raises(ValueError, match="not found"):
            analyze_mri_evidence("case-999-nonexistent")

    def test_all_fields_populated(self):
        evidence = analyze_mri_evidence("case-001-consistent")
        assert evidence.baseline_date
        assert evidence.current_date
        assert 0.0 <= evidence.quality_score <= 1.0
        assert 0.0 <= evidence.foreground_percentage <= 100.0
        assert evidence.mean_intensity > 0
        assert evidence.change_description

    def test_deterministic_repeated_calls(self):
        e1 = analyze_mri_evidence("case-002-change")
        e2 = analyze_mri_evidence("case-002-change")
        assert e1 == e2


class TestExplainReviewStatus:
    """Test explain_review_status tool."""

    def test_consistent_case_explanation(self):
        evidence = analyze_mri_evidence("case-001-consistent")
        explanation = explain_review_status(evidence)
        assert explanation.status == ReviewStatus.CONSISTENT
        assert "consistent" in explanation.summary.lower()
        assert explanation.requires_human_approval is True
        assert len(explanation.quality_notes) > 0
        assert len(explanation.change_notes) > 0

    def test_change_case_explanation(self):
        evidence = analyze_mri_evidence("case-002-change")
        explanation = explain_review_status(evidence)
        assert explanation.status == ReviewStatus.NEEDS_REVIEW
        assert ("review" in explanation.summary.lower() or "differences" in explanation.summary.lower())
        assert explanation.requires_human_approval is True

    def test_uncertain_case_explanation(self):
        evidence = analyze_mri_evidence("case-003-uncertain")
        explanation = explain_review_status(evidence)
        assert explanation.status == ReviewStatus.UNCERTAIN
        assert "uncertain" in explanation.summary.lower() or "quality" in explanation.summary.lower()
        assert explanation.requires_human_approval is True

    def test_always_requires_human_approval(self):
        for case_id in ["case-001-consistent", "case-002-change", "case-003-uncertain"]:
            evidence = analyze_mri_evidence(case_id)
            explanation = explain_review_status(evidence)
            assert explanation.requires_human_approval is True

    def test_quality_assessment_non_empty(self):
        evidence = analyze_mri_evidence("case-002-change")
        explanation = explain_review_status(evidence)
        assert len(explanation.quality_notes) >= 1
        assert all(isinstance(n, str) for n in explanation.quality_notes)

    def test_change_notes_non_empty(self):
        evidence = analyze_mri_evidence("case-002-change")
        explanation = explain_review_status(evidence)
        assert len(explanation.change_notes) >= 1
        assert all(isinstance(n, str) for n in explanation.change_notes)

    def test_deterministic_repeated_calls(self):
        evidence = analyze_mri_evidence("case-003-uncertain")
        e1 = explain_review_status(evidence)
        e2 = explain_review_status(evidence)
        assert e1 == e2


class TestRequestHumanApproval:
    """Test request_human_approval tool."""

    def test_creates_approval_request(self):
        approval = request_human_approval(
            "case-001-consistent", "Dr. Smith", "checked source"
        )
        assert approval.case_id == "case-001-consistent"
        assert approval.reviewer_name == "Dr. Smith"
        assert approval.comment == "checked source"

    def test_approval_state_always_pending(self):
        approval = request_human_approval("case-001-consistent", "Dr. Jones")
        assert approval.approval_state.value == "PENDING_HUMAN_APPROVAL"

    def test_raises_on_empty_reviewer(self):
        with pytest.raises(ValueError, match="named human reviewer"):
            request_human_approval("case-001-consistent", "")

    def test_raises_on_whitespace_only_reviewer(self):
        with pytest.raises(ValueError, match="named human reviewer"):
            request_human_approval("case-001-consistent", "   ")

    def test_strips_reviewer_whitespace(self):
        approval = request_human_approval("case-001-consistent", "  Dr. Lee  ")
        assert approval.reviewer_name == "Dr. Lee"

    def test_optional_comment_can_be_empty(self):
        approval = request_human_approval("case-001-consistent", "Dr. Brown")
        assert approval.comment is None

    def test_non_empty_comment_preserved(self):
        approval = request_human_approval(
            "case-001-consistent", "Dr. White", "reviewed carefully"
        )
        assert approval.comment == "reviewed carefully"

    def test_deterministic_repeated_calls(self):
        r1 = request_human_approval("case-002-change", "Dr. Smith", "OK")
        r2 = request_human_approval("case-002-change", "Dr. Smith", "OK")
        assert r1 == r2
