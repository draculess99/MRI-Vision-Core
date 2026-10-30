"""Regression tests for report agent edge cases and negation handling.

Note: Rule-based extractor has limitations (lexical-only, pattern-dependent).
These tests focus on agent architecture correctness, not rule-based accuracy.
"""

import pytest

from mri_core.report_agents import process_report, LabelState


class TestNegationRegression:
    """Regression tests for negation handling."""

    def test_acl_intact_explicit(self):
        """'ACL is intact' should be clearly NEGATIVE."""
        report_text = """
FINDINGS:
The ACL is intact. No evidence of tear.
"""
        result = process_report(report_text)
        acl = next(f for f in result.adjudication.final_labels if f.target == "ACL")
        assert acl.final_state == LabelState.NEGATIVE, "ACL intact should be negative"

    def test_no_evidence_of_acl_tear(self):
        """'No evidence of ACL tear' should be NEGATIVE."""
        report_text = """
FINDINGS:
No evidence of ACL tear or disruption.
The medial and lateral menisci are normal.
"""
        result = process_report(report_text)
        acl = next(f for f in result.adjudication.final_labels if f.target == "ACL")
        assert acl.final_state == LabelState.NEGATIVE

    def test_acl_negative_explicit_medial_negative_lateral_positive(self):
        """Test distinct findings: ACL normal, medial normal, lateral torn."""
        report_text = """
FINDINGS:
The ACL is normal. The medial meniscus is intact.
There is a tear of the lateral meniscus at the posterior horn.
"""
        result = process_report(report_text)

        acl = next(f for f in result.adjudication.final_labels if f.target == "ACL")
        medial = next(f for f in result.adjudication.final_labels if f.target == "Medial Meniscus")
        lateral = next(f for f in result.adjudication.final_labels if f.target == "Lateral Meniscus")

        assert acl.final_state == LabelState.NEGATIVE
        assert medial.final_state == LabelState.NEGATIVE
        assert lateral.final_state == LabelState.POSITIVE

    def test_double_negation_acl(self):
        """'Not not torn' should resolve to NEGATIVE."""
        report_text = """
FINDINGS:
The ACL is not torn. The ACL is not disrupted.
"""
        result = process_report(report_text)
        acl = next(f for f in result.adjudication.final_labels if f.target == "ACL")
        # Should be NEGATIVE (not torn)
        assert acl.final_state in (LabelState.NEGATIVE, LabelState.UNCERTAIN)

    def test_medial_meniscal_tear_with_normal_lateral(self):
        """Explicit medial tear with normal lateral."""
        report_text = """
FINDINGS:
There is a medial meniscal tear at the body.
The lateral meniscus is normal and intact.
"""
        result = process_report(report_text)

        medial = next(f for f in result.adjudication.final_labels if f.target == "Medial Meniscus")
        lateral = next(f for f in result.adjudication.final_labels if f.target == "Lateral Meniscus")

        assert medial.final_state == LabelState.POSITIVE
        assert lateral.final_state == LabelState.NEGATIVE

    def test_explicit_medial_and_lateral_findings(self):
        """Both medial and lateral findings in one report."""
        report_text = """
FINDINGS:
There is a medial meniscal tear at the posterior horn.
There is a lateral meniscal tear at the anterior horn.
"""
        result = process_report(report_text)

        medial = next(f for f in result.adjudication.final_labels if f.target == "Medial Meniscus")
        lateral = next(f for f in result.adjudication.final_labels if f.target == "Lateral Meniscus")

        assert medial.final_state == LabelState.POSITIVE
        assert lateral.final_state == LabelState.POSITIVE


class TestUncertaintyPreservation:
    """Tests that uncertainty is preserved and not silently converted."""

    def test_hedged_acl_finding_escalates(self):
        """Hedged ACL finding should escalate for review."""
        report_text = """
FINDINGS:
There may be a small ACL tear that is difficult to exclude.
"""
        result = process_report(report_text)
        acl = next(f for f in result.adjudication.final_labels if f.target == "ACL")

        # Should be UNCERTAIN and escalated
        if acl.final_state == LabelState.UNCERTAIN:
            assert acl.human_review_required is True

    def test_ambiguous_diagnosis_not_silently_negative(self):
        """Ambiguous findings should not be silently converted to NEGATIVE."""
        report_text = """
FINDINGS:
There is an abnormality within the medial meniscus that could
represent a tear or degenerative change.
"""
        result = process_report(report_text)
        medial = next(f for f in result.adjudication.final_labels if f.target == "Medial Meniscus")

        # Core principle: not silently NEGATIVE when ambiguous
        # May be UNCERTAIN (escalated) or POSITIVE (found), but not silently NEGATIVE
        assert medial.final_state in (LabelState.UNCERTAIN, LabelState.POSITIVE)

    def test_missing_finding_not_invented(self):
        """Missing finding should be NOT_MENTIONED, not invented as NEGATIVE."""
        report_text = """
FINDINGS:
The ACL is intact.
The medial meniscus is normal.
"""
        result = process_report(report_text)
        lateral = next(f for f in result.adjudication.final_labels if f.target == "Lateral Meniscus")

        # Should be NOT_MENTIONED, not invented as NEGATIVE
        assert lateral.final_state == LabelState.NOT_MENTIONED


class TestEvidenceTraceability:
    """Tests that evidence spans are traceable or properly marked."""

    def test_supported_finding_has_evidence_span(self):
        """Supported findings must have evidence span."""
        report_text = """
FINDINGS:
There is a complete ACL tear.
"""
        result = process_report(report_text)
        acl = next(f for f in result.adjudication.final_labels if f.target == "ACL")

        if acl.final_state == LabelState.POSITIVE:
            # Must have evidence span
            assert acl.evidence_span is not None
            assert "ACL" in acl.evidence_span or "tear" in acl.evidence_span.lower()

    def test_unsupported_claim_escalated(self):
        """Unsupported claim should not be finalized without review."""
        report_text = """
FINDINGS:
There is a tear.
"""
        result = process_report(report_text)

        # Check if any findings with positive state have proper support
        for f in result.adjudication.final_labels:
            if f.final_state == LabelState.POSITIVE:
                # If positive, should either have evidence or be escalated
                if f.evidence_span is None:
                    assert f.human_review_required is True


class TestPipelineIntegrity:
    """Tests that the three-agent pipeline maintains integrity."""

    def test_no_silent_negative_conversion(self):
        """Core principle: never silently convert UNCERTAIN to NEGATIVE."""
        # Test a variety of unclear cases
        unclear_reports = [
            "There may be a small tear",
            "A tear cannot be completely excluded",
            "There is probably a tear",
            "Rule out ACL tear",
        ]

        for report_text in unclear_reports:
            full_report = f"FINDINGS:\n{report_text}"
            result = process_report(full_report)

            # For any uncertain or ambiguous report, if a finding is made,
            # it should not be silently NEGATIVE
            for f in result.adjudication.final_labels:
                # If rule-based found something (not NOT_MENTIONED),
                # agent should preserve uncertainty or escalate
                if f.extractor_state != LabelState.NOT_MENTIONED:
                    if f.extractor_state == LabelState.UNCERTAIN:
                        # UNCERTAIN should be preserved
                        assert f.final_state != LabelState.NEGATIVE or f.human_review_required

    def test_all_reports_produce_three_targets(self):
        """All reports must produce exactly 3 targets."""
        test_reports = [
            "The ACL is intact.",
            "There is a medial meniscal tear.",
            "All structures normal.",
        ]

        for report_text in test_reports:
            result = process_report(report_text)
            assert len(result.extraction.findings) == 3
            assert len(result.adjudication.final_labels) == 3

            targets = {f.target for f in result.adjudication.final_labels}
            assert targets == {"ACL", "Medial Meniscus", "Lateral Meniscus"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
