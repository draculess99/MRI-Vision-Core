"""Integration tests: full three-agent pipeline on synthetic reports."""

import pytest

from mri_core.report_agents import (
    process_report,
    LabelState,
    VerificationStatus,
    AdjudicationConsensus,
)
from tests.fixtures_synthetic_reports import get_all_synthetic_reports


class TestThreeAgentPipeline:
    """Full end-to-end pipeline tests."""

    def test_simple_acl_tear(self):
        """Test extraction, verification, adjudication on simple positive case."""
        report = get_all_synthetic_reports()["simple_acl_tear"]

        result = process_report(report["text"], study_uid=report["study_uid"])

        # Should extract ACL as positive
        acl_finding = next(f for f in result.extraction.findings if f.target == "ACL")
        assert acl_finding.state == LabelState.POSITIVE
        assert acl_finding.evidence_span is not None
        assert "tear" in acl_finding.evidence_span.lower()

        # Verification should support it
        acl_verification = next(f for f in result.verification.findings if f.target == "ACL")
        assert acl_verification.verification_status == VerificationStatus.SUPPORTED

        # Adjudication should agree
        acl_adjudicated = next(f for f in result.adjudication.final_labels if f.target == "ACL")
        assert acl_adjudicated.final_state == LabelState.POSITIVE
        assert acl_adjudicated.consensus == AdjudicationConsensus.UNANIMOUS
        assert acl_adjudicated.human_review_required is False

    def test_negation_with_intact(self):
        """Test all-negative case with explicit 'intact' language."""
        report = get_all_synthetic_reports()["negation_with_intact"]

        result = process_report(report["text"], study_uid=report["study_uid"])

        # All three targets should be NEGATIVE
        for target in ["ACL", "Medial Meniscus", "Lateral Meniscus"]:
            adj_finding = next(f for f in result.adjudication.final_labels if f.target == target)
            assert adj_finding.final_state == LabelState.NEGATIVE, f"{target} should be negative"
            assert adj_finding.human_review_required is False

    def test_uncertain_medial_meniscus(self):
        """Test extraction of uncertain finding."""
        report = get_all_synthetic_reports()["uncertain_medial_meniscus"]

        result = process_report(report["text"], study_uid=report["study_uid"])

        # Medial meniscus should be uncertain
        medial_finding = next(
            f for f in result.extraction.findings if f.target == "Medial Meniscus"
        )
        assert medial_finding.state == LabelState.UNCERTAIN

        # Adjudication should preserve uncertainty and flag for review
        medial_adj = next(
            f for f in result.adjudication.final_labels if f.target == "Medial Meniscus"
        )
        assert medial_adj.final_state == LabelState.UNCERTAIN
        assert medial_adj.human_review_required is True

    def test_medial_lateral_confusion(self):
        """Test lateral meniscus finding (avoid lateral/medial confusion)."""
        report = get_all_synthetic_reports()["medial_lateral_confusion_risk"]

        result = process_report(report["text"], study_uid=report["study_uid"])

        # Lateral should be positive
        lateral_adj = next(
            f for f in result.adjudication.final_labels if f.target == "Lateral Meniscus"
        )
        assert lateral_adj.final_state == LabelState.POSITIVE

        # Medial should be negative
        medial_adj = next(
            f for f in result.adjudication.final_labels if f.target == "Medial Meniscus"
        )
        assert medial_adj.final_state == LabelState.NEGATIVE

    def test_missing_finding(self):
        """Test NOT_MENTIONED state (report silent on lateral meniscus)."""
        report = get_all_synthetic_reports()["missing_finding"]

        result = process_report(report["text"], study_uid=report["study_uid"])

        # Lateral should be NOT_MENTIONED
        lateral_adj = next(
            f for f in result.adjudication.final_labels if f.target == "Lateral Meniscus"
        )
        assert lateral_adj.final_state == LabelState.NOT_MENTIONED
        assert lateral_adj.evidence_span is None

    def test_double_negation(self):
        """Test double-negative phrasing."""
        report = get_all_synthetic_reports()["double_negation"]

        result = process_report(report["text"], study_uid=report["study_uid"])

        # All should be negative despite double negation
        for target in ["ACL", "Medial Meniscus", "Lateral Meniscus"]:
            adj_finding = next(f for f in result.adjudication.final_labels if f.target == target)
            assert adj_finding.final_state == LabelState.NEGATIVE

    def test_escalation_tracking(self):
        """Test that uncertain findings are escalated."""
        report = get_all_synthetic_reports()["hedging_and_history"]

        result = process_report(report["text"], study_uid=report["study_uid"])

        # Should have escalations
        assert result.adjudication.escalated_count > 0
        assert len(result.adjudication.escalation_reasons) > 0

        # Escalated findings should have review_required = True
        escalated_findings = [
            f for f in result.adjudication.final_labels if f.human_review_required
        ]
        assert len(escalated_findings) > 0


class TestPipelineConsistency:
    """Tests for consistency across the three agents."""

    def test_study_uid_flows_through_pipeline(self):
        """Study UID should be consistent across all three agents."""
        report = get_all_synthetic_reports()["simple_acl_tear"]
        study_uid = report["study_uid"]

        result = process_report(report["text"], study_uid=study_uid)

        assert result.extraction.study_uid == study_uid
        assert result.verification.study_uid == study_uid
        assert result.adjudication.study_uid == study_uid

    def test_target_count_preserved(self):
        """All three agents should report on same 3 targets."""
        report = get_all_synthetic_reports()["simple_acl_tear"]

        result = process_report(report["text"])

        assert len(result.extraction.findings) == 3
        assert len(result.verification.findings) == 3
        assert len(result.adjudication.final_labels) == 3

        # All should be ACL, Medial Meniscus, Lateral Meniscus
        targets = {f.target for f in result.extraction.findings}
        assert targets == {"ACL", "Medial Meniscus", "Lateral Meniscus"}

    def test_evidence_span_consistency(self):
        """Evidence spans should be traceable through pipeline."""
        report = get_all_synthetic_reports()["simple_acl_tear"]

        result = process_report(report["text"])

        acl_extraction = next(f for f in result.extraction.findings if f.target == "ACL")
        acl_verification = next(f for f in result.verification.findings if f.target == "ACL")
        acl_adjudication = next(f for f in result.adjudication.final_labels if f.target == "ACL")

        # If extraction has evidence, verification should find it
        if acl_extraction.evidence_span:
            assert acl_verification.supporting_text is not None

        # Adjudication should report evidence
        if acl_adjudication.final_state != LabelState.NOT_MENTIONED:
            assert acl_adjudication.evidence_span is not None or acl_adjudication.human_review_required


class TestComparisonAgainstSynthetic:
    """Compare agent outputs against expected labels for synthetic reports."""

    def test_all_synthetic_reports_processable(self):
        """All 8 synthetic reports should process without error."""
        for key, report in get_all_synthetic_reports().items():
            result = process_report(report["text"], study_uid=report["study_uid"])

            # Should have all three agent outputs
            assert result.extraction is not None
            assert result.verification is not None
            assert result.adjudication is not None

            # Should be fully serializable
            serialized = result.to_json()
            assert serialized is not None
            assert len(serialized) > 100

    def test_rule_based_extractions_match_expected(self):
        """Rule-based extraction should produce expected states for synthetic reports."""
        for key, report in get_all_synthetic_reports().items():
            result = process_report(report["text"])

            # Build map of extracted states
            extracted_states = {
                f.target: f.state for f in result.extraction.findings
            }

            # Compare against expected
            for target, expected_state in report["targets"].items():
                extracted_state = extracted_states.get(target)
                expected = LabelState(expected_state)

                # Rule-based may not always get uncertain right, but should not
                # call negative what should be positive or vice versa
                if expected == LabelState.POSITIVE:
                    assert extracted_state in (LabelState.POSITIVE, LabelState.UNCERTAIN), \
                        f"{key}/{target}: expected positive or uncertain, got {extracted_state}"
                elif expected == LabelState.NEGATIVE:
                    assert extracted_state in (LabelState.NEGATIVE, LabelState.UNCERTAIN), \
                        f"{key}/{target}: expected negative or uncertain, got {extracted_state}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
