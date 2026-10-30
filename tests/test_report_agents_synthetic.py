"""Unit tests for report agents using only synthetic data.

These tests verify the agent framework structure without requiring real RSNA reports
or any external API calls. All tests use synthetic, fully-synthetic reports.

This module can run offline and serves as the baseline for Phase 20A.
"""

import pytest

from mri_core.report_agents.contracts import (
    LabelState,
    VerificationStatus,
    AdjudicationConsensus,
    Finding,
    ExtractionResult,
    VerificationFinding,
    VerificationResult,
    AdjudicatedFinding,
    AdjudicationResult,
)
from tests.fixtures_synthetic_reports import get_synthetic_report, get_all_synthetic_reports


class TestLabelStates:
    """Label state enum tests."""

    def test_four_states_exist(self):
        states = {LabelState.POSITIVE, LabelState.NEGATIVE, LabelState.UNCERTAIN, LabelState.NOT_MENTIONED}
        assert len(states) == 4

    def test_state_serialization(self):
        assert LabelState.POSITIVE.value == "positive"
        assert LabelState.NEGATIVE.value == "negative"
        assert LabelState.UNCERTAIN.value == "uncertain"
        assert LabelState.NOT_MENTIONED.value == "not_mentioned"


class TestFinding:
    """Finding contract tests."""

    def test_finding_requires_valid_confidence(self):
        # Valid range
        f1 = Finding(target="ACL", state=LabelState.POSITIVE, confidence=0.95)
        assert f1.confidence == 0.95

        # Boundary values
        f2 = Finding(target="ACL", state=LabelState.POSITIVE, confidence=0.0)
        f3 = Finding(target="ACL", state=LabelState.POSITIVE, confidence=1.0)

        # Invalid
        with pytest.raises(ValueError):
            Finding(target="ACL", state=LabelState.POSITIVE, confidence=1.5)

        with pytest.raises(ValueError):
            Finding(target="ACL", state=LabelState.POSITIVE, confidence=-0.1)

    def test_finding_immutability(self):
        f = Finding(target="ACL", state=LabelState.POSITIVE, confidence=0.95)
        with pytest.raises(AttributeError):
            f.confidence = 0.9  # frozen dataclass


class TestExtractionResultContract:
    """Extraction output contract tests."""

    def test_extraction_result_serialization(self):
        findings = (
            Finding(
                target="ACL",
                state=LabelState.POSITIVE,
                confidence=0.95,
                evidence_span="complete ACL tear",
                evidence_section="findings",
                location_specificity="high",
            ),
            Finding(
                target="Medial Meniscus",
                state=LabelState.NEGATIVE,
                confidence=0.92,
                evidence_span="medial meniscus intact",
                evidence_section="findings",
            ),
            Finding(
                target="Lateral Meniscus",
                state=LabelState.NEGATIVE,
                confidence=0.90,
                evidence_span="lateral meniscus intact",
                evidence_section="findings",
            ),
        )

        result = ExtractionResult(
            study_uid="1.2.826.synth.001",
            timestamp="2026-10-29T12:00:00Z",
            model_id="claude-sonnet-5-5",
            prompt_version="extractor-0.1",
            language_detected="en",
            findings=findings,
        )

        d = result.to_dict()
        assert d["study_uid"] == "1.2.826.synth.001"
        assert len(d["findings"]) == 3
        assert d["findings"][0]["state"] == "positive"
        assert d["findings"][1]["state"] == "negative"


class TestVerificationResultContract:
    """Verification output contract tests."""

    def test_verification_result_summary_counts(self):
        vf1 = VerificationFinding(
            target="ACL",
            extractor_state=LabelState.POSITIVE,
            verification_status=VerificationStatus.SUPPORTED,
            supporting_text="ACL tear",
        )
        vf2 = VerificationFinding(
            target="Medial Meniscus",
            extractor_state=LabelState.NEGATIVE,
            verification_status=VerificationStatus.SUPPORTED,
            supporting_text="medial meniscus intact",
        )
        vf3 = VerificationFinding(
            target="Lateral Meniscus",
            extractor_state=LabelState.UNCERTAIN,
            verification_status=VerificationStatus.UNSUPPORTED,
            issue="hedging without support",
        )

        result = VerificationResult(
            study_uid="1.2.826.synth.001",
            verification_timestamp="2026-10-29T12:00:01Z",
            verified_by_model="claude-sonnet-5-5",
            findings=(vf1, vf2, vf3),
        )

        d = result.to_dict()
        assert d["summary"]["supported_count"] == 2
        assert d["summary"]["unsupported_count"] == 1
        assert d["summary"]["contradicted_count"] == 0


class TestAdjudicationResultContract:
    """Adjudication output contract tests."""

    def test_adjudication_result_escalation_tracking(self):
        af1 = AdjudicatedFinding(
            target="ACL",
            final_state=LabelState.POSITIVE,
            consensus=AdjudicationConsensus.UNANIMOUS,
            extractor_state=LabelState.POSITIVE,
            verifier_status=VerificationStatus.SUPPORTED,
            evidence_span="ACL tear",
            confidence=0.97,
            human_review_required=False,
        )
        af2 = AdjudicatedFinding(
            target="Medial Meniscus",
            final_state=LabelState.UNCERTAIN,
            consensus=AdjudicationConsensus.RESOLVED_VIA_ESCALATION,
            extractor_state=LabelState.UNCERTAIN,
            verifier_status=VerificationStatus.UNSUPPORTED,
            evidence_span=None,
            confidence=0.0,
            human_review_required=True,
            review_reason="hedged_without_support",
        )

        result = AdjudicationResult(
            study_uid="1.2.826.synth.001",
            adjudication_timestamp="2026-10-29T12:00:02Z",
            schema_version="adjudicator-0.1",
            final_labels=(af1, af2),
            escalated_count=1,
            escalation_reasons=("hedged_without_support",),
        )

        d = result.to_dict()
        assert d["escalated_count"] == 1
        assert len(d["escalation_reasons"]) == 1
        assert result.escalated_count == 1


class TestSyntheticReportCoverage:
    """Verify synthetic reports cover challenge cases."""

    def test_all_synthetic_reports_have_required_fields(self):
        for key, report in get_all_synthetic_reports().items():
            assert "study_uid" in report, f"{key} missing study_uid"
            assert "text" in report, f"{key} missing text"
            assert "targets" in report, f"{key} missing targets"
            assert len(report["targets"]) == 3, f"{key} should have 3 targets"

    def test_target_states_are_valid(self):
        valid_states = {"positive", "negative", "uncertain", "not_mentioned"}
        for key, report in get_all_synthetic_reports().items():
            for target, state in report["targets"].items():
                assert state in valid_states, f"{key}: {target} has invalid state {state}"

    def test_challenge_cases_present(self):
        """Verify synthetic reports cover known challenge scenarios."""
        all_reports = get_all_synthetic_reports()

        # Simple positive case
        assert "simple_acl_tear" in all_reports
        assert all_reports["simple_acl_tear"]["targets"]["ACL"] == "positive"

        # Uncertainty case
        assert "uncertain_medial_meniscus" in all_reports
        assert all_reports["uncertain_medial_meniscus"]["targets"]["Medial Meniscus"] == "uncertain"

        # Negation case
        assert "negation_with_intact" in all_reports
        assert all_reports["negation_with_intact"]["targets"]["ACL"] == "negative"

        # Medial/lateral confusion risk
        assert "medial_lateral_confusion_risk" in all_reports
        assert all_reports["medial_lateral_confusion_risk"]["targets"]["Lateral Meniscus"] == "positive"

        # Missing findings
        assert "missing_finding" in all_reports
        assert all_reports["missing_finding"]["targets"]["Lateral Meniscus"] == "not_mentioned"

        # Hedging and history
        assert "hedging_and_history" in all_reports

        # Spanish section
        assert "spanish_section" in all_reports

        # Double negation
        assert "double_negation" in all_reports


class TestReportAgentFrameworkIntegration:
    """Integration-level tests demonstrating how agents would work together."""

    def test_extraction_verification_must_have_matching_study_uid(self):
        """Agents should track study UIDs consistently."""
        study_uid = "1.2.826.synth.001"

        # Create extraction result
        extraction = ExtractionResult(
            study_uid=study_uid,
            timestamp="2026-10-29T12:00:00Z",
            model_id="claude-sonnet-5-5",
            prompt_version="extractor-0.1",
            language_detected="en",
            findings=(),
        )

        # Verification should reference same study
        verification = VerificationResult(
            study_uid=study_uid,  # Must match
            verification_timestamp="2026-10-29T12:00:01Z",
            verified_by_model="claude-sonnet-5-5",
            findings=(),
        )

        assert extraction.study_uid == verification.study_uid

    def test_adjudication_reconciles_agent_outputs(self):
        """Adjudication should reference both extraction and verification."""
        extraction_state = LabelState.UNCERTAIN
        verifier_status = VerificationStatus.UNSUPPORTED

        # Create adjudicated finding that reconciles disagreement
        af = AdjudicatedFinding(
            target="ACL",
            final_state=LabelState.UNCERTAIN,  # Preserve uncertainty
            consensus=AdjudicationConsensus.RESOLVED_VIA_ESCALATION,
            extractor_state=extraction_state,
            verifier_status=verifier_status,
            evidence_span=None,
            confidence=0.0,
            human_review_required=True,
            review_reason="extraction_unverified",
        )

        assert af.human_review_required is True
        assert af.final_state == LabelState.UNCERTAIN  # Never silently converts to NEGATIVE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
