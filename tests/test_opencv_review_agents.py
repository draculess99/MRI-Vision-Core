"""Deterministic tests for Evidence Review Agent.

All tests are offline, non-diagnostic, and verify that the agent correctly
applies deterministic rules to manifest and ChangeGuard inputs.
"""

import json
import pytest
from hackathon.opencv_aws.review_contracts import (
    ReviewOutcome,
    AgentDecision,
    ReviewAction,
)
from hackathon.opencv_aws.evidence_review_agent import review_evidence


@pytest.fixture
def valid_manifest():
    """Minimal valid manifest."""
    return {
        "source_hash_sha256": "abc123def456" + "0" * 40,
        "processing_steps": [
            {"step": "normalize_to_uint8", "deterministic": True},
            {"step": "enhance_contrast", "deterministic": True},
            {"step": "detect_edges", "deterministic": True},
            {"step": "create_overlay", "deterministic": True},
        ],
        "generated_artifacts": ["original", "enhanced", "edges", "overlay"],
        "input_width": 512,
        "input_height": 512,
        "input_dtype": "uint8",
    }


@pytest.fixture
def stable_comparison():
    """STABLE baseline comparison result."""
    return {
        "status": "STABLE",
        "foreground_delta": 8.5,
        "mean_intensity_delta": 12.0,
        "reason": None,
    }


@pytest.fixture
def review_comparison():
    """REVIEW baseline comparison result (changed significantly)."""
    return {
        "status": "REVIEW",
        "foreground_delta": 25.0,
        "mean_intensity_delta": 35.0,
        "reason": "Thresholds exceeded: baseline changed significantly",
    }


@pytest.fixture
def no_baseline_comparison():
    """First-run no-baseline condition."""
    return {
        "status": "INVALID",
        "foreground_delta": None,
        "mean_intensity_delta": None,
        "reason": "No baseline image available for comparison",
    }


class TestFinalizeEvidenceBundle:
    """Tests for FINALIZE_EVIDENCE_BUNDLE outcome."""

    def test_stable_baseline_finalizes(self, valid_manifest, stable_comparison):
        """Valid manifest + STABLE comparison → FINALIZE."""
        decision = review_evidence(valid_manifest, stable_comparison)

        assert decision.outcome == ReviewOutcome.FINALIZE_EVIDENCE_BUNDLE
        assert decision.human_approval_required is False
        assert "deterministically" in decision.reasoning.lower()
        assert decision.next_action.action_type == "finalize"

    def test_first_run_no_baseline_finalizes(self, valid_manifest, no_baseline_comparison):
        """Valid manifest + no-baseline condition → FINALIZE (recognized first-run)."""
        decision = review_evidence(valid_manifest, no_baseline_comparison)

        assert decision.outcome == ReviewOutcome.FINALIZE_EVIDENCE_BUNDLE
        assert decision.human_approval_required is False

    def test_safety_notice_in_finalize(self, valid_manifest, stable_comparison):
        """FINALIZE outcome includes safety notice."""
        decision = review_evidence(valid_manifest, stable_comparison)

        assert "research" in decision.safety_notice.lower()
        assert "non-diagnostic" in decision.safety_notice.lower()

    def test_finalize_decision_json_serializable(self, valid_manifest, stable_comparison):
        """FINALIZE decision converts to JSON without error."""
        decision = review_evidence(valid_manifest, stable_comparison)

        json_str = decision.to_json_str()
        assert isinstance(json_str, str)

        # Verify it deserializes back
        data = json.loads(json_str)
        assert data["outcome"] == "FINALIZE_EVIDENCE_BUNDLE"
        assert "next_action" in data
        assert "trace" in data


class TestRequestMoreEvidence:
    """Tests for REQUEST_MORE_EVIDENCE outcome."""

    def test_review_status_requests_evidence(self, valid_manifest, review_comparison):
        """REVIEW comparison → REQUEST_MORE_EVIDENCE."""
        decision = review_evidence(valid_manifest, review_comparison)

        assert decision.outcome == ReviewOutcome.REQUEST_MORE_EVIDENCE
        assert decision.human_approval_required is False
        assert decision.next_action.action_type == "request_input"
        assert decision.next_action.is_recoverable is True

    def test_missing_artifacts_requests_evidence(self, valid_manifest):
        """Missing artifacts → REQUEST_MORE_EVIDENCE."""
        manifest = valid_manifest.copy()
        manifest["generated_artifacts"] = ["original", "enhanced"]  # Missing edges, overlay

        decision = review_evidence(manifest, None)

        assert decision.outcome == ReviewOutcome.REQUEST_MORE_EVIDENCE

    def test_non_deterministic_step_requests_evidence(self, valid_manifest, stable_comparison):
        """Non-deterministic processing step → REQUEST_MORE_EVIDENCE."""
        manifest = valid_manifest.copy()
        steps = manifest["processing_steps"].copy()
        steps[0] = {"step": "enhance_contrast", "deterministic": False}
        manifest["processing_steps"] = steps

        decision = review_evidence(manifest, stable_comparison)

        assert decision.outcome == ReviewOutcome.REQUEST_MORE_EVIDENCE

    def test_request_more_evidence_has_safety_notice(self, valid_manifest, review_comparison):
        """REQUEST_MORE_EVIDENCE includes safety notice."""
        decision = review_evidence(valid_manifest, review_comparison)

        assert "research" in decision.safety_notice.lower()
        assert "non-diagnostic" in decision.safety_notice.lower()

    def test_request_more_evidence_json_serializable(self, valid_manifest, review_comparison):
        """REQUEST_MORE_EVIDENCE decision converts to JSON."""
        decision = review_evidence(valid_manifest, review_comparison)

        json_str = decision.to_json_str()
        data = json.loads(json_str)
        assert data["outcome"] == "REQUEST_MORE_EVIDENCE"
        assert data["human_approval_required"] is False


class TestPendingHumanApproval:
    """Tests for PENDING_HUMAN_APPROVAL outcome."""

    def test_malformed_metadata_requires_approval(self):
        """Missing manifest fields → PENDING_HUMAN_APPROVAL."""
        manifest = {}  # Empty manifest

        decision = review_evidence(manifest, None)

        assert decision.outcome == ReviewOutcome.PENDING_HUMAN_APPROVAL
        assert decision.human_approval_required is True
        assert decision.next_action.action_type == "halt_pending_approval"

    def test_invalid_width_requires_approval(self, valid_manifest):
        """Width = 0 → PENDING_HUMAN_APPROVAL."""
        manifest = valid_manifest.copy()
        manifest["input_width"] = 0

        decision = review_evidence(manifest, None)

        assert decision.outcome == ReviewOutcome.PENDING_HUMAN_APPROVAL
        assert decision.human_approval_required is True

    def test_invalid_height_requires_approval(self, valid_manifest):
        """Height > 10000 → PENDING_HUMAN_APPROVAL."""
        manifest = valid_manifest.copy()
        manifest["input_height"] = 20000

        decision = review_evidence(manifest, None)

        assert decision.outcome == ReviewOutcome.PENDING_HUMAN_APPROVAL
        assert decision.human_approval_required is True

    def test_unexpected_dtype_requires_approval(self, valid_manifest):
        """Unexpected dtype → PENDING_HUMAN_APPROVAL."""
        manifest = valid_manifest.copy()
        manifest["input_dtype"] = "int16"  # Unexpected

        decision = review_evidence(manifest, None)

        assert decision.outcome == ReviewOutcome.PENDING_HUMAN_APPROVAL
        assert decision.human_approval_required is True

    def test_pending_approval_has_safety_notice(self, valid_manifest):
        """PENDING_HUMAN_APPROVAL includes safety notice."""
        manifest = valid_manifest.copy()
        manifest["input_width"] = -1

        decision = review_evidence(manifest, None)

        assert "research" in decision.safety_notice.lower()
        assert "non-diagnostic" in decision.safety_notice.lower()

    def test_pending_approval_cannot_be_finalized(self, valid_manifest):
        """PENDING_HUMAN_APPROVAL sets human_approval_required=True."""
        manifest = valid_manifest.copy()
        manifest["input_width"] = -1  # Invalid dimension (unrecoverable)

        decision = review_evidence(manifest, None)

        assert decision.human_approval_required is True
        assert decision.outcome == ReviewOutcome.PENDING_HUMAN_APPROVAL
        # Verify that outcome is definitely PENDING, not FINALIZE
        assert decision.outcome != ReviewOutcome.FINALIZE_EVIDENCE_BUNDLE


class TestDeterminism:
    """Tests for deterministic reproducibility."""

    def test_same_inputs_produce_same_decision(self, valid_manifest, stable_comparison):
        """Identical inputs → byte-for-byte identical JSON trace."""
        decision1 = review_evidence(valid_manifest, stable_comparison)
        decision2 = review_evidence(valid_manifest, stable_comparison)

        json1 = decision1.to_json_str()
        json2 = decision2.to_json_str()

        assert json1 == json2

    def test_same_inputs_produce_same_outcome(self, valid_manifest, review_comparison):
        """Multiple calls with same data → same outcome."""
        outcomes = [
            review_evidence(valid_manifest, review_comparison).outcome
            for _ in range(5)
        ]

        assert len(set(outcomes)) == 1

    def test_trace_records_all_three_tool_steps(self, valid_manifest, stable_comparison):
        """Trace includes all three tool steps in order."""
        decision = review_evidence(valid_manifest, stable_comparison)
        trace = decision.trace

        assert len(trace.tool_steps) == 3
        assert trace.tool_steps[0].tool_name == "validate_manifest"
        assert trace.tool_steps[1].tool_name == "assess_change_guard"
        assert trace.tool_steps[2].tool_name == "choose_next_action"


class TestSafetyNotice:
    """Tests for safety notice presence in all outcomes."""

    def test_finalize_has_safety_notice(self, valid_manifest, stable_comparison):
        """FINALIZE includes safety_notice field."""
        decision = review_evidence(valid_manifest, stable_comparison)

        assert decision.safety_notice
        assert "research" in decision.safety_notice.lower()

    def test_request_has_safety_notice(self, valid_manifest, review_comparison):
        """REQUEST has safety_notice field."""
        decision = review_evidence(valid_manifest, review_comparison)

        assert decision.safety_notice
        assert "research" in decision.safety_notice.lower()

    def test_pending_has_safety_notice(self):
        """PENDING has safety_notice field."""
        decision = review_evidence({}, None)

        assert decision.safety_notice
        assert "research" in decision.safety_notice.lower()

    def test_safety_notice_never_makes_medical_claims(
        self, valid_manifest, stable_comparison
    ):
        """Safety notice never contains medical claims."""
        decision = review_evidence(valid_manifest, stable_comparison)

        # Check it doesn't have dangerous patterns (allowing "non-diagnostic")
        lowered = decision.safety_notice.lower()
        assert "this is a diagnosis" not in lowered
        assert "has pathology" not in lowered
        assert "patient has" not in lowered
        assert "ruled out" not in lowered
        # It should explicitly say non-diagnostic or research-only
        assert "non-diagnostic" in lowered or "research" in lowered


class TestTraceStructure:
    """Tests for audit trail structure."""

    def test_trace_includes_source_hash(self, valid_manifest, stable_comparison):
        """Trace includes source_hash_sha256 from manifest."""
        decision = review_evidence(valid_manifest, stable_comparison)

        assert decision.trace.source_hash_sha256 == valid_manifest["source_hash_sha256"]

    def test_trace_includes_observations(self, valid_manifest, stable_comparison):
        """Trace includes observations dict."""
        decision = review_evidence(valid_manifest, stable_comparison)

        assert "manifest_status" in decision.trace.observations
        assert "comparison_status" in decision.trace.observations

    def test_trace_version_set(self, valid_manifest, stable_comparison):
        """Trace has trace_version."""
        decision = review_evidence(valid_manifest, stable_comparison)

        assert decision.trace.trace_version == "1.0"

    def test_trace_json_serializable(self, valid_manifest, stable_comparison):
        """Trace converts to valid JSON."""
        decision = review_evidence(valid_manifest, stable_comparison)

        trace_dict = decision.trace.to_dict()
        json_str = json.dumps(trace_dict)
        data = json.loads(json_str)

        assert "source_hash_sha256" in data
        assert "tool_steps" in data
        assert len(data["tool_steps"]) == 3


class TestNextAction:
    """Tests for action prescription."""

    def test_finalize_action_type(self, valid_manifest, stable_comparison):
        """FINALIZE → action_type='finalize'."""
        decision = review_evidence(valid_manifest, stable_comparison)

        assert decision.next_action.action_type == "finalize"

    def test_request_action_type(self, valid_manifest, review_comparison):
        """REQUEST → action_type='request_input'."""
        decision = review_evidence(valid_manifest, review_comparison)

        assert decision.next_action.action_type == "request_input"

    def test_pending_action_type(self):
        """PENDING → action_type='halt_pending_approval'."""
        decision = review_evidence({}, None)

        assert decision.next_action.action_type == "halt_pending_approval"

    def test_action_includes_user_facing_text(self, valid_manifest, stable_comparison):
        """Every action includes user_facing_text."""
        decision = review_evidence(valid_manifest, stable_comparison)

        assert decision.next_action.user_facing_text
        assert isinstance(decision.next_action.user_facing_text, str)
        assert len(decision.next_action.user_facing_text) > 0

    def test_action_includes_is_recoverable_flag(self, valid_manifest, stable_comparison):
        """Every action has is_recoverable bool."""
        decision = review_evidence(valid_manifest, stable_comparison)

        assert isinstance(decision.next_action.is_recoverable, bool)

    def test_request_action_is_recoverable(self, valid_manifest, review_comparison):
        """REQUEST action is_recoverable=True."""
        decision = review_evidence(valid_manifest, review_comparison)

        assert decision.next_action.is_recoverable is True

    def test_pending_action_not_recoverable(self):
        """PENDING action is_recoverable=False."""
        decision = review_evidence({}, None)

        assert decision.next_action.is_recoverable is False
