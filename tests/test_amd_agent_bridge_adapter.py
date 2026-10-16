"""Tests for hackathon/amd/agent_bridge_adapter.py: the bridge result -> StudyInferenceResult
adapter that lets the bridged Kaggle checkpoint's output flow through the shared,
unmodified mri_core.agents evidence-review layer.
"""

import pytest

pytest.importorskip("torch")

from hackathon.amd.agent_bridge_adapter import BRIDGE_WARNING, _parse_fold, to_study_inference_result
from hackathon.amd.kaggle_mil_bridge import TARGET_COLUMNS
from mri_core.agents.contracts import EvidenceBundleError, QualityGateStatus, WorkflowStatus, build_evidence_bundle
from mri_core.agents.orchestrator import process_inference_result
from mri_core.rsna_inference_engine import STATUS_OK


def _probs():
    return {col: round(0.1 * (i % 7), 4) for i, col in enumerate(TARGET_COLUMNS)}


class TestParseFold:
    def test_matches_standard_filename(self):
        assert _parse_fold("fold0-knee-mil-best.ckpt") == 0
        assert _parse_fold("/x/y/fold3-knee-mil-best.ckpt") == 3

    def test_case_insensitive(self):
        assert _parse_fold("FOLD2-knee-mil-best.ckpt") == 2

    def test_no_match_returns_none_not_a_guess(self):
        assert _parse_fold("my-custom-checkpoint.ckpt") is None


class TestToStudyInferenceResult:
    def test_preserves_real_values(self):
        probs = _probs()
        result = to_study_inference_result(
            study_uid="1.2.3", probabilities=probs, latency_seconds=0.42,
            checkpoint_path="fold1-knee-mil-best.ckpt", checkpoint_sha256="abc123",
            disclaimer="bridge disclaimer text",
        )
        assert result.study_uid == "1.2.3"
        assert result.status == STATUS_OK
        assert result.probabilities == probs
        assert result.is_ensemble is False
        assert result.latency_seconds == pytest.approx(0.42)
        assert result.disclaimer == "bridge disclaimer text"
        assert BRIDGE_WARNING in result.warnings

    def test_provenance_carries_fold_and_sha256_and_marks_missing_fold_as_none(self):
        result_with_fold = to_study_inference_result(
            study_uid="s", probabilities=_probs(), latency_seconds=0.1,
            checkpoint_path="fold4-knee-mil-best.ckpt", checkpoint_sha256="deadbeef",
            disclaimer="d",
        )
        provenance = result_with_fold.model_provenance[0]
        assert provenance.checkpoint_sha256 == "deadbeef"
        assert provenance.fold == 4
        assert provenance.target_columns == tuple(TARGET_COLUMNS)

        result_without_fold = to_study_inference_result(
            study_uid="s", probabilities=_probs(), latency_seconds=0.1,
            checkpoint_path="custom.ckpt", checkpoint_sha256="x",
            disclaimer="d",
        )
        assert result_without_fold.model_provenance[0].fold is None

    def test_extra_warnings_are_appended_after_the_bridge_warning(self):
        result = to_study_inference_result(
            study_uid="s", probabilities=_probs(), latency_seconds=0.1,
            checkpoint_path="fold0-knee-mil-best.ckpt", checkpoint_sha256="x",
            disclaimer="d", extra_warnings=["custom note"],
        )
        assert result.warnings == [BRIDGE_WARNING, "custom note"]

    def test_compatible_with_shared_evidence_bundle(self):
        """The whole point of this adapter: build_evidence_bundle() must accept it
        without raising, with no change to contracts.py."""
        result = to_study_inference_result(
            study_uid="s", probabilities=_probs(), latency_seconds=0.1,
            checkpoint_path="fold0-knee-mil-best.ckpt", checkpoint_sha256="x",
            disclaimer="d",
        )
        bundle = build_evidence_bundle(result, quality_gate_status=QualityGateStatus.NOT_ASSESSED)
        assert bundle.inference_status == STATUS_OK
        assert bundle.model_count == 1
        assert bundle.probabilities is not None

    def test_full_orchestrator_pipeline_runs_unmodified(self):
        """specialist -> critic -> consensus -> review gate, end to end, on a bridged result."""
        result = to_study_inference_result(
            study_uid="s", probabilities=_probs(), latency_seconds=0.1,
            checkpoint_path="fold0-knee-mil-best.ckpt", checkpoint_sha256="x",
            disclaimer="d",
        )
        workflow = process_inference_result(result, quality_gate_status=QualityGateStatus.NOT_ASSESSED)
        assert len(workflow.specialist_summaries) == 4
        # NOT_ASSESSED quality + the bridge warning both route to MORE_EVIDENCE_REQUIRED --
        # a bridged result must never be reported as evidence-complete.
        assert workflow.consensus.workflow_status == WorkflowStatus.MORE_EVIDENCE_REQUIRED
        assert workflow.review_gate.state.value == "PENDING_REVIEW"

    def test_label_order_mismatch_is_rejected_not_silently_reordered(self):
        """If probabilities were ever out of order, build_evidence_bundle must still raise --
        this adapter does not try to paper over that upstream."""
        reordered = dict(reversed(list(_probs().items())))
        result = to_study_inference_result(
            study_uid="s", probabilities=reordered, latency_seconds=0.1,
            checkpoint_path="fold0-knee-mil-best.ckpt", checkpoint_sha256="x",
            disclaimer="d",
        )
        with pytest.raises(EvidenceBundleError):
            build_evidence_bundle(result)
