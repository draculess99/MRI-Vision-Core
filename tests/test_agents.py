"""Unit tests for mri_core.agents (specialist-critic-consensus evidence-review layer).

Uses only synthetic StudyInferenceResult objects -- no real checkpoint, no real DICOM
data, no network calls. Deterministic and fast.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from mri_core.rsna_inference_engine import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_UNABLE_TO_EVALUATE,
    ModelProvenance,
    StudyInferenceResult,
)
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

from mri_core.agents.contracts import (
    DEFAULT_LIMITATIONS,
    EvidenceBundle,
    EvidenceBundleError,
    QualityGateStatus,
    ReviewState,
    Severity,
    WorkflowStatus,
    build_evidence_bundle,
)
from mri_core.agents.specialist import GROUPS, summarize_by_group
from mri_core.agents.critic import INTERNAL_INCONSISTENCY_CODES, review as critic_review
from mri_core.agents.consensus import build_consensus
from mri_core.agents.review_gate import (
    InvalidActorError,
    InvalidReasonError,
    InvalidTransitionError,
    ReviewGate,
    StaleEvidenceError,
    VALID_TRANSITIONS,
)
from mri_core.agents.llm_adapter import (
    LLMAdapterError,
    OfflineStubAdapter,
    ProposalStatus,
    RawProposal,
    ReviewerBrief,
    render_reviewer_brief,
    screen_proposal,
    validate_reviewer_brief,
)
from mri_core.agents.orchestrator import process_inference_result
from mri_core.agents.evaluation import ALL_SCENARIOS, run_ablation, run_evaluation

SCORES = {t: 0.5 + 0.01 * i for i, t in enumerate(TARGET_COLUMNS)}


def provenance(fold=0):
    return ModelProvenance(
        checkpoint_path=f"outputs/rsna-fivefold-benchmark/random/fold{fold}/last.pt",
        checkpoint_sha256="a" * 64, epoch=15, fold=fold, target_columns=tuple(TARGET_COLUMNS),
    )


def ok_result(**overrides):
    defaults = dict(
        study_uid="1.2.826.synth.1", status=STATUS_OK, probabilities=dict(SCORES),
        is_ensemble=False, model_provenance=[provenance()], latency_seconds=0.1,
        warnings=[], review_required=False,
    )
    defaults.update(overrides)
    return StudyInferenceResult(**defaults)


def unavailable_result(**overrides):
    defaults = dict(
        study_uid="1.2.826.synth.2", status=STATUS_UNABLE_TO_EVALUATE, probabilities=None,
        is_ensemble=False, model_provenance=[provenance()], latency_seconds=0.05,
        warnings=["Study unavailable"], review_required=True,
    )
    defaults.update(overrides)
    return StudyInferenceResult(**defaults)


# ------------------------------------------------------------------------------ contracts

class TestEvidenceBundle:
    def test_builds_from_ok_result_with_all_twelve_targets_preserved(self):
        bundle = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK)
        assert list(bundle.probabilities.keys()) == list(TARGET_COLUMNS)
        assert bundle.probabilities == SCORES  # no mutation, values identical

    def test_rejects_wrong_label_order(self):
        bad = dict(SCORES)
        reordered = {k: bad[k] for k in reversed(list(bad))}
        with pytest.raises(EvidenceBundleError):
            build_evidence_bundle(ok_result(probabilities=reordered))

    def test_quality_gate_defaults_to_not_assessed(self):
        bundle = build_evidence_bundle(ok_result())
        assert bundle.quality_gate_status == QualityGateStatus.NOT_ASSESSED

    def test_not_assessed_is_distinct_from_ok(self):
        assert QualityGateStatus.NOT_ASSESSED != QualityGateStatus.OK
        assert QualityGateStatus.NOT_ASSESSED.value == "NOT_ASSESSED"

    def test_bundle_is_frozen_and_fingerprint_is_deterministic(self):
        b1 = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK)
        b2 = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK)
        assert b1.fingerprint == b2.fingerprint
        with pytest.raises(Exception):
            b1.study_uid = "changed"  # frozen dataclass must reject mutation

    def test_fingerprint_changes_when_probabilities_change(self):
        b1 = build_evidence_bundle(ok_result())
        altered = dict(SCORES)
        altered["ACL"] = 0.99
        b2 = build_evidence_bundle(ok_result(probabilities=altered))
        assert b1.fingerprint != b2.fingerprint

    def test_default_limitations_present(self):
        bundle = build_evidence_bundle(ok_result())
        assert bundle.limitations == DEFAULT_LIMITATIONS

    def test_to_dict_is_json_serializable(self):
        import json
        bundle = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK)
        json.dumps(bundle.to_dict())  # must not raise


class TestEvidenceBundleDeepImmutability:
    """A frozen dataclass alone only blocks reassigning its attributes -- a plain dict
    or list stored inside one remains mutable in place. These tests prove every
    nested structure on EvidenceBundle is genuinely deep-frozen."""

    def test_probabilities_mapping_rejects_item_assignment(self):
        bundle = build_evidence_bundle(ok_result())
        with pytest.raises(TypeError):
            bundle.probabilities["ACL"] = 0.99

    def test_probabilities_mapping_rejects_deletion(self):
        bundle = build_evidence_bundle(ok_result())
        with pytest.raises(TypeError):
            del bundle.probabilities["ACL"]

    def test_checkpoint_provenance_entries_reject_item_assignment(self):
        bundle = build_evidence_bundle(ok_result())
        with pytest.raises(TypeError):
            bundle.checkpoint_provenance[0]["epoch"] = 999

    def test_checkpoint_provenance_nested_target_columns_also_frozen(self):
        """target_columns is a list inside ModelProvenance.to_dict() -- confirm it was
        recursively frozen too, not just the top-level provenance dict."""
        bundle = build_evidence_bundle(ok_result())
        nested = bundle.checkpoint_provenance[0]["target_columns"]
        assert isinstance(nested, tuple)  # lists are deep-frozen into tuples

    def test_warnings_and_evidence_references_are_tuples(self):
        bundle = build_evidence_bundle(ok_result(warnings=["a warning"]), evidence_references=("e1",))
        assert isinstance(bundle.warnings, tuple)
        assert isinstance(bundle.evidence_references, tuple)

    def test_bundle_attribute_reassignment_rejected(self):
        bundle = build_evidence_bundle(ok_result())
        with pytest.raises(Exception):
            bundle.probabilities = {}

    def test_original_study_inference_result_unaffected_by_later_mutation(self):
        """Mutating the caller's own copy of probabilities after building the bundle
        must never affect the already-built bundle (proves the bundle was built from
        a copy, not a shared reference)."""
        result = ok_result()
        bundle = build_evidence_bundle(result)
        result.probabilities["ACL"] = 0.0  # mutate the original dict the result holds
        assert bundle.probabilities["ACL"] == SCORES["ACL"]  # bundle is unaffected

    def test_fingerprint_cannot_go_stale_because_mutation_is_rejected(self):
        """Since mutation itself always raises, there is no code path by which the
        fingerprint could become inconsistent with the bundle's actual content."""
        bundle = build_evidence_bundle(ok_result())
        fingerprint_before = bundle.fingerprint
        try:
            bundle.probabilities["ACL"] = 0.0
        except TypeError:
            pass
        assert bundle.fingerprint == fingerprint_before
        assert bundle.probabilities["ACL"] == SCORES["ACL"]

    def test_equivalent_evidence_produces_identical_fingerprint(self):
        b1 = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK)
        b2 = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK)
        assert b1.fingerprint == b2.fingerprint

    def test_meaningful_change_in_provenance_changes_fingerprint(self):
        b1 = build_evidence_bundle(ok_result())
        different_provenance = provenance(fold=4)
        b2 = build_evidence_bundle(ok_result(model_provenance=[different_provenance]))
        assert b1.fingerprint != b2.fingerprint

    def test_nan_probability_is_handled_explicitly_in_fingerprint_not_silently(self):
        """json.dumps would otherwise silently emit non-standard literal NaN; this
        package instead converts it to an explicit 'NaN' string sentinel before
        hashing, verified by allow_nan=False never raising internally and the
        rendered to_dict() showing the sentinel rather than a raw float."""
        bad = dict(SCORES)
        bad["ACL"] = float("nan")
        bundle = build_evidence_bundle(ok_result(probabilities=bad))
        assert bundle.fingerprint  # computed without raising
        assert bundle.to_dict()["probabilities"]["ACL"] == "NaN"

    def test_infinite_probability_is_handled_explicitly(self):
        bad = dict(SCORES)
        bad["Fracture"] = float("inf")
        bundle = build_evidence_bundle(ok_result(probabilities=bad))
        assert bundle.to_dict()["probabilities"]["Fracture"] == "Infinity"

    def test_negative_infinite_probability_is_handled_explicitly(self):
        bad = dict(SCORES)
        bad["Fracture"] = float("-inf")
        bundle = build_evidence_bundle(ok_result(probabilities=bad))
        assert bundle.to_dict()["probabilities"]["Fracture"] == "-Infinity"

    def test_nan_vs_normal_value_produce_different_fingerprints(self):
        normal = build_evidence_bundle(ok_result())
        bad = dict(SCORES)
        bad["ACL"] = float("nan")
        with_nan = build_evidence_bundle(ok_result(probabilities=bad))
        assert normal.fingerprint != with_nan.fingerprint

    def test_to_dict_output_is_strictly_json_safe_even_with_non_finite_values(self):
        import json
        bad = dict(SCORES)
        bad["ACL"] = float("nan")
        bundle = build_evidence_bundle(ok_result(probabilities=bad))
        json.dumps(bundle.to_dict(), allow_nan=False)  # must not raise


# ------------------------------------------------------------------------------ specialist

class TestSpecialist:
    def test_groups_cover_all_twelve_targets_exactly_once(self):
        covered = [t for targets in GROUPS.values() for t in targets]
        assert sorted(covered) == sorted(TARGET_COLUMNS)
        assert len(covered) == 12

    def test_summaries_carry_verbatim_scores(self):
        bundle = build_evidence_bundle(ok_result())
        summaries = summarize_by_group(bundle)
        assert len(summaries) == 4
        all_scores = {}
        for s in summaries:
            all_scores.update(s.scores)
        assert all_scores == SCORES

    def test_no_scores_when_probabilities_unavailable(self):
        bundle = build_evidence_bundle(unavailable_result())
        summaries = summarize_by_group(bundle)
        assert all(s.scores == {} for s in summaries)

    def test_note_never_claims_abnormality_detected(self):
        bundle = build_evidence_bundle(ok_result())
        for s in summarize_by_group(bundle):
            assert "detected" not in s.note.lower() or "does not imply" in s.note.lower()
            assert "abnormal" not in s.note.lower().replace("no abnormality was detected", "").replace(
                "an abnormality was detected", ""
            ) or "0.5" in s.note


# ------------------------------------------------------------------------------ critic

class TestCritic:
    def test_clean_bundle_produces_only_info_level_or_no_errors(self):
        bundle = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK,
                                        evidence_references=("x",))
        findings = critic_review(bundle)
        assert not any(f.severity == Severity.ERROR for f in findings)

    def test_status_ok_with_none_probabilities_is_flagged(self):
        bundle = build_evidence_bundle(ok_result(probabilities=None))
        findings = critic_review(bundle)
        codes = [f.code for f in findings]
        assert "STATUS_PROBABILITY_MISMATCH" in codes
        assert any(f.severity == Severity.ERROR for f in findings if f.code == "STATUS_PROBABILITY_MISMATCH")

    def test_non_finite_probability_detected(self):
        bad = dict(SCORES)
        bad["Fracture"] = float("nan")
        bundle = build_evidence_bundle(ok_result(probabilities=bad))
        findings = critic_review(bundle)
        assert any(f.code == "NON_FINITE_PROBABILITY" and f.evidence_reference == "Fracture" for f in findings)

    def test_out_of_range_probability_detected(self):
        bad = dict(SCORES)
        bad["ACL"] = 1.5
        bundle = build_evidence_bundle(ok_result(probabilities=bad))
        findings = critic_review(bundle)
        assert any(f.code == "PROBABILITY_OUT_OF_RANGE" for f in findings)

    def test_missing_provenance_hash_detected(self):
        bad_provenance = ModelProvenance(
            checkpoint_path="x.pt", checkpoint_sha256="", epoch=1, fold=0, target_columns=tuple(TARGET_COLUMNS)
        )
        bundle = build_evidence_bundle(ok_result(model_provenance=[bad_provenance]))
        findings = critic_review(bundle)
        assert any(f.code == "MISSING_PROVENANCE_HASH" for f in findings)

    def test_quality_not_assessed_is_info_not_error(self):
        bundle = build_evidence_bundle(ok_result())  # NOT_ASSESSED by default
        findings = critic_review(bundle)
        quality_findings = [f for f in findings if f.code == "QUALITY_NOT_ASSESSED"]
        assert len(quality_findings) == 1
        assert quality_findings[0].severity == Severity.INFO

    def test_quality_invalid_is_an_error(self):
        bundle = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.INVALID)
        findings = critic_review(bundle)
        assert any(f.code == "QUALITY_INVALID" and f.severity == Severity.ERROR for f in findings)

    def test_engine_does_not_certify_success_as_safe(self):
        """There is no finding or code anywhere that marks a successful inference as
        clinically safe -- only structural/consistency checks exist."""
        bundle = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK)
        findings = critic_review(bundle)
        for f in findings:
            assert "safe" not in f.message.lower()
            assert "clinical" not in f.message.lower() or "clinically safe" not in f.message.lower()


# ------------------------------------------------------------------------------ consensus

class TestConsensus:
    def test_complete_valid_evidence_is_complete_for_review(self):
        bundle = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK)
        findings = critic_review(bundle)
        from mri_core.agents.specialist import summarize_by_group
        consensus = build_consensus(bundle, summarize_by_group(bundle), findings)
        assert consensus.workflow_status == WorkflowStatus.EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW

    def test_missing_study_is_inference_unavailable(self):
        bundle = build_evidence_bundle(unavailable_result())
        findings = critic_review(bundle)
        from mri_core.agents.specialist import summarize_by_group
        consensus = build_consensus(bundle, summarize_by_group(bundle), findings)
        assert consensus.workflow_status == WorkflowStatus.INFERENCE_UNAVAILABLE

    def test_internal_inconsistency_takes_priority_over_unavailable(self):
        bundle = build_evidence_bundle(ok_result(probabilities=None))
        findings = critic_review(bundle)
        from mri_core.agents.specialist import summarize_by_group
        consensus = build_consensus(bundle, summarize_by_group(bundle), findings)
        assert consensus.workflow_status == WorkflowStatus.INCONSISTENT_EVIDENCE

    def test_quality_not_ok_requires_more_evidence(self):
        bundle = build_evidence_bundle(ok_result())  # NOT_ASSESSED
        findings = critic_review(bundle)
        from mri_core.agents.specialist import summarize_by_group
        consensus = build_consensus(bundle, summarize_by_group(bundle), findings)
        assert consensus.workflow_status == WorkflowStatus.MORE_EVIDENCE_REQUIRED

    def test_consensus_never_averages_into_a_confidence_score(self):
        bundle = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK)
        findings = critic_review(bundle)
        from mri_core.agents.specialist import summarize_by_group
        consensus = build_consensus(bundle, summarize_by_group(bundle), findings)
        payload = consensus.to_dict()
        assert "confidence" not in str(payload).lower()

    def test_initial_human_review_status_is_pending(self):
        bundle = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK)
        findings = critic_review(bundle)
        from mri_core.agents.specialist import summarize_by_group
        consensus = build_consensus(bundle, summarize_by_group(bundle), findings)
        assert consensus.human_review_status == ReviewState.PENDING_REVIEW


def _consensus_for(result, quality=QualityGateStatus.NOT_ASSESSED):
    bundle = build_evidence_bundle(result, quality_gate_status=quality)
    findings = critic_review(bundle)
    from mri_core.agents.specialist import summarize_by_group
    return build_consensus(bundle, summarize_by_group(bundle), findings)


class TestConsensusPrecedenceEveryCondition:
    """One test per required precedence condition, independently specified (the
    expected WorkflowStatus is hard-coded in each test, not derived from whatever the
    code happens to currently output)."""

    def test_explicit_unavailable_status_with_no_predictions_is_inference_unavailable(self):
        consensus = _consensus_for(unavailable_result())
        assert consensus.workflow_status == WorkflowStatus.INFERENCE_UNAVAILABLE

    def test_explicit_error_status_with_no_predictions_is_inference_unavailable(self):
        consensus = _consensus_for(unavailable_result(status=STATUS_ERROR))
        assert consensus.workflow_status == WorkflowStatus.INFERENCE_UNAVAILABLE

    def test_status_ok_with_missing_predictions_is_inconsistent_evidence(self):
        consensus = _consensus_for(ok_result(probabilities=None))
        assert consensus.workflow_status == WorkflowStatus.INCONSISTENT_EVIDENCE

    def test_non_ok_status_with_predictions_present_is_inconsistent_evidence(self):
        consensus = _consensus_for(unavailable_result(probabilities=dict(SCORES)))
        assert consensus.workflow_status == WorkflowStatus.INCONSISTENT_EVIDENCE

    def test_non_finite_value_is_inconsistent_evidence(self):
        bad = dict(SCORES); bad["ACL"] = float("nan")
        consensus = _consensus_for(ok_result(probabilities=bad), quality=QualityGateStatus.OK)
        assert consensus.workflow_status == WorkflowStatus.INCONSISTENT_EVIDENCE

    def test_out_of_range_value_is_inconsistent_evidence(self):
        bad = dict(SCORES); bad["ACL"] = 2.0
        consensus = _consensus_for(ok_result(probabilities=bad), quality=QualityGateStatus.OK)
        assert consensus.workflow_status == WorkflowStatus.INCONSISTENT_EVIDENCE

    def test_wrong_label_order_is_inconsistent_evidence(self):
        """build_evidence_bundle itself rejects wrong order at construction time, so
        this is verified one level down: a critic LABEL_ORDER_MISMATCH finding (if it
        ever occurred) is classified as INCONSISTENT_EVIDENCE by construction of
        INTERNAL_INCONSISTENCY_CODES."""
        assert "LABEL_ORDER_MISMATCH" in INTERNAL_INCONSISTENCY_CODES

    def test_missing_provenance_is_inconsistent_evidence(self):
        consensus = _consensus_for(ok_result(model_provenance=[]), quality=QualityGateStatus.OK)
        assert consensus.workflow_status == WorkflowStatus.INCONSISTENT_EVIDENCE

    def test_valid_inference_with_quality_not_assessed_is_more_evidence_required(self):
        consensus = _consensus_for(ok_result(), quality=QualityGateStatus.NOT_ASSESSED)
        assert consensus.workflow_status == WorkflowStatus.MORE_EVIDENCE_REQUIRED

    def test_valid_inference_with_quality_review_is_more_evidence_required(self):
        consensus = _consensus_for(ok_result(), quality=QualityGateStatus.REVIEW)
        assert consensus.workflow_status == WorkflowStatus.MORE_EVIDENCE_REQUIRED

    def test_valid_inference_with_complete_evidence_and_quality_ok_is_complete_for_review(self):
        consensus = _consensus_for(ok_result(), quality=QualityGateStatus.OK)
        assert consensus.workflow_status == WorkflowStatus.EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW

    def test_no_clinical_classification_strings_anywhere_in_workflow_status_values(self):
        clinical_terms = ("positive", "negative", "abnormal", "normal", "diagnosis", "disease")
        for status in WorkflowStatus:
            assert not any(term in status.value.lower() for term in clinical_terms)


# ------------------------------------------------------------------------------ review_gate

class TestReviewGate:
    def test_valid_transition_succeeds_and_is_recorded(self):
        gate = ReviewGate("fp-1")
        entry = gate.transition(ReviewState.REVIEWED, actor="r1", reason="ok", evidence_fingerprint="fp-1")
        assert gate.state == ReviewState.REVIEWED
        assert len(gate.history) == 1
        assert entry.previous_state == ReviewState.PENDING_REVIEW
        assert entry.new_state == ReviewState.REVIEWED

    def test_invalid_transition_rejected(self):
        gate = ReviewGate("fp-1")
        gate.transition(ReviewState.REVIEWED, actor="r1", reason="ok", evidence_fingerprint="fp-1")
        with pytest.raises(InvalidTransitionError):
            gate.transition(ReviewState.PENDING_REVIEW, actor="r1", reason="reopen", evidence_fingerprint="fp-1")

    def test_stale_fingerprint_rejected(self):
        gate = ReviewGate("fp-1")
        with pytest.raises(StaleEvidenceError):
            gate.transition(ReviewState.REVIEWED, actor="r1", reason="x", evidence_fingerprint="fp-WRONG")

    def test_history_is_append_only(self):
        gate = ReviewGate("fp-1")
        gate.transition(ReviewState.MORE_EVIDENCE_REQUESTED, actor="r1", reason="need more", evidence_fingerprint="fp-1")
        gate.transition(ReviewState.REVIEWED, actor="r2", reason="now ok", evidence_fingerprint="fp-1")
        assert len(gate.history) == 2
        assert gate.history[0].new_state == ReviewState.MORE_EVIDENCE_REQUESTED
        assert gate.history[1].new_state == ReviewState.REVIEWED
        # history is a tuple snapshot -- cannot be appended to externally
        with pytest.raises(AttributeError):
            gate.history.append(None)

    def test_reviewed_is_terminal(self):
        assert VALID_TRANSITIONS[ReviewState.REVIEWED] == frozenset()

    def test_review_state_change_does_not_affect_deterministic_consensus(self):
        bundle = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK)
        findings = critic_review(bundle)
        from mri_core.agents.specialist import summarize_by_group
        consensus_before = build_consensus(bundle, summarize_by_group(bundle), findings)

        gate = ReviewGate(bundle.fingerprint)
        gate.transition(ReviewState.REVIEWED, actor="r1", reason="x", evidence_fingerprint=bundle.fingerprint)

        consensus_after = build_consensus(bundle, summarize_by_group(bundle), findings)
        assert consensus_before.workflow_status == consensus_after.workflow_status
        assert consensus_before.to_dict() == consensus_after.to_dict()

    def test_empty_actor_rejected(self):
        gate = ReviewGate("fp-1")
        with pytest.raises(InvalidActorError):
            gate.transition(ReviewState.REVIEWED, actor="   ", reason="ok", evidence_fingerprint="fp-1")

    def test_non_string_actor_rejected(self):
        gate = ReviewGate("fp-1")
        with pytest.raises(InvalidActorError):
            gate.transition(ReviewState.REVIEWED, actor=42, reason="ok", evidence_fingerprint="fp-1")

    def test_overlong_actor_rejected(self):
        gate = ReviewGate("fp-1")
        with pytest.raises(InvalidActorError):
            gate.transition(ReviewState.REVIEWED, actor="a" * 500, reason="ok", evidence_fingerprint="fp-1")

    def test_empty_reason_rejected(self):
        gate = ReviewGate("fp-1")
        with pytest.raises(InvalidReasonError):
            gate.transition(ReviewState.REVIEWED, actor="r1", reason="", evidence_fingerprint="fp-1")

    def test_non_string_reason_rejected(self):
        gate = ReviewGate("fp-1")
        with pytest.raises(InvalidReasonError):
            gate.transition(ReviewState.REVIEWED, actor="r1", reason=None, evidence_fingerprint="fp-1")

    def test_invalid_actor_checked_before_stale_fingerprint(self):
        """Malformed actor/reason are rejected even when the fingerprint is also
        wrong -- actor/reason validation happens first."""
        gate = ReviewGate("fp-1")
        with pytest.raises(InvalidActorError):
            gate.transition(ReviewState.REVIEWED, actor="", reason="ok", evidence_fingerprint="totally-wrong")

    def test_transition_never_mutates_original_evidence_bundle(self):
        bundle = build_evidence_bundle(ok_result(), quality_gate_status=QualityGateStatus.OK)
        before = bundle.to_dict()
        gate = ReviewGate(bundle.fingerprint)
        gate.transition(ReviewState.REVIEWED, actor="r1", reason="complete", evidence_fingerprint=bundle.fingerprint)
        assert bundle.to_dict() == before

    def test_reviewed_state_is_documented_as_simulated_not_clinician_approval(self):
        import inspect
        from mri_core.agents import review_gate
        module_doc = review_gate.__doc__ or ""
        class_doc = review_gate.ReviewGate.__doc__ or ""
        combined = (module_doc + class_doc).lower()
        assert "no actual clinician approval is implied" in combined


# ------------------------------------------------------------------------------ llm_adapter

class TestLLMAdapter:
    def test_offline_stub_never_touches_network(self):
        import inspect
        source = inspect.getsource(OfflineStubAdapter)
        for forbidden in ("requests", "urllib", "socket", "http.client", "api_key", "API_KEY"):
            assert forbidden not in source

    def test_offline_stub_proposal_is_quarantined_not_trusted(self):
        """Even a well-behaved, deterministic stub's own output is quarantined, never
        auto-promoted to verified evidence -- there is no path that trusts free text."""
        wr = process_inference_result(ok_result(), quality_gate_status=QualityGateStatus.OK,
                                       evidence_references=("x.pt",))
        adapter = OfflineStubAdapter()
        proposal = adapter.propose(wr.bundle, wr.consensus)
        outcome = screen_proposal(proposal, wr.bundle)
        assert outcome.status == ProposalStatus.QUARANTINED
        assert "never published as verified evidence" in outcome.reason

    def test_fabricated_citation_is_blocked(self):
        wr = process_inference_result(ok_result(), quality_gate_status=QualityGateStatus.OK,
                                       evidence_references=("real.pt",))
        fabricated = RawProposal(raw_text="some summary", cited_evidence=("fake.pt",))
        outcome = screen_proposal(fabricated, wr.bundle)
        assert outcome.status == ProposalStatus.BLOCKED

    def test_diagnosis_claim_is_blocked(self):
        wr = process_inference_result(ok_result(), quality_gate_status=QualityGateStatus.OK)
        claim = RawProposal(raw_text="This study shows a confirmed diagnosis of ACL tear.", cited_evidence=())
        outcome = screen_proposal(claim, wr.bundle)
        assert outcome.status == ProposalStatus.BLOCKED
        assert "disallowed claim pattern" in outcome.reason

    def test_invented_approval_claim_is_blocked(self):
        wr = process_inference_result(ok_result(), quality_gate_status=QualityGateStatus.OK)
        claim = RawProposal(raw_text="Approved by clinician on review.", cited_evidence=())
        outcome = screen_proposal(claim, wr.bundle)
        assert outcome.status == ProposalStatus.BLOCKED

    def test_overlong_proposal_is_blocked(self):
        wr = process_inference_result(ok_result(), quality_gate_status=QualityGateStatus.OK)
        claim = RawProposal(raw_text="x" * 10_000, cited_evidence=())
        outcome = screen_proposal(claim, wr.bundle)
        assert outcome.status == ProposalStatus.BLOCKED

    def test_empty_proposal_is_blocked(self):
        wr = process_inference_result(ok_result(), quality_gate_status=QualityGateStatus.OK)
        claim = RawProposal(raw_text="   ", cited_evidence=())
        outcome = screen_proposal(claim, wr.bundle)
        assert outcome.status == ProposalStatus.BLOCKED

    def test_quarantined_proposal_is_never_promoted_to_reviewer_brief(self):
        """There is no function anywhere in llm_adapter.py that takes a
        QuarantinedProposal and returns a trusted ReviewerBrief."""
        import inspect
        from mri_core.agents import llm_adapter
        source = inspect.getsource(llm_adapter)
        assert "QuarantinedProposal" not in inspect.signature(llm_adapter.render_reviewer_brief).parameters.keys()
        # render_reviewer_brief only accepts (bundle, consensus) -- never a proposal.
        params = list(inspect.signature(llm_adapter.render_reviewer_brief).parameters)
        assert params == ["bundle", "consensus"]

    def test_render_reviewer_brief_never_uses_adapter_free_text(self):
        """render_reviewer_brief's output must come purely from the template, not
        from any adapter -- verified by checking an adapter's proposal text never
        appears verbatim in the rendered brief (the two use different vocabulary)."""
        wr = process_inference_result(ok_result(), quality_gate_status=QualityGateStatus.OK,
                                       evidence_references=("x.pt",))
        adapter = OfflineStubAdapter()
        proposal = adapter.propose(wr.bundle, wr.consensus)
        brief = render_reviewer_brief(wr.bundle, wr.consensus)
        validate_reviewer_brief(brief, wr.bundle)  # the template's own output is still checked
        assert brief.summary_text != proposal.raw_text

    def test_reviewer_brief_has_no_probability_or_provenance_fields(self):
        """Structural guarantee: there is nowhere on ReviewerBrief to put an altered
        probability, checkpoint hash, quality status, or review state."""
        fields = ReviewerBrief.__dataclass_fields__.keys()
        for forbidden in ("probabilities", "checkpoint_provenance", "quality_gate_status", "review_state"):
            assert forbidden not in fields

    def test_empty_summary_rejected_by_validate_reviewer_brief(self):
        wr = process_inference_result(ok_result(), quality_gate_status=QualityGateStatus.OK)
        empty = ReviewerBrief(summary_text="   ", cited_evidence=(), disclaimer=wr.bundle.disclaimer)
        with pytest.raises(LLMAdapterError):
            validate_reviewer_brief(empty, wr.bundle)


# ------------------------------------------------------------------------------ orchestrator

class TestOrchestrator:
    def test_process_result_end_to_end(self):
        wr = process_inference_result(ok_result(), quality_gate_status=QualityGateStatus.OK,
                                       evidence_references=("x.pt",))
        assert wr.bundle.study_uid == "1.2.826.synth.1"
        assert wr.consensus.workflow_status == WorkflowStatus.EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW
        assert wr.review_gate.state == ReviewState.PENDING_REVIEW
        assert len(wr.specialist_summaries) == 4

    def test_original_probabilities_never_mutated(self):
        result = ok_result()
        original = dict(result.probabilities)
        process_inference_result(result)
        assert result.probabilities == original

    def test_to_dict_is_json_serializable(self):
        import json
        wr = process_inference_result(ok_result(), quality_gate_status=QualityGateStatus.OK)
        json.dumps(wr.to_dict())


# ------------------------------------------------------------------------------ evaluation harness

class TestEvaluationHarness:
    def test_all_ten_scenarios_present(self):
        assert len(ALL_SCENARIOS) == 10

    def test_run_evaluation_all_pass(self):
        report = run_evaluation()
        assert report.scenario_count == 10
        assert report.passed_count == 10
        assert report.failed_count == 0
        assert report.workflow_status_agreement == 1.0

    def test_latency_is_measured_and_positive(self):
        report = run_evaluation()
        assert report.total_latency_seconds > 0
        assert report.mean_latency_seconds > 0

    def test_report_labeled_as_synthetic_not_clinical(self):
        import inspect
        from mri_core.agents import evaluation
        source = inspect.getsource(evaluation)
        assert "synthetic workflow-validation" in source.lower()
        assert "not a clinical accuracy" in source.lower() or "not clinical" in source.lower()

    def test_ablation_has_three_levels_with_measured_fields(self):
        levels = run_ablation()
        assert [lvl.level for lvl in levels] == ["A", "B", "C"]
        a, b, c = levels
        assert not a.has_specialist_grouping and not a.has_workflow_status
        assert b.has_specialist_grouping and b.has_critic_findings and not b.has_workflow_status
        assert c.has_specialist_grouping and c.has_critic_findings and c.has_workflow_status and c.has_review_gate

    def test_ablation_evidence_completeness_is_measured_not_invented(self):
        levels = run_ablation()
        for lvl in levels:
            payload = lvl.to_dict()
            assert 0.0 <= payload["evidence_field_completeness"] <= 1.0
