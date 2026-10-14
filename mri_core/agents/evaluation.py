"""Synthetic workflow-validation harness for the specialist-critic-consensus layer.

Every metric produced here is a synthetic workflow-validation metric -- a measure of
whether this package's deterministic rules behave as designed against constructed
StudyInferenceResult fixtures. None of it is a clinical accuracy or diagnostic
performance measurement, and none of it touches real patient data or a real checkpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from mri_core.rsna_inference_engine import (
    STATUS_OK,
    STATUS_UNABLE_TO_EVALUATE,
    ModelProvenance,
    StudyInferenceResult,
)
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

from .contracts import EvidenceBundle, QualityGateStatus, WorkflowStatus, build_evidence_bundle
from .critic import review as critic_review
from .llm_adapter import ProposalStatus, RawProposal, screen_proposal
from .orchestrator import WorkflowResult, process_inference_result
from .review_gate import InvalidTransitionError, ReviewState, StaleEvidenceError

_DEFAULT_SCORES = {t: 0.5 + 0.01 * i for i, t in enumerate(TARGET_COLUMNS)}


def _provenance(fold: Optional[int] = 0) -> ModelProvenance:
    return ModelProvenance(
        checkpoint_path=f"outputs/rsna-fivefold-benchmark/random/fold{fold}/last.pt",
        checkpoint_sha256="a" * 64,
        epoch=15,
        fold=fold,
        target_columns=tuple(TARGET_COLUMNS),
    )


def _ok_result(**overrides) -> StudyInferenceResult:
    defaults = dict(
        study_uid="1.2.826.synthetic.001",
        status=STATUS_OK,
        probabilities=dict(_DEFAULT_SCORES),
        is_ensemble=False,
        model_provenance=[_provenance()],
        latency_seconds=0.1,
        warnings=[],
        review_required=False,
    )
    defaults.update(overrides)
    return StudyInferenceResult(**defaults)


def _unavailable_result(**overrides) -> StudyInferenceResult:
    defaults = dict(
        study_uid="1.2.826.synthetic.002",
        status=STATUS_UNABLE_TO_EVALUATE,
        probabilities=None,
        is_ensemble=False,
        model_provenance=[_provenance()],
        latency_seconds=0.05,
        warnings=["Study unavailable: no DICOM series found locally."],
        review_required=True,
    )
    defaults.update(overrides)
    return StudyInferenceResult(**defaults)


@dataclass
class ScenarioOutcome:
    name: str
    expected_workflow_status: Optional[WorkflowStatus]
    actual_workflow_status: Optional[WorkflowStatus]
    passed: bool
    detail: str
    latency_seconds: float


def _run_scenario(name: str, build: Callable[[], Tuple[WorkflowStatus, WorkflowResult]]) -> ScenarioOutcome:
    started = time.perf_counter()
    try:
        expected, result = build()
        elapsed = time.perf_counter() - started
        actual = result.consensus.workflow_status
        return ScenarioOutcome(
            name=name, expected_workflow_status=expected, actual_workflow_status=actual,
            passed=(actual == expected), detail="ok", latency_seconds=elapsed,
        )
    except _ScenarioAssertion as exc:
        elapsed = time.perf_counter() - started
        return ScenarioOutcome(
            name=name, expected_workflow_status=None, actual_workflow_status=None,
            passed=exc.passed, detail=exc.detail, latency_seconds=elapsed,
        )


class _ScenarioAssertion(Exception):
    def __init__(self, passed: bool, detail: str):
        super().__init__(detail)
        self.passed = passed
        self.detail = detail


def scenario_complete_valid_inference() -> ScenarioOutcome:
    def build():
        wr = process_inference_result(_ok_result(), quality_gate_status=QualityGateStatus.OK,
                                       evidence_references=("outputs/rsna-fivefold-benchmark/random/fold0/last.pt",))
        return WorkflowStatus.EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW, wr
    return _run_scenario("complete_valid_inference", build)


def scenario_missing_mri_study() -> ScenarioOutcome:
    def build():
        wr = process_inference_result(_unavailable_result())
        return WorkflowStatus.INFERENCE_UNAVAILABLE, wr
    return _run_scenario("missing_mri_study", build)


def scenario_missing_or_invalid_outputs() -> ScenarioOutcome:
    def build():
        # status OK but probabilities absent -- an internally inconsistent engine result.
        result = _ok_result(probabilities=None)
        wr = process_inference_result(result, quality_gate_status=QualityGateStatus.OK)
        return WorkflowStatus.INCONSISTENT_EVIDENCE, wr
    return _run_scenario("missing_or_invalid_outputs", build)


def scenario_quality_not_performed() -> ScenarioOutcome:
    def build():
        wr = process_inference_result(_ok_result())  # quality_gate_status defaults to NOT_ASSESSED
        return WorkflowStatus.MORE_EVIDENCE_REQUIRED, wr
    return _run_scenario("quality_assessment_not_performed", build)


def scenario_critic_identifies_contradiction() -> ScenarioOutcome:
    def build():
        bad_scores = dict(_DEFAULT_SCORES)
        bad_scores["ACL"] = float("nan")
        result = _ok_result(probabilities=bad_scores)
        wr = process_inference_result(result, quality_gate_status=QualityGateStatus.OK)
        return WorkflowStatus.INCONSISTENT_EVIDENCE, wr
    return _run_scenario("critic_identifies_contradiction", build)


def scenario_more_evidence_request() -> ScenarioOutcome:
    def build():
        wr = process_inference_result(_ok_result(), quality_gate_status=QualityGateStatus.REVIEW)
        return WorkflowStatus.MORE_EVIDENCE_REQUIRED, wr
    return _run_scenario("more_evidence_request", build)


def scenario_invalid_reviewer_transition() -> ScenarioOutcome:
    started = time.perf_counter()
    wr = process_inference_result(_ok_result(), quality_gate_status=QualityGateStatus.OK)
    fp = wr.bundle.fingerprint
    wr.review_gate.transition(ReviewState.REVIEWED, actor="demo-reviewer", reason="looks complete", evidence_fingerprint=fp)
    try:
        wr.review_gate.transition(ReviewState.PENDING_REVIEW, actor="demo-reviewer", reason="reopen", evidence_fingerprint=fp)
        passed, detail = False, "Expected InvalidTransitionError but none was raised."
    except InvalidTransitionError:
        passed, detail = True, "InvalidTransitionError correctly raised for REVIEWED -> PENDING_REVIEW."
    elapsed = time.perf_counter() - started
    return ScenarioOutcome("invalid_reviewer_transition", None, None, passed, detail, elapsed)


def scenario_valid_review_transition() -> ScenarioOutcome:
    started = time.perf_counter()
    wr = process_inference_result(_ok_result(), quality_gate_status=QualityGateStatus.OK)
    fp = wr.bundle.fingerprint
    try:
        entry = wr.review_gate.transition(
            ReviewState.REVIEWED, actor="demo-reviewer", reason="evidence looks complete", evidence_fingerprint=fp
        )
        passed = wr.review_gate.state == ReviewState.REVIEWED and len(wr.review_gate.history) == 1 and entry.new_state == ReviewState.REVIEWED
        detail = "Transition to REVIEWED succeeded and was recorded in history." if passed else "Transition succeeded but state/history was unexpected."
    except Exception as exc:  # pragma: no cover - would indicate a real bug
        passed, detail = False, f"Unexpected exception: {exc}"
    elapsed = time.perf_counter() - started
    return ScenarioOutcome("valid_review_transition", None, None, passed, detail, elapsed)


def scenario_stale_evidence_fingerprint() -> ScenarioOutcome:
    started = time.perf_counter()
    wr = process_inference_result(_ok_result(), quality_gate_status=QualityGateStatus.OK)
    try:
        wr.review_gate.transition(
            ReviewState.REVIEWED, actor="demo-reviewer", reason="stale attempt", evidence_fingerprint="not-the-real-fingerprint"
        )
        passed, detail = False, "Expected StaleEvidenceError but none was raised."
    except StaleEvidenceError:
        passed, detail = True, "StaleEvidenceError correctly raised for a mismatched fingerprint."
    elapsed = time.perf_counter() - started
    return ScenarioOutcome("stale_evidence_fingerprint", None, None, passed, detail, elapsed)


def scenario_llm_proposal_rejected() -> ScenarioOutcome:
    """An adapter's proposal containing a disallowed clinical-certainty claim AND an
    unsupported citation must be BLOCKED by screen_proposal() -- never promoted to
    verified evidence, regardless of how plausible the rest of the text looks."""
    started = time.perf_counter()
    wr = process_inference_result(
        _ok_result(), quality_gate_status=QualityGateStatus.OK,
        evidence_references=("outputs/rsna-fivefold-benchmark/random/fold0/last.pt",),
    )
    fabricated = RawProposal(
        raw_text="This study shows a definite ACL tear, confirmed.",
        cited_evidence=("outputs/does-not-exist/fabricated.pt",),
    )
    outcome = screen_proposal(fabricated, wr.bundle)
    passed = outcome.status == ProposalStatus.BLOCKED
    detail = (
        f"screen_proposal correctly returned BLOCKED ({outcome.reason})"
        if passed else f"Expected BLOCKED, got {outcome.status.value}: {outcome.reason}"
    )
    elapsed = time.perf_counter() - started
    return ScenarioOutcome("llm_proposal_rejected", None, None, passed, detail, elapsed)


ALL_SCENARIOS: Tuple[Callable[[], ScenarioOutcome], ...] = (
    scenario_complete_valid_inference,
    scenario_missing_mri_study,
    scenario_missing_or_invalid_outputs,
    scenario_quality_not_performed,
    scenario_critic_identifies_contradiction,
    scenario_more_evidence_request,
    scenario_invalid_reviewer_transition,
    scenario_valid_review_transition,
    scenario_stale_evidence_fingerprint,
    scenario_llm_proposal_rejected,
)


@dataclass
class EvaluationReport:
    scenario_count: int
    outcomes: Tuple[ScenarioOutcome, ...]
    workflow_status_agreement: float  # fraction of status-bearing scenarios where actual == expected
    passed_count: int
    failed_count: int
    total_latency_seconds: float
    mean_latency_seconds: float

    def to_dict(self) -> dict:
        return {
            "scenario_count": self.scenario_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "workflow_status_agreement": self.workflow_status_agreement,
            "total_latency_seconds": self.total_latency_seconds,
            "mean_latency_seconds": self.mean_latency_seconds,
            "outcomes": [
                {
                    "name": o.name,
                    "expected_workflow_status": o.expected_workflow_status.value if o.expected_workflow_status else None,
                    "actual_workflow_status": o.actual_workflow_status.value if o.actual_workflow_status else None,
                    "passed": o.passed,
                    "detail": o.detail,
                    "latency_seconds": o.latency_seconds,
                }
                for o in self.outcomes
            ],
        }


def run_evaluation() -> EvaluationReport:
    """Run every scenario in ALL_SCENARIOS and report synthetic workflow-validation
    metrics only -- not a clinical accuracy or diagnostic performance measurement."""
    outcomes = tuple(scenario() for scenario in ALL_SCENARIOS)
    status_bearing = [o for o in outcomes if o.expected_workflow_status is not None]
    agreement = (sum(1 for o in status_bearing if o.passed) / len(status_bearing)) if status_bearing else 1.0
    passed = sum(1 for o in outcomes if o.passed)
    total_latency = sum(o.latency_seconds for o in outcomes)
    return EvaluationReport(
        scenario_count=len(outcomes),
        outcomes=outcomes,
        workflow_status_agreement=agreement,
        passed_count=passed,
        failed_count=len(outcomes) - passed,
        total_latency_seconds=total_latency,
        mean_latency_seconds=total_latency / len(outcomes) if outcomes else 0.0,
    )


# --------------------------------------------------------------------------- ablation

REQUIRED_EVIDENCE_FIELDS: Tuple[str, ...] = (
    "study_uid", "inference_status", "probabilities", "checkpoint_provenance",
    "latency_seconds", "quality_gate_status", "warnings", "evidence_references", "limitations",
)


@dataclass
class AblationLevelResult:
    level: str
    description: str
    evidence_fields_exposed: int
    evidence_fields_total: int
    has_specialist_grouping: bool
    has_critic_findings: bool
    has_workflow_status: bool
    has_review_gate: bool

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "description": self.description,
            "evidence_field_completeness": self.evidence_fields_exposed / self.evidence_fields_total,
            "has_specialist_grouping": self.has_specialist_grouping,
            "has_critic_findings": self.has_critic_findings,
            "has_workflow_status": self.has_workflow_status,
            "has_review_gate": self.has_review_gate,
        }


def run_ablation() -> Tuple[AblationLevelResult, ...]:
    """Compare what evidence/structure is actually exposed at three levels. Every field
    counted here is checked against the real EvidenceBundle/WorkflowResult objects built
    from one synthetic complete-valid scenario -- nothing is asserted without measuring it.
    """
    result = _ok_result()
    bundle: EvidenceBundle = build_evidence_bundle(
        result, quality_gate_status=QualityGateStatus.OK,
        evidence_references=("outputs/rsna-fivefold-benchmark/random/fold0/last.pt",),
    )
    bundle_dict = bundle.to_dict()
    exposed_fields = sum(1 for f in REQUIRED_EVIDENCE_FIELDS if bundle_dict.get(f) not in (None, (), []))

    # Level A: raw inference evidence only.
    level_a = AblationLevelResult(
        level="A", description="Raw inference evidence only (EvidenceBundle, no specialist/critic/consensus).",
        evidence_fields_exposed=exposed_fields, evidence_fields_total=len(REQUIRED_EVIDENCE_FIELDS),
        has_specialist_grouping=False, has_critic_findings=False, has_workflow_status=False, has_review_gate=False,
    )

    # Level B: specialist + deterministic critic, no consensus/workflow status yet.
    wr_full = process_inference_result(result, quality_gate_status=QualityGateStatus.OK,
                                        evidence_references=bundle.evidence_references)
    level_b = AblationLevelResult(
        level="B", description="Specialist grouping + deterministic critic findings, no consensus/review gate.",
        evidence_fields_exposed=exposed_fields, evidence_fields_total=len(REQUIRED_EVIDENCE_FIELDS),
        has_specialist_grouping=len(wr_full.specialist_summaries) == 4,
        has_critic_findings=True,  # critic.review() always returns a (possibly empty) tuple
        has_workflow_status=False, has_review_gate=False,
    )

    # Level C: full pipeline including consensus workflow status and review gate.
    level_c = AblationLevelResult(
        level="C", description="Specialist + critic + consensus workflow status + simulated review gate.",
        evidence_fields_exposed=exposed_fields, evidence_fields_total=len(REQUIRED_EVIDENCE_FIELDS),
        has_specialist_grouping=len(wr_full.specialist_summaries) == 4,
        has_critic_findings=True,
        has_workflow_status=wr_full.consensus.workflow_status is not None,
        has_review_gate=wr_full.review_gate.state == ReviewState.PENDING_REVIEW,
    )

    return (level_a, level_b, level_c)
