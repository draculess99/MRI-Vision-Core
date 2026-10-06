import itertools

from hackathon.nebius_nvidia.eval_harness import (
    METRICS_LABEL,
    Scenario,
    default_scenarios,
    run_evaluation,
)
from hackathon.nebius_nvidia.evidence_dossier import EvidenceStatus


def _fake_timer():
    # each build measures exactly 2 ms
    ticks = itertools.count()
    return lambda: next(ticks) * 0.002


def test_metrics_on_default_suite():
    r = run_evaluation(timer=_fake_timer())
    assert r["scenarios_evaluated"] == 4
    assert r["expected_status_agreement"] == 1.0
    assert r["evidence_field_completeness"] == 1.0
    assert r["human_approval_gate_enforcement_rate"] == 1.0
    assert abs(r["mean_dossier_build_latency_ms"] - 2.0) < 1e-6


def test_metrics_are_labelled_synthetic_not_clinical():
    r = run_evaluation()
    assert r["label"] == METRICS_LABEL
    assert "Synthetic" in r["label"] and "not clinical accuracy" in r["label"]


def test_non_latency_metrics_reproducible():
    a, b = run_evaluation(), run_evaluation()
    for key in (
        "scenarios_evaluated",
        "expected_status_agreement",
        "evidence_field_completeness",
        "human_approval_gate_enforcement_rate",
        "per_scenario",
    ):
        assert a[key] == b[key]
    assert a["mean_dossier_build_latency_ms"] > 0


def test_agreement_detects_wrong_expectation():
    base = default_scenarios()[0]
    wrong = Scenario(base.name, base.image, base.baseline, EvidenceStatus.REVIEW)
    r = run_evaluation((wrong,))
    assert r["expected_status_agreement"] == 0.0
    assert r["per_scenario"][0]["status_match"] is False
