import itertools

from hackathon.nebius_nvidia.demo import main, render_demo
from hackathon.nebius_nvidia.eval_harness import METRICS_LABEL


def _fake_timer():
    ticks = itertools.count()
    return lambda: next(ticks) * 0.002


def test_output_is_stable_across_runs():
    assert render_demo(_fake_timer()) == render_demo(_fake_timer())


def test_all_scenarios_and_key_facts_visible():
    text = render_demo(_fake_timer())
    for name in (
        "consistent_baseline_current",
        "large_change_review",
        "low_information_uncertain",
        "reviewer_approval",
    ):
        assert name in text
    for status in ("CONSISTENT", "REVIEW", "UNCERTAIN"):
        assert f"Status            : {status}" in text
    assert text.count("PENDING_HUMAN_APPROVAL") >= 4
    assert text.count("Non-diagnostic evidence summary") == 4
    assert "foreground=" in text and "Quality flags" in text


def test_reviewer_transition_keeps_safeguard():
    text = render_demo(_fake_timer())
    assert "PENDING_HUMAN_APPROVAL -> APPROVED_BY_HUMAN by Synthetic Reviewer" in text
    assert "human approval requirement retained (yes)" in text
    assert text.count("APPROVED_BY_HUMAN") == 1


def test_metrics_with_exact_label_and_values():
    text = render_demo(_fake_timer())
    assert METRICS_LABEL == (
        "Synthetic workflow-validation metrics only; not clinical accuracy "
        "or medical performance."
    )
    assert METRICS_LABEL in text
    assert "Scenarios evaluated            : 4" in text
    assert "Expected-status agreement      : 100%" in text
    assert "Mean dossier-build latency     : 2.0 ms" in text


def test_no_diagnosis_claims_and_ascii_safe():
    text = render_demo(_fake_timer())
    text.encode("ascii")
    assert "diagnosis" not in text.lower().replace("not a clinical diagnosis", "")


def test_main_prints(capsys):
    main()
    out = capsys.readouterr().out
    assert "MRI Evidence Dossier demo" in out
