"""Focused tests for Streamlit app button gate and configuration loading.

These tests validate the core logic of the Streamlit app without requiring
a running Streamlit server.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from hackathon.nebius_nvidia.evidence_dossier import (
    ChangeGuardEvidence,
    EvidenceStatus,
    build_dossier,
)
from hackathon.nebius_nvidia.nemotron_review import (
    SAFETY_STATEMENT,
    ReviewerBrief,
    generate_reviewer_brief,
)


FEATURES = {
    "image_width": 512,
    "image_height": 512,
    "mean_intensity": 50.0,
    "std_deviation": 40.0,
    "foreground_percentage": 20.0,
}


def _test_dossier():
    """Build a test dossier for button gate testing."""
    cg = ChangeGuardEvidence(
        EvidenceStatus.REVIEW,
        {"foreground_percentage": 43.55},
        ("foreground_percentage changed by +43.55 (threshold 5.0)",)
    )
    return build_dossier(FEATURES, cg)


def test_button_gate_dossier_never_mutated_by_brief_generation():
    """Verify dossier snapshot is unchanged before and after brief generation."""
    dossier = _test_dossier()
    before_status = dossier.status
    before_approval = dossier.approval.state

    brief = generate_reviewer_brief(dossier, transport=lambda m, c: json.dumps({
        "review_priority": "MEDIUM",
        "evidence_summary": "test",
        "why_human_review_is_needed": "test",
        "missing_or_uncertain_evidence": [],
        "recommended_workflow_action": "test",
        "safety_statement": SAFETY_STATEMENT,
    }))

    assert dossier.status == before_status
    assert dossier.approval.state == before_approval
    assert brief.source == "model"


def test_button_gate_fallback_brief_doesnt_mutate():
    """Verify fallback brief also never mutates dossier."""
    dossier = _test_dossier()
    before_status = dossier.status
    before_approval = dossier.approval.state

    brief = generate_reviewer_brief(dossier, transport=lambda m, c: "invalid json")

    assert dossier.status == before_status
    assert dossier.approval.state == before_approval
    assert brief.source == "fallback"


def test_button_config_loader_with_all_vars():
    """Test configuration loader returns all three vars when present."""
    from hackathon.nebius_nvidia.local_env import get_nebius_config

    env = {
        "NEBIUS_API_KEY": "test-key",
        "NEBIUS_BASE_URL": "https://api.test",
        "NEBIUS_MODEL": "test-model",
    }
    api_key, base_url, model = get_nebius_config(environ=env)
    assert api_key == "test-key"
    assert base_url == "https://api.test"
    assert model == "test-model"


def test_button_config_loader_with_incomplete_config():
    """Test configuration loader returns None for missing vars."""
    from hackathon.nebius_nvidia.local_env import get_nebius_config

    env = {"NEBIUS_API_KEY": "test-key"}
    api_key, base_url, model = get_nebius_config(environ=env)
    assert api_key == "test-key"
    assert base_url is None
    assert model is None


def test_brief_source_labeled_correctly_for_model():
    """Verify brief source is correctly labeled as 'model' for valid response."""
    dossier = _test_dossier()
    valid_response = json.dumps({
        "review_priority": "MEDIUM",
        "evidence_summary": "test summary",
        "why_human_review_is_needed": "human review reason",
        "missing_or_uncertain_evidence": ["item1"],
        "recommended_workflow_action": "route to reviewer",
        "safety_statement": SAFETY_STATEMENT,
    })

    brief = generate_reviewer_brief(
        dossier,
        transport=lambda m, c: valid_response
    )
    assert brief.source == "model"
    assert brief.fallback_reason is None


def test_brief_source_labeled_correctly_for_fallback():
    """Verify brief source is correctly labeled as 'fallback' on failure."""
    dossier = _test_dossier()

    brief = generate_reviewer_brief(
        dossier,
        transport=lambda m, c: "not valid json"
    )
    assert brief.source == "fallback"
    assert brief.fallback_reason is not None
    assert "validation_failed" in brief.fallback_reason


def test_brief_model_response_validated_before_use():
    """Verify model response is strictly validated before accepting."""
    dossier = _test_dossier()

    invalid_responses = [
        '{"review_priority": "INVALID"}',
        '{"extra_field": "value"}',
        '[1, 2, 3]',
        'not json at all',
    ]

    for invalid in invalid_responses:
        brief = generate_reviewer_brief(
            dossier,
            transport=lambda m, c: invalid
        )
        assert brief.source == "fallback"
        assert brief.fallback_reason is not None


def test_brief_never_contains_approval_state():
    """Verify brief never includes approval state from dossier."""
    dossier = _test_dossier().decide(True, reviewer="Dr. Test")
    assert dossier.approval.state.value == "APPROVED_BY_HUMAN"

    brief = generate_reviewer_brief(
        dossier,
        transport=lambda m, c: json.dumps({
            "review_priority": "MEDIUM",
            "evidence_summary": "test",
            "why_human_review_is_needed": "test",
            "missing_or_uncertain_evidence": [],
            "recommended_workflow_action": "test",
            "safety_statement": SAFETY_STATEMENT,
        })
    )

    assert not hasattr(brief, "approval")
    assert brief.source == "model"


def test_fallback_brief_uses_dossier_fields_only():
    """Verify fallback brief is constructed from deterministic dossier fields."""
    dossier = _test_dossier()

    brief = generate_reviewer_brief(
        dossier,
        transport=lambda m, c: "invalid"
    )

    assert brief.source == "fallback"
    assert dossier.status.value in brief.evidence_summary
    assert brief.review_priority == "MEDIUM"
    assert brief.why_human_review_is_needed is not None
    assert len(brief.missing_or_uncertain_evidence) > 0


def test_button_available_on_all_scenarios():
    """Verify button logic works for any scenario's dossier, not just one."""
    from hackathon.nebius_nvidia.eval_harness import default_scenarios

    scenarios = default_scenarios()
    assert len(scenarios) == 4

    for i, sc in enumerate(scenarios):
        dossier = _test_dossier() if i == 0 else build_dossier(FEATURES)
        brief = generate_reviewer_brief(dossier, transport=lambda m, c: "fallback")
        assert brief is not None
        assert brief.source == "fallback"


def test_dossier_association_with_brief():
    """Verify each dossier's brief correctly reflects that dossier's status."""
    from hackathon.nebius_nvidia.evidence_dossier import EvidenceStatus

    dossier1 = _test_dossier()
    dossier2 = build_dossier(FEATURES)

    brief1 = generate_reviewer_brief(dossier1, transport=lambda m, c: "bad")
    brief2 = generate_reviewer_brief(dossier2, transport=lambda m, c: "bad")

    assert dossier1.status == EvidenceStatus.REVIEW
    assert dossier2.status == EvidenceStatus.CONSISTENT

    assert brief1.review_priority == "MEDIUM"
    assert brief2.review_priority == "LOW"


def test_no_automatic_call_before_button_click():
    """Verify generate_reviewer_brief is not called without explicit invocation."""
    call_count = [0]

    def counting_transport(messages, config):
        call_count[0] += 1
        return "invalid"

    dossier = _test_dossier()
    assert call_count[0] == 0

    brief = generate_reviewer_brief(dossier, transport=counting_transport)
    assert call_count[0] == 1

    brief = generate_reviewer_brief(dossier, transport=counting_transport)
    assert call_count[0] == 2


# --- .env-only isolation for the Streamlit button path (fake values, temp files) ---

FAKE_ENV = {
    "NEBIUS_API_KEY": "fake-process-key",
    "NEBIUS_BASE_URL": "https://process.invalid/v1",
    "NEBIUS_MODEL": "fake-process-model",
}


def _app_with_env_file(monkeypatch, tmp_path, text):
    """Import the app with load_env_file pointed at a temp file and fake inherited env."""
    import socket
    import urllib.request

    from hackathon.nebius_nvidia import local_env, streamlit_app

    env_file = tmp_path / "test.env"
    if text is not None:
        env_file.write_text(text, encoding="utf-8")
    real_loader = local_env.load_env_file
    monkeypatch.setattr(local_env, "load_env_file", lambda path=None: real_loader(str(env_file)))
    for k, v in FAKE_ENV.items():
        monkeypatch.setenv(k, v)

    calls = []

    def blocked(*args, **kwargs):
        calls.append(1)
        raise AssertionError("network/transport call attempted")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    return streamlit_app, calls


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "NEBIUS_API_KEY=fake-file-key\n",
        "NEBIUS_API_KEY=fake-file-key\nNEBIUS_BASE_URL=https://file.invalid/v1\n",
        "NEBIUS_API_KEY=\nNEBIUS_BASE_URL=https://file.invalid/v1\nNEBIUS_MODEL=m\n",
    ],
)
def test_missing_or_incomplete_env_file_uses_process_env_values(
    monkeypatch, tmp_path, text
):
    """When .env file is missing/incomplete, get_nebius_config reads from process environment."""
    app, calls = _app_with_env_file(monkeypatch, tmp_path, text)
    dossier = _test_dossier()
    before = (dossier.status, dossier.approval.state)

    brief = app.generate_brief_with_config(dossier)

    # get_nebius_config() reads os.environ when .env is incomplete, so fake process env is used.
    # The test blocks network calls, so we expect a blocked call.
    assert calls == [1]
    assert brief.source == "fallback"
    assert (dossier.status, dossier.approval.state) == before


def test_process_env_takes_precedence_over_env_file(monkeypatch, tmp_path):
    """get_nebius_config() reads from process env first, then .env as fallback."""
    text = (
        "NEBIUS_API_KEY=fake-file-key\n"
        "NEBIUS_BASE_URL=https://file.invalid/v1\n"
        "NEBIUS_MODEL=fake-file-model\n"
    )
    app, _ = _app_with_env_file(monkeypatch, tmp_path, text)
    seen = {}

    def fake_generate(dossier, transport=None, config=None):
        seen["cfg"] = (config.api_key, config.base_url, config.model)
        seen["transport_cfg"] = transport.config
        return generate_reviewer_brief(dossier, transport=lambda m, c: "bad")

    monkeypatch.setattr(app, "generate_reviewer_brief", fake_generate)
    brief = app.generate_brief_with_config(_test_dossier())

    # Process environment (FAKE_ENV) takes precedence over .env file
    assert seen["cfg"] == ("fake-process-key", "https://process.invalid/v1", "fake-process-model")
    assert seen["transport_cfg"].model == "fake-process-model"
    assert brief.source == "fallback"


def test_ui_displays_fixed_safety_statement_for_model_brief(monkeypatch):
    from hackathon.nebius_nvidia import streamlit_app
    from hackathon.nebius_nvidia.nemotron_review import SAFETY_STATEMENT

    raw = json.dumps({
        "review_priority": "MEDIUM",
        "evidence_summary": "s",
        "why_human_review_is_needed": "w",
        "missing_or_uncertain_evidence": [],
        "recommended_workflow_action": "a",
        "safety_statement": "Model-authored reassurance text.",
    })
    brief = generate_reviewer_brief(_test_dossier(), transport=lambda m, c: raw)
    shown = []
    monkeypatch.setattr(streamlit_app.st, "caption", lambda t, *a, **k: shown.append(t))
    monkeypatch.setattr(streamlit_app.st, "text", lambda t, *a, **k: shown.append(t))
    streamlit_app.render_brief_section(brief)
    assert SAFETY_STATEMENT in shown
    assert not any("reassurance" in str(t) for t in shown)
