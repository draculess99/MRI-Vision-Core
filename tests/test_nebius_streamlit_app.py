"""Tests for the Nebius Streamlit demo app.

Verifies the app can be imported and rendered without network calls,
credentials, or environment variables. Does not launch a Streamlit server
or browser.
"""

import pytest


def test_streamlit_app_imports_without_credentials():
    """Verify the Streamlit app module can be imported without API credentials."""
    try:
        import hackathon.nebius_nvidia.streamlit_app as app_module
        assert app_module is not None
    except ImportError as e:
        pytest.fail(f"Failed to import streamlit_app: {e}")


def test_app_title_exact_unicode():
    """Verify the app title is exactly correct with no replacement characters."""
    from hackathon.nebius_nvidia.streamlit_app import APP_TITLE

    expected = "MRI Evidence Dossier — Nebius × NVIDIA"
    assert APP_TITLE == expected
    # Verify no replacement characters (U+FFFD) are present
    assert "�" not in APP_TITLE


def test_source_has_no_replacement_characters():
    """Verify the app source contains no replacement characters."""
    import inspect
    import hackathon.nebius_nvidia.streamlit_app as app_module

    source = inspect.getsource(app_module)
    assert "�" not in source, "Source contains replacement characters"


def test_app_is_credential_free():
    """Verify the app does NOT read environment variables or request credentials."""
    import inspect
    import hackathon.nebius_nvidia.streamlit_app as app_module

    source = inspect.getsource(app_module)

    # Should not read environment variables
    assert "os.environ" not in source, "App reads environment variables"
    assert "os.getenv" not in source, "App reads environment variables"
    assert "getenv" not in source, "App reads environment variables"

    # Should not reference credential-related constants
    assert "NEBIUS_API_KEY" not in source, "App references NEBIUS_API_KEY"
    assert "NEBIUS_BASE_URL" not in source, "App references NEBIUS_BASE_URL"
    assert "NEBIUS_MODEL" not in source, "App references NEBIUS_MODEL"


def test_app_does_not_import_or_call_credential_modules():
    """Verify the app does NOT import or use credential-dependent modules."""
    import inspect
    import hackathon.nebius_nvidia.streamlit_app as app_module

    source = inspect.getsource(app_module)

    # Should not import credential-related modules
    assert "NebiusConfig" not in source, "App imports or uses NebiusConfig"
    assert "UrllibTransport" not in source, "App imports or uses UrllibTransport"
    assert "generate_reviewer_brief" not in source, "App calls generate_reviewer_brief"
    assert "nemotron_review" not in source, "App imports nemotron_review module"


def test_streamlit_app_uses_existing_eval_logic():
    """Verify the app reuses existing deterministic scenario and evaluation logic."""
    from hackathon.nebius_nvidia.eval_harness import default_scenarios, METRICS_LABEL
    from hackathon.nebius_nvidia.evidence_dossier import build_dossier_from_images

    # The app must use these without modification
    scenarios = default_scenarios()
    assert len(scenarios) == 4

    scenario_names = {sc.name for sc in scenarios}
    assert "consistent_baseline_current" in scenario_names
    assert "large_change_review" in scenario_names
    assert "low_information_uncertain" in scenario_names
    assert "reviewer_approval" in scenario_names

    # Verify metrics label is exact
    assert METRICS_LABEL == (
        "Synthetic workflow-validation metrics only; not clinical accuracy "
        "or medical performance."
    )

    # Verify building a dossier works
    d = build_dossier_from_images(scenarios[0].image, baseline_image=scenarios[0].baseline)
    assert d is not None
    assert d.requires_human_approval is True


def test_app_demonstrates_approval_transition():
    """Verify the app can show the PENDING -> APPROVED transition."""
    from hackathon.nebius_nvidia.eval_harness import default_scenarios
    from hackathon.nebius_nvidia.evidence_dossier import (
        ApprovalState,
        build_dossier_from_images,
    )

    # Scenario 4 has reviewer_decision=True
    reviewer_scenario = default_scenarios()[3]
    assert reviewer_scenario.reviewer_decision is True

    d = build_dossier_from_images(
        reviewer_scenario.image, baseline_image=reviewer_scenario.baseline
    )
    before_state = d.approval.state
    assert before_state is ApprovalState.PENDING

    # Simulate the app's decision flow
    decided = d.decide(True, reviewer="Synthetic Reviewer", comment="demo review")
    after_state = decided.approval.state
    assert after_state is ApprovalState.APPROVED

    # Original dossier must not change
    assert d.approval.state is ApprovalState.PENDING


def test_app_content_includes_required_elements():
    """Verify the app source includes all required non-diagnostic and safeguard text."""
    import inspect
    import hackathon.nebius_nvidia.streamlit_app as app_module

    source = inspect.getsource(app_module)

    # Title (checked separately via APP_TITLE constant)
    assert "st.title(APP_TITLE)" in source

    # Non-diagnostic disclaimer
    assert "non-diagnostic" in source.lower()
    assert "Not for medical diagnosis" in source

    # Scenario references (via default_scenarios())
    assert "default_scenarios" in source

    # Approval safeguard
    assert "PENDING_HUMAN_APPROVAL" in source
    assert "human approval required" in source.lower()
    assert "cannot be bypassed" in source.lower()

    # Named-reviewer transition
    assert "APPROVED_BY_HUMAN" in source
    assert "Synthetic Reviewer" in source

    # Nebius integration info
    assert "NVIDIA/NVIDIA-Nemotron-3-Nano-30B-A3B" in source
    assert "deterministic workflow only" in source.lower()


def test_readme_has_clean_utf8():
    """Verify the README contains no replacement characters and the command is present."""
    from pathlib import Path

    readme_path = Path(__file__).parent.parent / "hackathon" / "nebius_nvidia" / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    # No replacement characters
    assert "�" not in content, "README contains replacement characters"

    # Command is present
    assert "streamlit run hackathon/nebius_nvidia/streamlit_app.py" in content
