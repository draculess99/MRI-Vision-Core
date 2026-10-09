"""Tests for the Nebius Streamlit demo app.

Verifies the app can be imported and rendered without network calls,
credentials, or environment variables. Does not launch a Streamlit server
or browser.
"""

import pytest


def test_streamlit_app_imports_without_credentials():
    """Verify the Streamlit app module can be imported without API credentials or environment variables."""
    import sys
    from pathlib import Path

    # Test that the module can be imported even when run from repo root
    # This verifies the path bootstrap works correctly
    try:
        import hackathon.nebius_nvidia.streamlit_app as app_module
        assert app_module is not None
        # Verify the bootstrap is in place
        assert "_repo_root" in dir(app_module) or "sys" in dir(app_module)
    except ImportError as e:
        pytest.fail(f"Failed to import streamlit_app: {e}")


def test_streamlit_app_path_bootstrap():
    """Verify the app includes path bootstrap for local launches."""
    import inspect
    import hackathon.nebius_nvidia.streamlit_app as app_module

    source = inspect.getsource(app_module)

    # Verify path bootstrap is present
    assert "sys.path" in source, "Path bootstrap not found"
    assert "_repo_root" in source, "Repository root derivation not found"
    assert "Path(__file__)" in source, "Path derivation from __file__ not found"


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

    # Human-readable scenario labels
    assert "Consistent baseline/current" in source
    assert "Large-change review" in source
    assert "Low-information uncertainty" in source
    assert "Named-reviewer approval" in source

    # Selector label
    assert "Choose a synthetic workflow scenario" in source
    assert "Select a scenario to view its non-diagnostic evidence dossier" in source

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


def test_app_scenario_ids_unchanged():
    """Verify the original scenario IDs are still available to the app logic."""
    from hackathon.nebius_nvidia.eval_harness import default_scenarios

    scenarios = default_scenarios()
    assert len(scenarios) == 4

    # Original scenario IDs must be unchanged
    assert scenarios[0].name == "consistent_baseline_current"
    assert scenarios[1].name == "large_change_review"
    assert scenarios[2].name == "low_information_uncertain"
    assert scenarios[3].name == "reviewer_approval"

    # The app can select by index
    for i in range(4):
        assert scenarios[i] is not None


def test_readme_has_clean_utf8():
    """Verify the README contains no replacement characters and the command is present."""
    from pathlib import Path

    readme_path = Path(__file__).parent.parent / "hackathon" / "nebius_nvidia" / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    # No replacement characters
    assert "�" not in content, "README contains replacement characters"

    # Command is present
    assert "streamlit run hackathon/nebius_nvidia/streamlit_app.py" in content


def test_scenario_selector_css_styling():
    """Verify the scenario selector has strong CSS styling for visual prominence."""
    import inspect
    import hackathon.nebius_nvidia.streamlit_app as app_module

    source = inspect.getsource(app_module)

    # Verify selected field dimensions
    assert "64px" in source, "Selected field height/min-height (64px) not found"
    assert "1.2rem" in source, "Selected text font-size (1.2rem) not found"

    # Verify selected text weight
    assert "font-weight: 600" in source, "Selected text font-weight (600) not found"

    # Verify dropdown arrow/button size
    assert "1.5rem" in source, "Dropdown arrow/button size (1.5rem) not found"

    # Verify menu option dimensions
    assert "52px" in source, "Menu option min-height (52px) not found"
    assert "1.15rem" in source, "Menu option font-size (1.15rem) not found"

    # Verify styling approach targets the actual react-aria-components markup
    assert "!important" in source, "CSS !important declarations not found"
    assert "stSelectbox" in source, "Streamlit selectbox selector not found"
    assert 'role="group"' in source, "react-aria group container selector not found"
    assert 'role="combobox"' in source, "Combobox input selector not found"
    assert "box-sizing: border-box" in source, "Box-sizing property not found"

    # Verify menu selector
    assert 'role="listbox"' in source, "Listbox menu selector not found"
    assert 'role="option"' in source, "Option selector not found"

    # Verify colors and effects are retained
    assert "#00d4ff" in source, "Sky-blue border color not found"
    assert "#00e6ff" in source, "Hover/focus bright blue not found"
    assert "border-radius" in source, "Rounded corners not found"
