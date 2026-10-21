"""Focused test for Streamlit explanation environment fix."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from copy import copy

sys.path.insert(0, str(Path(__file__).parent.parent))

from hackathon.amazon_alexa.tools import ReviewStatus, MRIEvidence
from hackathon.amazon_alexa.tools import explain_review_status
from hackathon.amazon_alexa.review_agents import generate_review_support
from hackathon.amazon_alexa.llm_explanation import generate_enhanced_explanation, MODE_AUTO, MODE_DETERMINISTIC


def test_auto_mode_preserves_provider_configuration():
    """Verify Auto mode preserves GROQ_ENABLED, GROQ_API_KEY, GROQ_MODEL_ID."""
    # Create synthetic evidence
    evidence = MRIEvidence(
        case_id="case-001-consistent",
        baseline_date="2024-10-21",
        current_date="2024-10-21",
        quality_score=0.85,
        foreground_percentage=45.0,
        mean_intensity=128.0,
        status=ReviewStatus.CONSISTENT,
        change_description="Measured values are consistent.",
    )

    explanation = explain_review_status(evidence)
    packet = generate_review_support(evidence, explanation)

    # Create an environment dict that mimics the Streamlit fix:
    # Copy os.environ, then override only EXPLANATION_MODE
    env_with_mode = {
        "EXPLANATION_MODE": MODE_AUTO,
        "GROQ_ENABLED": "true",
        "GROQ_API_KEY": "test_key",
        "GROQ_MODEL_ID": "test_model",
        "LLM_TIMEOUT_SECONDS": "10",
    }

    # Mock the Groq API call (we don't want to call it, just verify env is passed correctly)
    with patch("hackathon.amazon_alexa.llm_explanation._call_groq") as mock_groq:
        mock_groq.return_value = "Mocked Groq response"

        # Call with Auto mode (should try Groq)
        result = generate_enhanced_explanation(packet, env_with_mode)

        # Verify _call_groq was called (meaning auto mode attempted the provider)
        assert mock_groq.called, "Auto mode should call _call_groq"

        # Verify the result comes from Groq
        assert result.source == "Groq", "Auto mode with valid config should return Groq source"


def test_deterministic_mode_overrides_explanation_mode():
    """Verify Deterministic-only mode forces EXPLANATION_MODE without breaking."""
    # Create synthetic evidence
    evidence = MRIEvidence(
        case_id="case-001-consistent",
        baseline_date="2024-10-21",
        current_date="2024-10-21",
        quality_score=0.85,
        foreground_percentage=45.0,
        mean_intensity=128.0,
        status=ReviewStatus.CONSISTENT,
        change_description="Measured values are consistent.",
    )

    explanation = explain_review_status(evidence)
    packet = generate_review_support(evidence, explanation)

    # Environment with MODE_DETERMINISTIC override
    env_with_mode = {
        "EXPLANATION_MODE": MODE_DETERMINISTIC,
        "GROQ_ENABLED": "true",  # Even if enabled, deterministic should skip
        "GROQ_API_KEY": "test_key",
        "GROQ_MODEL_ID": "test_model",
    }

    with patch("hackathon.amazon_alexa.llm_explanation._call_groq") as mock_groq:
        # Call with Deterministic mode (should NOT call Groq)
        result = generate_enhanced_explanation(packet, env_with_mode)

        # Verify _call_groq was NOT called
        assert not mock_groq.called, "Deterministic mode should not call _call_groq"

        # Verify the result is deterministic
        assert result.source == "Deterministic fallback", "Deterministic mode should use fallback"
