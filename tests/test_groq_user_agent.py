"""Focused test for Groq User-Agent header and error logging."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).parent.parent))

from hackathon.amazon_alexa.tools import ReviewStatus, MRIEvidence
from hackathon.amazon_alexa.tools import explain_review_status
from hackathon.amazon_alexa.review_agents import generate_review_support
from hackathon.amazon_alexa.llm_explanation import _call_groq


def test_groq_request_includes_user_agent_header():
    """Verify that Groq requests include explicit User-Agent header."""
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

    # Mock urllib.request.urlopen to capture the request
    with patch("urllib.request.urlopen") as mock_urlopen:
        # Set up mock response
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices": [{"message": {"content": "test response"}}]}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # Mock environment
        env = {
            "GROQ_ENABLED": "true",
            "GROQ_API_KEY": "test_key",
            "GROQ_MODEL_ID": "test_model",
            "LLM_TIMEOUT_SECONDS": "10",
        }

        # Call the function
        result = _call_groq(packet, env)

        # Verify the request was made
        assert mock_urlopen.called, "urlopen should have been called"

        # Get the Request object passed to urlopen
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]  # First positional argument

        # Verify User-Agent header is present and correct
        assert request_obj.headers.get("User-agent") == "MRI-Evidence-Copilot/1.0", \
            f"User-Agent header should be 'MRI-Evidence-Copilot/1.0', got {request_obj.headers.get('User-agent')}"

        # Verify other headers are still present
        assert request_obj.headers.get("Authorization") is not None, "Authorization header should be present"
        assert request_obj.headers.get("Content-type") == "application/json", "Content-Type header should be present"

        # Verify response was processed
        assert result == "test response", "Response should be parsed correctly"

        # Verify the outbound request body contains the new plain-language instruction
        request_body = json.loads(call_args[0][0].data.decode())
        prompt_text = request_body["messages"][0]["content"]
        assert "Write exactly three short sentences in clear, plain language" in prompt_text, \
            "Prompt should instruct three plain-language sentences"
        assert "does not diagnose, make clinical decisions, or approve cases" in prompt_text, \
            "Prompt should require the non-diagnostic/non-approval disclosure"


def test_groq_logs_http_error_status_only():
    """Verify Groq logs only HTTP status code on HTTP error."""
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

    # Mock urllib.request.urlopen to raise HTTPError
    with patch("urllib.request.urlopen") as mock_urlopen:
        # Create HTTPError with status code 429 (rate limit)
        http_error = HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=None
        )
        mock_urlopen.side_effect = http_error

        # Mock environment
        env = {
            "GROQ_ENABLED": "true",
            "GROQ_API_KEY": "test_key",
            "GROQ_MODEL_ID": "test_model",
            "LLM_TIMEOUT_SECONDS": "10",
        }

        # Capture log output
        with patch("hackathon.amazon_alexa.llm_explanation.logger") as mock_logger:
            result = _call_groq(packet, env)

            # Verify returns None (fallback behavior preserved)
            assert result is None, "Should return None on HTTP error"

            # Verify warning log was called with only the HTTP status
            mock_logger.warning.assert_called_once()
            log_call = mock_logger.warning.call_args[0][0]
            assert "429" in log_call, "Should log HTTP status code"
            assert "Too Many Requests" not in log_call, "Should not log error message"


def test_groq_logs_exception_class_only():
    """Verify Groq logs only exception class name on non-HTTP exception."""
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

    # Mock urllib.request.urlopen to raise generic exception
    with patch("urllib.request.urlopen") as mock_urlopen:
        # Create a timeout exception
        timeout_error = TimeoutError("Connection timed out")
        mock_urlopen.side_effect = timeout_error

        # Mock environment
        env = {
            "GROQ_ENABLED": "true",
            "GROQ_API_KEY": "test_key",
            "GROQ_MODEL_ID": "test_model",
            "LLM_TIMEOUT_SECONDS": "10",
        }

        # Capture log output
        with patch("hackathon.amazon_alexa.llm_explanation.logger") as mock_logger:
            result = _call_groq(packet, env)

            # Verify returns None (fallback behavior preserved)
            assert result is None, "Should return None on exception"

            # Verify warning log was called with only the exception class name
            mock_logger.warning.assert_called_once()
            log_call = mock_logger.warning.call_args[0][0]
            assert "TimeoutError" in log_call, "Should log exception class name"
            assert "Connection timed out" not in log_call, "Should not log exception message"
