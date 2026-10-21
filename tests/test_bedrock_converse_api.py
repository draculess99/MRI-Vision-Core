"""Focused test for Bedrock Converse API adapter."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from hackathon.amazon_alexa.tools import ReviewStatus, MRIEvidence
from hackathon.amazon_alexa.tools import explain_review_status
from hackathon.amazon_alexa.review_agents import generate_review_support
from hackathon.amazon_alexa.llm_explanation import _call_bedrock


def test_bedrock_uses_converse_api_with_configured_model():
    """Verify Bedrock adapter uses client.converse with configured model ID."""
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

    # Mock boto3.client (imported inside the function)
    with patch("boto3.client") as mock_client_constructor:
        mock_client = MagicMock()
        mock_client_constructor.return_value = mock_client

        # Set up converse response with expected structure
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {"text": "Quality 0.85 (45% FG). Status: CONSISTENT."}
                    ]
                }
            }
        }

        # Mock environment
        env = {
            "BEDROCK_ENABLED": "true",
            "AWS_REGION": "us-east-1",
            "BEDROCK_MODEL_ID": "us.amazon.nova-lite-v1:0",
            "LLM_TIMEOUT_SECONDS": "10",
        }

        # Call the function
        result = _call_bedrock(packet, env)

        # Verify boto3.client was called with correct region
        mock_client_constructor.assert_called_once_with("bedrock-runtime", region_name="us-east-1")

        # Verify client.converse was called (not invoke_model)
        mock_client.converse.assert_called_once()

        # Get the actual converse call arguments
        call_args = mock_client.converse.call_args
        call_kwargs = call_args.kwargs

        # Verify model ID is passed through
        assert call_kwargs["modelId"] == "us.amazon.nova-lite-v1:0", \
            f"Model ID should be passed, got {call_kwargs['modelId']}"

        # Verify messages contain the prompt as user text
        messages = call_kwargs["messages"]
        assert len(messages) == 1, "Should have one message"
        assert messages[0]["role"] == "user", "Message should be from user"
        assert len(messages[0]["content"]) == 1, "Should have one content item"
        assert "text" in messages[0]["content"][0], "Content should have text field"

        prompt_text = messages[0]["content"][0]["text"]
        assert "Review assistant" in prompt_text, "Prompt should contain preamble"
        assert "Evidence:" in prompt_text, "Prompt should contain evidence section"
        assert "Key findings:" in prompt_text, "Prompt should contain findings section"

        # Verify inference config
        assert "inferenceConfig" in call_kwargs, "Should set inference config"
        assert call_kwargs["inferenceConfig"]["maxTokens"] == 200, "Should set maxTokens to 200"

        # Verify returned text is extracted correctly
        assert result == "Quality 0.85 (45% FG). Status: CONSISTENT.", \
            f"Should return extracted text, got {result}"
