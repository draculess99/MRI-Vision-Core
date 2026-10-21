"""Tests for optional LLM-based explanation providers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from hackathon.amazon_alexa.llm_explanation import (
    ENV_BEDROCK_ENABLED,
    ENV_BEDROCK_MODEL_ID,
    ENV_BEDROCK_REGION,
    ENV_EXPLANATION_MODE,
    ENV_GROQ_API_KEY,
    ENV_GROQ_ENABLED,
    ENV_GROQ_MODEL_ID,
    ENV_LLM_TIMEOUT,
    MAX_TIMEOUT,
    MIN_TIMEOUT,
    MODE_AUTO,
    MODE_DETERMINISTIC,
    _call_bedrock,
    _call_groq,
    _deterministic_explanation,
    _get_bedrock_model_id,
    _get_bedrock_region,
    _get_explanation_mode,
    _get_groq_api_key,
    _get_groq_model_id,
    _get_llm_timeout,
    _is_bedrock_enabled,
    _is_groq_enabled,
    _validate_explanation_output,
    generate_enhanced_explanation,
)
from hackathon.amazon_alexa.review_agents import ReviewSupportPacket, generate_review_support
from hackathon.amazon_alexa.tools import analyze_mri_evidence, explain_review_status


class TestCredentialHandling:
    """Test credential retrieval from environment variables."""

    def test_get_bedrock_region_from_env(self):
        """_get_bedrock_region returns region from AWS_REGION."""
        env = {ENV_BEDROCK_REGION: "us-west-2"}
        region = _get_bedrock_region(env)
        assert region == "us-west-2"

    def test_get_bedrock_region_missing(self):
        """_get_bedrock_region returns None when not set."""
        env = {}
        region = _get_bedrock_region(env)
        assert region is None

    def test_get_bedrock_region_whitespace_only(self):
        """_get_bedrock_region returns None for whitespace."""
        env = {ENV_BEDROCK_REGION: "  "}
        region = _get_bedrock_region(env)
        assert region is None

    def test_get_groq_api_key_present(self):
        """_get_groq_api_key returns key when set."""
        env = {ENV_GROQ_API_KEY: "gsk_test123"}
        key = _get_groq_api_key(env)
        assert key == "gsk_test123"

    def test_get_groq_api_key_missing(self):
        """_get_groq_api_key returns None when not set."""
        env = {}
        key = _get_groq_api_key(env)
        assert key is None

    def test_get_groq_api_key_whitespace_only(self):
        """_get_groq_api_key returns None for whitespace."""
        env = {ENV_GROQ_API_KEY: "  "}
        key = _get_groq_api_key(env)
        assert key is None


class TestOutputValidation:
    """Test LLM output validation and safety checks."""

    def test_valid_explanation(self):
        """Valid explanation passes validation."""
        text = "The evidence shows consistent measurements with good image quality."
        valid, error = _validate_explanation_output(text)
        assert valid is True
        assert error is None

    def test_empty_output_invalid(self):
        """Empty output fails validation."""
        valid, error = _validate_explanation_output("")
        assert valid is False
        assert "Empty" in error

    def test_whitespace_only_invalid(self):
        """Whitespace-only output fails validation."""
        valid, error = _validate_explanation_output("   ")
        assert valid is False
        assert "Empty" in error

    def test_too_long_output_invalid(self):
        """Output exceeding max length fails validation."""
        text = "x" * 600
        valid, error = _validate_explanation_output(text, max_len=500)
        assert valid is False
        assert "exceeds maximum length" in error

    def test_approval_language_rejected(self):
        """Output containing approval language is rejected."""
        rejected_texts = [
            "This case is approved.",
            "The reviewer can approve this case.",
            "Authorization is recommended.",
            "This should be signed off.",
        ]
        for text in rejected_texts:
            valid, error = _validate_explanation_output(text)
            assert valid is False, f"Should reject: {text}"
            assert "approval" in error.lower() or "sign" in error.lower()

    def test_diagnosis_language_rejected(self):
        """Output containing diagnostic language is rejected."""
        rejected_texts = [
            "The diagnosis is cancer.",
            "This patient should be treated with chemotherapy.",
            "I recommend surgery.",
            "The condition must be treated immediately.",
        ]
        for text in rejected_texts:
            valid, error = _validate_explanation_output(text)
            assert valid is False, f"Should reject: {text}"

    def test_treatment_language_rejected(self):
        """Output containing treatment recommendations is rejected."""
        rejected_texts = [
            "Recommend treatment with medication.",
            "The patient should receive therapy.",
            "Treatment is indicated.",
        ]
        for text in rejected_texts:
            valid, error = _validate_explanation_output(text)
            assert valid is False, f"Should reject: {text}"


class TestBedrockProvider:
    """Test Bedrock provider."""

    def test_bedrock_no_region_returns_none(self):
        """_call_bedrock returns None when region is not configured."""
        with patch.dict("os.environ", {}, clear=True):
            result = _call_bedrock(
                ReviewSupportPacket(
                    evidence_summary="Test",
                    critic_findings=[],
                    reviewer_checks=[],
                )
            )
            assert result is None

    def test_bedrock_boto3_not_installed_returns_none(self):
        """_call_bedrock returns None when boto3 is unavailable."""
        env = {ENV_BEDROCK_REGION: "us-west-2"}
        with patch.dict("os.environ", env):
            with patch("builtins.__import__", side_effect=ImportError):
                result = _call_bedrock(
                    ReviewSupportPacket(
                        evidence_summary="Test",
                        critic_findings=[],
                        reviewer_checks=[],
                    )
                )
                assert result is None

    def test_bedrock_success_mocked(self):
        """_call_bedrock returns text on successful mocked call."""
        env = {
            ENV_BEDROCK_ENABLED: "true",
            ENV_BEDROCK_REGION: "us-west-2",
            ENV_BEDROCK_MODEL_ID: "test-model",
        }
        packet = ReviewSupportPacket(
            evidence_summary="Case 001",
            critic_findings=["Finding 1"],
            reviewer_checks=["Check 1"],
        )

        mock_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps({
                        "content": [{"text": "Enhanced explanation from Bedrock."}]
                    }).encode()
                )
            )
        }

        with patch("boto3.client") as mock_client_func:
            mock_client = MagicMock()
            mock_client_func.return_value = mock_client
            mock_client.invoke_model.return_value = mock_response

            result = _call_bedrock(packet, env)
            assert result == "Enhanced explanation from Bedrock."


class TestGroqProvider:
    """Test Groq provider."""

    def test_groq_no_key_returns_none(self):
        """_call_groq returns None when API key is not configured."""
        with patch.dict("os.environ", {}, clear=True):
            result = _call_groq(
                ReviewSupportPacket(
                    evidence_summary="Test",
                    critic_findings=[],
                    reviewer_checks=[],
                )
            )
            assert result is None

    def test_groq_success_mocked(self):
        """_call_groq returns text on successful mocked call."""
        env = {
            ENV_GROQ_ENABLED: "true",
            ENV_GROQ_API_KEY: "gsk_test123",
            ENV_GROQ_MODEL_ID: "test-model",
        }
        packet = ReviewSupportPacket(
            evidence_summary="Case 001",
            critic_findings=["Finding 1"],
            reviewer_checks=["Check 1"],
        )

        mock_response = MagicMock(
            read=MagicMock(
                return_value=json.dumps({
                    "choices": [{"message": {"content": "Enhanced explanation from Groq."}}]
                }).encode()
            ),
            __enter__=MagicMock(return_value=MagicMock(
                read=MagicMock(
                    return_value=json.dumps({
                        "choices": [{"message": {"content": "Enhanced explanation from Groq."}}]
                    }).encode()
                )
            )),
            __exit__=MagicMock(return_value=None),
        )

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = _call_groq(packet, env)
            assert result == "Enhanced explanation from Groq."


class TestDeterministicFallback:
    """Test deterministic fallback explanation."""

    def test_deterministic_explanation_synthesized(self):
        """_deterministic_explanation synthesizes from packet."""
        packet = ReviewSupportPacket(
            evidence_summary="Case 001 summary",
            critic_findings=["Finding 1", "Finding 2"],
            reviewer_checks=["Check 1", "Check 2"],
        )
        explanation = _deterministic_explanation(packet)

        assert "Case 001 summary" in explanation
        assert "Finding 1" in explanation
        assert "Check 1" in explanation
        assert "human reviewer" in explanation.lower()

    def test_deterministic_explanation_is_non_diagnostic(self):
        """_deterministic_explanation does not contain diagnostic language."""
        evidence = analyze_mri_evidence("case-002-change")
        exp = explain_review_status(evidence)
        packet = generate_review_support(evidence, exp)

        explanation = _deterministic_explanation(packet)

        # Should not contain diagnostic/treatment language
        assert "diagnose" not in explanation.lower()
        assert "treatment" not in explanation.lower()
        assert "recommend" not in explanation.lower()


class TestFallbackChain:
    """Test the Bedrock → Groq → deterministic fallback chain."""

    def test_bedrock_success_no_groq_call(self):
        """When Bedrock succeeds, Groq is not called."""
        env = {
            ENV_EXPLANATION_MODE: MODE_AUTO,
            ENV_BEDROCK_ENABLED: "true",
            ENV_BEDROCK_REGION: "us-west-2",
            ENV_BEDROCK_MODEL_ID: "test-model",
            ENV_GROQ_ENABLED: "true",
            ENV_GROQ_API_KEY: "gsk_test",
            ENV_GROQ_MODEL_ID: "groq-model",
        }
        evidence = analyze_mri_evidence("case-001-consistent")
        exp = explain_review_status(evidence)
        packet = generate_review_support(evidence, exp)

        mock_bedrock_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps({
                        "content": [{"text": "Bedrock explanation."}]
                    }).encode()
                )
            )
        }

        with patch("boto3.client") as mock_boto3_client:
            with patch("urllib.request.urlopen") as mock_groq:
                mock_client = MagicMock()
                mock_boto3_client.return_value = mock_client
                mock_client.invoke_model.return_value = mock_bedrock_response

                result = generate_enhanced_explanation(packet, env)

                assert result.source == "Amazon Bedrock"
                assert result.text == "Bedrock explanation."
                mock_groq.assert_not_called()

    def test_bedrock_failure_groq_success(self):
        """When Bedrock fails, Groq is tried."""
        env = {
            ENV_EXPLANATION_MODE: MODE_AUTO,
            ENV_BEDROCK_ENABLED: "true",
            ENV_BEDROCK_REGION: "us-west-2",
            ENV_BEDROCK_MODEL_ID: "test-model",
            ENV_GROQ_ENABLED: "true",
            ENV_GROQ_API_KEY: "gsk_test",
            ENV_GROQ_MODEL_ID: "groq-model",
        }
        evidence = analyze_mri_evidence("case-001-consistent")
        exp = explain_review_status(evidence)
        packet = generate_review_support(evidence, exp)

        mock_groq_response = MagicMock(
            read=MagicMock(
                return_value=json.dumps({
                    "choices": [{"message": {"content": "Groq explanation."}}]
                }).encode()
            ),
            __enter__=MagicMock(return_value=MagicMock(
                read=MagicMock(
                    return_value=json.dumps({
                        "choices": [{"message": {"content": "Groq explanation."}}]
                    }).encode()
                )
            )),
            __exit__=MagicMock(return_value=None),
        )

        with patch.dict("os.environ", env):
            with patch("boto3.client") as mock_boto3_client:
                with patch("urllib.request.urlopen", return_value=mock_groq_response):
                    mock_client = MagicMock()
                    mock_boto3_client.return_value = mock_client
                    mock_client.invoke_model.side_effect = Exception("Bedrock error")

                    result = generate_enhanced_explanation(packet)

                    assert result.source == "Groq"
                    assert result.text == "Groq explanation."

    def test_both_providers_fail_deterministic_fallback(self):
        """When both LLM providers fail, deterministic fallback is used."""
        env = {ENV_BEDROCK_REGION: "us-west-2", ENV_GROQ_API_KEY: "gsk_test"}
        evidence = analyze_mri_evidence("case-001-consistent")
        exp = explain_review_status(evidence)
        packet = generate_review_support(evidence, exp)

        with patch.dict("os.environ", env):
            with patch("boto3.client") as mock_boto3_client:
                with patch("urllib.request.urlopen") as mock_groq:
                    mock_client = MagicMock()
                    mock_boto3_client.return_value = mock_client
                    mock_client.invoke_model.side_effect = Exception("Bedrock error")
                    mock_groq.side_effect = Exception("Groq error")

                    result = generate_enhanced_explanation(packet)

                    assert result.source == "Deterministic fallback"
                    assert "Evidence Review Summary" in result.text
                    assert "human reviewer" in result.text

    def test_invalid_bedrock_output_tries_groq(self):
        """When Bedrock output fails validation, Groq is tried."""
        env = {
            ENV_EXPLANATION_MODE: MODE_AUTO,
            ENV_BEDROCK_ENABLED: "true",
            ENV_BEDROCK_REGION: "us-west-2",
            ENV_BEDROCK_MODEL_ID: "test-model",
            ENV_GROQ_ENABLED: "true",
            ENV_GROQ_API_KEY: "gsk_test",
            ENV_GROQ_MODEL_ID: "groq-model",
        }
        evidence = analyze_mri_evidence("case-001-consistent")
        exp = explain_review_status(evidence)
        packet = generate_review_support(evidence, exp)

        mock_bedrock_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps({
                        "content": [{"text": "This case is approved."}]  # Invalid: approval language
                    }).encode()
                )
            )
        }

        mock_groq_response = MagicMock(
            read=MagicMock(
                return_value=json.dumps({
                    "choices": [{"message": {"content": "Valid Groq explanation."}}]
                }).encode()
            ),
            __enter__=MagicMock(return_value=MagicMock(
                read=MagicMock(
                    return_value=json.dumps({
                        "choices": [{"message": {"content": "Valid Groq explanation."}}]
                    }).encode()
                )
            )),
            __exit__=MagicMock(return_value=None),
        )

        with patch.dict("os.environ", env):
            with patch("boto3.client") as mock_boto3_client:
                with patch("urllib.request.urlopen", return_value=mock_groq_response):
                    mock_client = MagicMock()
                    mock_boto3_client.return_value = mock_client
                    mock_client.invoke_model.return_value = mock_bedrock_response

                    result = generate_enhanced_explanation(packet)

                    assert result.source == "Groq"


class TestSafetyInvariants:
    """Test that safety invariants are maintained."""

    def test_approval_invariant_maintained(self):
        """Generated explanation never contains approval state."""
        evidence = analyze_mri_evidence("case-002-change")
        exp = explain_review_status(evidence)
        packet = generate_review_support(evidence, exp)

        # Packet itself
        assert packet.approval_required is True
        assert "APPROVED_BY_HUMAN" not in packet.evidence_summary

        # Fallback explanation
        explanation = _deterministic_explanation(packet)
        assert "approve" not in explanation.lower() or "cannot" in explanation.lower()

    def test_no_secret_exposure_in_fallback(self):
        """Fallback explanation contains no credentials or secrets."""
        evidence = analyze_mri_evidence("case-001-consistent")
        exp = explain_review_status(evidence)
        packet = generate_review_support(evidence, exp)

        explanation = _deterministic_explanation(packet)

        # Should not contain API key patterns (specific to credential exposure)
        assert "gsk_" not in explanation.lower()
        assert "api_key" not in explanation.lower()
        assert "apikey" not in explanation.lower()
        assert "bedrock_key" not in explanation.lower()
        assert "groq_key" not in explanation.lower()

    def test_packet_immutable_after_generation(self):
        """ReviewSupportPacket remains frozen after explanation generation."""
        evidence = analyze_mri_evidence("case-001-consistent")
        exp = explain_review_status(evidence)
        packet = generate_review_support(evidence, exp)

        # Try to modify (should fail)
        with pytest.raises(AttributeError):
            packet.evidence_summary = "Modified"

    def test_evidence_never_altered(self):
        """Source evidence is never altered during explanation generation."""
        evidence = analyze_mri_evidence("case-002-change")
        exp = explain_review_status(evidence)
        original_status = evidence.status
        original_case_id = evidence.case_id

        packet = generate_review_support(evidence, exp)
        _deterministic_explanation(packet)

        # Evidence unchanged
        assert evidence.status == original_status
        assert evidence.case_id == original_case_id


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""

    def test_get_explanation_mode_default(self):
        """EXPLANATION_MODE defaults to deterministic."""
        env = {}
        mode = _get_explanation_mode(env)
        assert mode == MODE_DETERMINISTIC

    def test_get_explanation_mode_auto(self):
        """EXPLANATION_MODE can be set to auto."""
        env = {ENV_EXPLANATION_MODE: "auto"}
        mode = _get_explanation_mode(env)
        assert mode == MODE_AUTO

    def test_get_explanation_mode_invalid_defaults_to_deterministic(self):
        """Invalid EXPLANATION_MODE defaults to deterministic."""
        env = {ENV_EXPLANATION_MODE: "invalid"}
        mode = _get_explanation_mode(env)
        assert mode == MODE_DETERMINISTIC

    def test_bedrock_disabled_by_default(self):
        """BEDROCK_ENABLED defaults to false."""
        env = {}
        assert _is_bedrock_enabled(env) is False

    def test_bedrock_enabled_only_with_explicit_true(self):
        """BEDROCK_ENABLED must be explicitly 'true'."""
        assert _is_bedrock_enabled({ENV_BEDROCK_ENABLED: "true"}) is True
        assert _is_bedrock_enabled({ENV_BEDROCK_ENABLED: "True"}) is True
        assert _is_bedrock_enabled({ENV_BEDROCK_ENABLED: "TRUE"}) is True
        assert _is_bedrock_enabled({ENV_BEDROCK_ENABLED: "false"}) is False
        assert _is_bedrock_enabled({ENV_BEDROCK_ENABLED: "1"}) is False
        assert _is_bedrock_enabled({ENV_BEDROCK_ENABLED: ""}) is False

    def test_get_bedrock_model_id(self):
        """Get BEDROCK_MODEL_ID from environment."""
        env = {ENV_BEDROCK_MODEL_ID: "anthropic.claude-3-haiku-20240307-v1:0"}
        model_id = _get_bedrock_model_id(env)
        assert model_id == "anthropic.claude-3-haiku-20240307-v1:0"

    def test_get_bedrock_model_id_missing(self):
        """BEDROCK_MODEL_ID returns None when not set."""
        env = {}
        model_id = _get_bedrock_model_id(env)
        assert model_id is None

    def test_groq_disabled_by_default(self):
        """GROQ_ENABLED defaults to false."""
        env = {}
        assert _is_groq_enabled(env) is False

    def test_groq_enabled_only_with_explicit_true(self):
        """GROQ_ENABLED must be explicitly 'true'."""
        assert _is_groq_enabled({ENV_GROQ_ENABLED: "true"}) is True
        assert _is_groq_enabled({ENV_GROQ_ENABLED: "false"}) is False
        assert _is_groq_enabled({ENV_GROQ_ENABLED: ""}) is False

    def test_get_groq_model_id(self):
        """Get GROQ_MODEL_ID from environment."""
        env = {ENV_GROQ_MODEL_ID: "mixtral-8x7b-32768"}
        model_id = _get_groq_model_id(env)
        assert model_id == "mixtral-8x7b-32768"

    def test_get_groq_model_id_missing(self):
        """GROQ_MODEL_ID returns None when not set."""
        env = {}
        model_id = _get_groq_model_id(env)
        assert model_id is None

    def test_get_llm_timeout_default(self):
        """LLM_TIMEOUT_SECONDS defaults to 10."""
        env = {}
        timeout = _get_llm_timeout(env)
        assert timeout == 10

    def test_get_llm_timeout_valid_values(self):
        """LLM_TIMEOUT_SECONDS accepts values from 1 to 60."""
        for seconds in [1, 5, 10, 30, 60]:
            env = {ENV_LLM_TIMEOUT: str(seconds)}
            timeout = _get_llm_timeout(env)
            assert timeout == seconds

    def test_get_llm_timeout_bounds_checking(self):
        """LLM_TIMEOUT_SECONDS enforces bounds (1-60)."""
        assert _get_llm_timeout({ENV_LLM_TIMEOUT: "0"}) == 10
        assert _get_llm_timeout({ENV_LLM_TIMEOUT: "61"}) == 10
        assert _get_llm_timeout({ENV_LLM_TIMEOUT: "-5"}) == 10

    def test_get_llm_timeout_invalid_format_defaults(self):
        """Invalid LLM_TIMEOUT_SECONDS format defaults to 10."""
        assert _get_llm_timeout({ENV_LLM_TIMEOUT: "not-a-number"}) == 10
        assert _get_llm_timeout({ENV_LLM_TIMEOUT: "10.5"}) == 10
        assert _get_llm_timeout({ENV_LLM_TIMEOUT: ""}) == 10


class TestModeBasedBehavior:
    """Test explanation mode behavior."""

    def test_deterministic_mode_no_network_calls(self):
        """Deterministic mode makes zero network calls."""
        env = {ENV_EXPLANATION_MODE: MODE_DETERMINISTIC}
        evidence = analyze_mri_evidence("case-001-consistent")
        exp = explain_review_status(evidence)
        packet = generate_review_support(evidence, exp)

        result = generate_enhanced_explanation(packet, env)

        assert result.source == "Deterministic fallback"
        assert "Evidence Review Summary" in result.text

    def test_auto_mode_with_no_enabled_providers(self):
        """Auto mode with no providers enabled falls back to deterministic."""
        env = {
            ENV_EXPLANATION_MODE: MODE_AUTO,
            # No providers enabled
        }
        evidence = analyze_mri_evidence("case-002-change")
        exp = explain_review_status(evidence)
        packet = generate_review_support(evidence, exp)

        result = generate_enhanced_explanation(packet, env)

        assert result.source == "Deterministic fallback"

    def test_auto_mode_with_disabled_bedrock_tries_groq(self):
        """Auto mode skips disabled Bedrock and tries Groq."""
        env = {
            ENV_EXPLANATION_MODE: MODE_AUTO,
            ENV_BEDROCK_ENABLED: "false",
            ENV_GROQ_ENABLED: "true",
            ENV_GROQ_API_KEY: "gsk_test",
            ENV_GROQ_MODEL_ID: "mixtral-8x7b-32768",
        }
        evidence = analyze_mri_evidence("case-001-consistent")
        exp = explain_review_status(evidence)
        packet = generate_review_support(evidence, exp)

        mock_groq_response = MagicMock(
            read=MagicMock(
                return_value=json.dumps({
                    "choices": [{"message": {"content": "Groq explanation."}}]
                }).encode()
            ),
            __enter__=MagicMock(return_value=MagicMock(
                read=MagicMock(
                    return_value=json.dumps({
                        "choices": [{"message": {"content": "Groq explanation."}}]
                    }).encode()
                )
            )),
            __exit__=MagicMock(return_value=None),
        )

        with patch("urllib.request.urlopen", return_value=mock_groq_response):
            result = generate_enhanced_explanation(packet, env)
            assert result.source == "Groq"

    def test_bedrock_requires_all_config_to_attempt(self):
        """Bedrock is skipped if any required config is missing."""
        # Bedrock enabled but no region
        env1 = {ENV_BEDROCK_ENABLED: "true", ENV_BEDROCK_MODEL_ID: "model"}
        assert _call_bedrock(ReviewSupportPacket("x", [], []), env1) is None

        # Bedrock enabled but no model ID
        env2 = {ENV_BEDROCK_ENABLED: "true", ENV_BEDROCK_REGION: "us-east-1"}
        assert _call_bedrock(ReviewSupportPacket("x", [], []), env2) is None

    def test_groq_requires_all_config_to_attempt(self):
        """Groq is skipped if any required config is missing."""
        # Groq enabled but no API key
        env1 = {ENV_GROQ_ENABLED: "true", ENV_GROQ_MODEL_ID: "model"}
        assert _call_groq(ReviewSupportPacket("x", [], []), env1) is None

        # Groq enabled but no model ID
        env2 = {ENV_GROQ_ENABLED: "true", ENV_GROQ_API_KEY: "key"}
        assert _call_groq(ReviewSupportPacket("x", [], []), env2) is None

    def test_custom_bedrock_model_id_used(self):
        """Custom BEDROCK_MODEL_ID is used when provided."""
        env = {
            ENV_BEDROCK_ENABLED: "true",
            ENV_BEDROCK_REGION: "us-west-2",
            ENV_BEDROCK_MODEL_ID: "custom-model-id",
        }
        packet = ReviewSupportPacket("Case X", ["Finding 1"], ["Check 1"])

        mock_response = {
            "body": MagicMock(
                read=MagicMock(
                    return_value=json.dumps({
                        "content": [{"text": "Output from custom model."}]
                    }).encode()
                )
            )
        }

        with patch("boto3.client") as mock_boto3_client:
            mock_client = MagicMock()
            mock_boto3_client.return_value = mock_client
            mock_client.invoke_model.return_value = mock_response

            _call_bedrock(packet, env)

            # Verify custom model ID was used
            call_kwargs = mock_client.invoke_model.call_args[1]
            assert call_kwargs["modelId"] == "custom-model-id"

    def test_custom_groq_model_id_used(self):
        """Custom GROQ_MODEL_ID is used when provided."""
        env = {
            ENV_GROQ_ENABLED: "true",
            ENV_GROQ_API_KEY: "test_key",
            ENV_GROQ_MODEL_ID: "custom-groq-model",
        }
        packet = ReviewSupportPacket("Case X", ["Finding 1"], ["Check 1"])

        mock_groq_response = MagicMock(
            read=MagicMock(
                return_value=json.dumps({
                    "choices": [{"message": {"content": "Output from custom Groq model."}}]
                }).encode()
            ),
            __enter__=MagicMock(return_value=MagicMock(
                read=MagicMock(
                    return_value=json.dumps({
                        "choices": [{"message": {"content": "Output from custom Groq model."}}]
                    }).encode()
                )
            )),
            __exit__=MagicMock(return_value=None),
        )

        with patch("urllib.request.urlopen", return_value=mock_groq_response) as mock_urlopen:
            _call_groq(packet, env)

            # Verify custom model ID was used in request
            call_args = mock_urlopen.call_args[0][0]
            body = json.loads(call_args.data.decode())
            assert body["model"] == "custom-groq-model"

    def test_timeout_applied_to_groq_call(self):
        """Custom LLM_TIMEOUT_SECONDS is applied to Groq calls."""
        env = {
            ENV_GROQ_ENABLED: "true",
            ENV_GROQ_API_KEY: "test_key",
            ENV_GROQ_MODEL_ID: "model",
            ENV_LLM_TIMEOUT: "25",
        }
        packet = ReviewSupportPacket("Case X", ["Finding 1"], ["Check 1"])

        with patch("urllib.request.urlopen") as mock_urlopen:
            _call_groq(packet, env)

            # Verify timeout was passed
            call_args = mock_urlopen.call_args
            assert call_args[1]["timeout"] == 25
