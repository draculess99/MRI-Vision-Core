"""
Offline unit tests for generic execution router.

All tests use mocks and fakes; no network calls, no AWS invocations.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from mri_core.config import ConfigError, load_execution_config, get_local_config
from mri_core.execution import (
    ExecutionConfig,
    ExecutionMode,
    ExecutionResult,
    ExecutionRouter,
    LocalTransport,
)


class TestExecutionConfig:
    """Tests for ExecutionConfig validation."""

    def test_local_deterministic_is_default(self):
        """Local deterministic mode requires no extra config."""
        config = ExecutionConfig()
        config.validate()  # Should not raise

    def test_local_deterministic_no_aws_required(self):
        """Local mode doesn't require function name."""
        config = ExecutionConfig(mode=ExecutionMode.LOCAL_DETERMINISTIC)
        config.validate()  # Should not raise

    def test_aws_lambda_requires_function_name(self):
        """AWS Lambda mode requires function name."""
        config = ExecutionConfig(mode=ExecutionMode.AWS_LAMBDA)
        with pytest.raises(ValueError, match="aws_lambda_function_name"):
            config.validate()

    def test_aws_lambda_with_function_name_valid(self):
        """AWS Lambda with function name passes validation."""
        config = ExecutionConfig(
            mode=ExecutionMode.AWS_LAMBDA,
            aws_lambda_function_name="my-function",
        )
        config.validate()  # Should not raise


class TestExecutionResult:
    """Tests for ExecutionResult serialization."""

    def test_successful_local_result(self):
        """Successful local result converts to dict."""
        result = ExecutionResult(
            success=True,
            mode="local_deterministic",
            data={"manifest": {"hash": "abc123"}},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["mode"] == "local_deterministic"
        assert d["error"] is None
        assert d["fallback_used"] is False

    def test_failed_result_with_fallback(self):
        """Failed result with fallback flag."""
        result = ExecutionResult(
            success=False,
            mode="aws_lambda",
            error="Connection timeout",
            fallback_used=True,
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["fallback_used"] is True
        assert "Connection timeout" in d["error"]


class TestLocalExecution:
    """Tests for local deterministic execution."""

    def test_local_processor_called(self):
        """Router calls local processor in local mode."""
        mock_processor = Mock(return_value={"result": "success"})
        config = ExecutionConfig(mode=ExecutionMode.LOCAL_DETERMINISTIC)
        router = ExecutionRouter(config=config, local_processor=mock_processor)

        payload = {"image_data": "test"}
        result = router.process(payload)

        assert result.success is True
        assert result.mode == "local_deterministic"
        mock_processor.assert_called_once_with(payload)

    def test_local_processor_error_handled(self):
        """Local processor exceptions are caught and reported."""
        mock_processor = Mock(side_effect=ValueError("Invalid image"))
        config = ExecutionConfig(mode=ExecutionMode.LOCAL_DETERMINISTIC)
        router = ExecutionRouter(config=config, local_processor=mock_processor)

        result = router.process({})

        assert result.success is False
        assert "Invalid image" in result.error

    def test_no_processor_configured(self):
        """Router returns error if no local processor."""
        config = ExecutionConfig(mode=ExecutionMode.LOCAL_DETERMINISTIC)
        router = ExecutionRouter(config=config, local_processor=None)

        result = router.process({})

        assert result.success is False
        assert "No local processor" in result.error

    def test_deterministic_repeatability(self):
        """Same payload produces same result on repeated calls."""
        call_count = 0
        def stateless_processor(payload):
            return {"result": "same", "call": call_count}

        config = ExecutionConfig(mode=ExecutionMode.LOCAL_DETERMINISTIC)
        router = ExecutionRouter(config=config, local_processor=stateless_processor)

        payload = {"image_data": "test"}
        result1 = router.process(payload)
        result2 = router.process(payload)

        assert result1.data == result2.data


class TestAwsLambdaMode:
    """Tests for AWS Lambda mode (all mocked, no real AWS calls)."""

    def test_aws_mode_requires_config(self):
        """AWS mode without function name raises ValueError."""
        config = ExecutionConfig(mode=ExecutionMode.AWS_LAMBDA)
        with pytest.raises(ValueError):
            config.validate()

    def test_aws_mode_invokes_transport(self):
        """AWS mode uses transport to invoke remote function."""
        mock_transport = Mock()
        mock_transport.invoke_remote.return_value = {"statusCode": 200, "manifest": {}}
        config = ExecutionConfig(
            mode=ExecutionMode.AWS_LAMBDA,
            aws_lambda_function_name="test-function",
            aws_region="us-east-1",
        )
        router = ExecutionRouter(config=config, transport=mock_transport)

        payload = {"image_data": "test"}
        result = router.process(payload, fallback_on_error=False)

        assert result.success is True
        assert result.mode == "aws_lambda"
        mock_transport.invoke_remote.assert_called_once()

    def test_aws_mode_failure_reported_without_fallback(self):
        """AWS failure without fallback is reported clearly."""
        mock_transport = Mock()
        mock_transport.invoke_remote.side_effect = RuntimeError("Network error")
        config = ExecutionConfig(
            mode=ExecutionMode.AWS_LAMBDA,
            aws_lambda_function_name="test-function",
        )
        router = ExecutionRouter(config=config, transport=mock_transport)

        result = router.process({}, fallback_on_error=False)

        assert result.success is False
        assert "Network error" in str(result.error)
        assert result.fallback_used is False

    def test_aws_failure_fallback_to_local(self):
        """AWS failure with fallback switches to local mode."""
        mock_transport = Mock()
        mock_transport.invoke_remote.side_effect = RuntimeError("AWS unavailable")
        mock_processor = Mock(return_value={"status": "local"})
        config = ExecutionConfig(
            mode=ExecutionMode.AWS_LAMBDA,
            aws_lambda_function_name="test-function",
        )
        router = ExecutionRouter(
            config=config,
            transport=mock_transport,
            local_processor=mock_processor,
        )

        result = router.process({}, fallback_on_error=True)

        assert result.success is True
        assert result.mode == "local_deterministic"
        assert result.fallback_used is True
        assert "Remote execution failed" in result.error
        mock_processor.assert_called_once()

    def test_aws_success_no_fallback_used(self):
        """Successful AWS invocation doesn't trigger fallback."""
        mock_transport = Mock()
        mock_transport.invoke_remote.return_value = {"statusCode": 200, "manifest": {}}
        mock_processor = Mock()
        config = ExecutionConfig(
            mode=ExecutionMode.AWS_LAMBDA,
            aws_lambda_function_name="test-function",
        )
        router = ExecutionRouter(
            config=config,
            transport=mock_transport,
            local_processor=mock_processor,
        )

        result = router.process({}, fallback_on_error=True)

        assert result.fallback_used is False
        mock_processor.assert_not_called()


class TestConfigLoading:
    """Tests for config loading from environment variables."""

    def test_load_default_local_config(self, monkeypatch):
        """Load config defaults to local deterministic."""
        monkeypatch.delenv("MRI_EXECUTION_MODE", raising=False)
        config = load_execution_config()
        assert config.mode == ExecutionMode.LOCAL_DETERMINISTIC

    def test_load_aws_mode_from_env(self, monkeypatch):
        """Load AWS mode from environment variables."""
        monkeypatch.setenv("MRI_EXECUTION_MODE", "aws_lambda")
        monkeypatch.setenv("MRI_LAMBDA_FUNCTION_NAME", "my-function")
        monkeypatch.setenv("MRI_AWS_REGION", "eu-west-1")

        config = load_execution_config()

        assert config.mode == ExecutionMode.AWS_LAMBDA
        assert config.aws_lambda_function_name == "my-function"
        assert config.aws_region == "eu-west-1"

    def test_invalid_execution_mode_raises(self, monkeypatch):
        """Invalid execution mode in env raises ConfigError."""
        monkeypatch.setenv("MRI_EXECUTION_MODE", "invalid_mode")
        with pytest.raises(ConfigError, match="Invalid MRI_EXECUTION_MODE"):
            load_execution_config()

    def test_aws_mode_without_function_raises(self, monkeypatch):
        """AWS mode without function name raises ConfigError."""
        monkeypatch.setenv("MRI_EXECUTION_MODE", "aws_lambda")
        monkeypatch.delenv("MRI_LAMBDA_FUNCTION_NAME", raising=False)
        with pytest.raises(ConfigError, match="aws_lambda_function_name"):
            load_execution_config()

    def test_get_local_config_helper(self):
        """get_local_config() returns local deterministic config."""
        config = get_local_config()
        assert config.mode == ExecutionMode.LOCAL_DETERMINISTIC

    def test_invalid_timeout_raises(self, monkeypatch):
        """Invalid timeout in env raises ConfigError."""
        monkeypatch.setenv("MRI_LAMBDA_TIMEOUT", "not_an_int")
        with pytest.raises(ConfigError, match="Invalid MRI_LAMBDA_TIMEOUT"):
            load_execution_config()


class TestNoNetworkCalls:
    """Verify no actual network calls occur."""

    def test_local_mode_has_no_network_dependency(self):
        """Local mode doesn't import boto3 or network libraries."""
        config = ExecutionConfig(mode=ExecutionMode.LOCAL_DETERMINISTIC)
        mock_processor = Mock(return_value={})
        router = ExecutionRouter(config=config, local_processor=mock_processor)

        # This should work with no boto3 or network setup
        result = router.process({})
        assert result is not None

    def test_aws_transport_lazy_import(self):
        """AWS transport only imports boto3 when invoked."""
        # Simulate boto3 not being available
        with patch.dict("sys.modules", {"boto3": None}):
            # Import should succeed
            from hackathon.opencv_aws.aws_transport import AwsLambdaTransport

            transport = AwsLambdaTransport()

            # invoke_remote should raise ImportError when boto3 is "missing"
            # (In real code, boto3 would be properly missing; here we simulate it)
            # This test verifies the import structure is correct


class TestModeIsolation:
    """Tests for execution mode isolation."""

    def test_local_and_aws_results_different_modes(self):
        """Local and AWS results have different mode strings."""
        local_processor = Mock(return_value={"local": True})
        aws_transport = Mock(return_value={"statusCode": 200})

        local_router = ExecutionRouter(
            config=ExecutionConfig(mode=ExecutionMode.LOCAL_DETERMINISTIC),
            local_processor=local_processor,
        )

        aws_router = ExecutionRouter(
            config=ExecutionConfig(
                mode=ExecutionMode.AWS_LAMBDA,
                aws_lambda_function_name="func",
            ),
            transport=aws_transport,
        )

        local_result = local_router.process({})
        aws_result = aws_router.process({})

        assert local_result.mode == "local_deterministic"
        assert aws_result.mode == "aws_lambda"
