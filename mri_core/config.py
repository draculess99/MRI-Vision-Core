"""
Safe configuration loading for execution router.

Loads configuration from environment variables only.
No secrets or account IDs in code.
"""

import os
import sys
from pathlib import Path

# Ensure mri_core.execution can be imported without loading heavy mri_core.__init__
from mri_core.execution import ExecutionConfig, ExecutionMode


class ConfigError(Exception):
    """Configuration validation error."""


def load_execution_config() -> ExecutionConfig:
    """
    Load execution configuration from environment variables.

    Supported environment variables:
    - MRI_EXECUTION_MODE: "local_deterministic" (default) or "aws_lambda"
    - MRI_AWS_REGION: AWS region (default: "us-east-1")
    - MRI_LAMBDA_FUNCTION_NAME: Lambda function name (required if aws_lambda mode)
    - MRI_LAMBDA_TIMEOUT: Timeout in seconds (default: 30)

    Returns:
        ExecutionConfig instance

    Raises:
        ConfigError: If configuration is invalid
    """
    mode_str = os.getenv("MRI_EXECUTION_MODE", "local_deterministic").lower()

    try:
        mode = ExecutionMode(mode_str)
    except ValueError:
        raise ConfigError(
            f"Invalid MRI_EXECUTION_MODE: {mode_str}. "
            f"Must be one of: {', '.join(m.value for m in ExecutionMode)}"
        )

    aws_region = os.getenv("MRI_AWS_REGION", "us-east-1")
    lambda_function = os.getenv("MRI_LAMBDA_FUNCTION_NAME")
    timeout_str = os.getenv("MRI_LAMBDA_TIMEOUT", "30")

    try:
        timeout = int(timeout_str)
    except ValueError:
        raise ConfigError(f"Invalid MRI_LAMBDA_TIMEOUT: {timeout_str} (must be integer)")

    config = ExecutionConfig(
        mode=mode,
        aws_region=aws_region,
        aws_lambda_function_name=lambda_function,
        timeout_seconds=timeout,
    )

    try:
        config.validate()
    except ValueError as e:
        raise ConfigError(str(e))

    return config


def get_local_config() -> ExecutionConfig:
    """Get default local deterministic configuration."""
    return ExecutionConfig(mode=ExecutionMode.LOCAL_DETERMINISTIC)
