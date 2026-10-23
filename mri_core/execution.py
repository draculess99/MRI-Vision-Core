"""
Generic reusable execution router for MRI Vision pipeline.

Supports local-deterministic (default, no network) and optional
remote execution (aws_lambda) modes with dependency injection
for testability and transport abstraction.

Platform-independent: Nebius, AWS, AMD, TechEx can reuse this
execution model without being forced to any specific backend.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ExecutionMode(Enum):
    """Execution target mode."""

    LOCAL_DETERMINISTIC = "local_deterministic"
    AWS_LAMBDA = "aws_lambda"


@dataclass
class ExecutionConfig:
    """Configuration for execution routing."""

    mode: ExecutionMode = ExecutionMode.LOCAL_DETERMINISTIC
    aws_region: str = "us-east-1"
    aws_lambda_function_name: Optional[str] = None
    timeout_seconds: int = 30

    def validate(self) -> None:
        """Validate configuration based on mode."""
        if self.mode == ExecutionMode.AWS_LAMBDA:
            if not self.aws_lambda_function_name:
                raise ValueError(
                    "aws_lambda_function_name required for AWS_LAMBDA mode"
                )


class ExecutionResult:
    """Standardized execution result."""

    def __init__(
        self,
        success: bool,
        mode: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        fallback_used: bool = False,
    ):
        self.success = success
        self.mode = mode
        self.data = data or {}
        self.error = error
        self.fallback_used = fallback_used

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "success": self.success,
            "mode": self.mode,
            "data": self.data,
            "error": self.error,
            "fallback_used": self.fallback_used,
        }


class Transport(ABC):
    """Abstract transport interface for dependency injection."""

    @abstractmethod
    def invoke_remote(
        self, function_name: str, payload: Dict[str, Any], region: str
    ) -> Dict[str, Any]:
        """
        Invoke remote function.

        Args:
            function_name: Name of remote function
            payload: Input payload
            region: Cloud region

        Returns:
            Response dict with statusCode and data

        Raises:
            Exception on any network or invocation error
        """


class LocalTransport(Transport):
    """Local no-op transport (always fails, triggers fallback)."""

    def invoke_remote(
        self, function_name: str, payload: Dict[str, Any], region: str
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "LocalTransport does not support remote invocation"
        )


class ExecutionRouter:
    """
    Route image processing to local or remote execution.

    Generic router supporting any local processor and any remote
    backend via transport abstraction. Preserves deterministic
    offline fallback as first-class behavior.
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        transport: Optional[Transport] = None,
        local_processor: Optional[Any] = None,
    ):
        """
        Initialize router.

        Args:
            config: ExecutionConfig (defaults to local_deterministic)
            transport: Transport implementation (defaults to LocalTransport)
            local_processor: Callable(payload) -> result_dict for local execution
        """
        self.config = config or ExecutionConfig()
        self.config.validate()

        self.transport = transport or LocalTransport()
        self.local_processor = local_processor

    def process(
        self,
        payload: Dict[str, Any],
        fallback_on_error: bool = True,
    ) -> ExecutionResult:
        """
        Route execution based on config.

        Args:
            payload: Input dict for processing
            fallback_on_error: If remote fails, fall back to local deterministic

        Returns:
            ExecutionResult with success, mode, data, error, fallback_used flags
        """
        if self.config.mode == ExecutionMode.LOCAL_DETERMINISTIC:
            return self._execute_local(payload)

        elif self.config.mode == ExecutionMode.AWS_LAMBDA:
            try:
                return self._execute_remote(payload)
            except Exception as e:
                if fallback_on_error:
                    error_msg = f"Remote execution failed: {str(e)}. Falling back to local deterministic."
                    result = self._execute_local(payload)
                    result.fallback_used = True
                    result.error = error_msg
                    return result
                else:
                    return ExecutionResult(
                        success=False,
                        mode="aws_lambda",
                        error=str(e),
                        fallback_used=False,
                    )

    def _execute_local(self, payload: Dict[str, Any]) -> ExecutionResult:
        """Execute using local processor."""
        if not self.local_processor:
            return ExecutionResult(
                success=False,
                mode="local_deterministic",
                error="No local processor configured",
            )

        try:
            result_data = self.local_processor(payload)
            return ExecutionResult(
                success=True,
                mode="local_deterministic",
                data=result_data,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                mode="local_deterministic",
                error=str(e),
            )

    def _execute_remote(self, payload: Dict[str, Any]) -> ExecutionResult:
        """Execute using remote transport."""
        try:
            response = self.transport.invoke_remote(
                self.config.aws_lambda_function_name,
                payload,
                self.config.aws_region,
            )

            success = response.get("statusCode") == 200
            return ExecutionResult(
                success=success,
                mode="aws_lambda",
                data=response,
                error=None if success else response.get("error"),
            )
        except Exception as e:
            raise
