"""Data contracts for deterministic evidence review.

Non-diagnostic, research-only. These contracts define the input/output shape
of the Evidence Review Agent and capture the auditable trace from observation
to decision to next action.
"""

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json


class ReviewOutcome(str, Enum):
    """Deterministic agent decision outcome. Research-workflow classification only."""

    FINALIZE_EVIDENCE_BUNDLE = "FINALIZE_EVIDENCE_BUNDLE"
    REQUEST_MORE_EVIDENCE = "REQUEST_MORE_EVIDENCE"
    PENDING_HUMAN_APPROVAL = "PENDING_HUMAN_APPROVAL"


@dataclass(frozen=True)
class ToolStep:
    """Record of a deterministic review-tool execution."""

    tool_name: str  # validate_manifest, assess_change_guard, choose_next_action
    status: str  # "PASS" or "FLAG"
    findings: Tuple[str, ...] = ()  # Observations from this step


@dataclass(frozen=True)
class ReviewAction:
    """Prescribed next action, non-diagnostic."""

    action_type: str  # "finalize", "request_input", "halt_pending_approval"
    user_facing_text: str
    is_recoverable: bool  # False means user must intervene; True means resolvable by user action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "user_facing_text": self.user_facing_text,
            "is_recoverable": self.is_recoverable,
        }


@dataclass(frozen=True)
class AgentTrace:
    """Auditable decision trace: observation → tool steps → outcome."""

    trace_version: str = "1.0"
    source_hash_sha256: str = ""
    observations: Dict[str, Any] = field(default_factory=dict)
    tool_steps: Tuple[ToolStep, ...] = ()
    outcome: str = ""
    safety_notice: str = (
        "Research prototype. Non-diagnostic. This is processing metadata only. "
        "No medical claims made or permitted."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_version": self.trace_version,
            "source_hash_sha256": self.source_hash_sha256,
            "observations": self.observations,
            "tool_steps": [
                {
                    "tool_name": s.tool_name,
                    "status": s.status,
                    "findings": list(s.findings),
                }
                for s in self.tool_steps
            ],
            "outcome": self.outcome,
            "safety_notice": self.safety_notice,
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class AgentDecision:
    """Final decision from the Evidence Review Agent."""

    outcome: ReviewOutcome
    reasoning: str
    next_action: ReviewAction
    human_approval_required: bool
    trace: AgentTrace
    safety_notice: str = (
        "Research prototype. Non-diagnostic. This is processing metadata only. "
        "No medical claims made or permitted."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "reasoning": self.reasoning,
            "next_action": self.next_action.to_dict(),
            "human_approval_required": self.human_approval_required,
            "safety_notice": self.safety_notice,
            "trace": self.trace.to_dict(),
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
