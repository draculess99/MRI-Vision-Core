"""Simulated human-review workflow gate.

No actual clinician approval is implied by any state transition here -- this exists to
demonstrate a review-state UI and audit trail, matching the pattern already used by the
OpenCV demo's "simulated workflow" review control, but as shared, reusable infrastructure.

The audit trail is append-only: history entries are never edited or removed, only
appended via transition(). Deterministic evidence/consensus computation (specialist,
critic, consensus modules) never reads a ReviewGate's state, so review-state changes can
never feed back into or alter those results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

from .contracts import ReviewState

VALID_TRANSITIONS: Dict[ReviewState, FrozenSet[ReviewState]] = {
    ReviewState.PENDING_REVIEW: frozenset({ReviewState.MORE_EVIDENCE_REQUESTED, ReviewState.REVIEWED}),
    ReviewState.MORE_EVIDENCE_REQUESTED: frozenset({ReviewState.PENDING_REVIEW, ReviewState.REVIEWED}),
    ReviewState.REVIEWED: frozenset(),  # terminal in this simulation; no reopening
}


class ReviewGateError(Exception):
    """Base class for review-gate failures."""


class InvalidTransitionError(ReviewGateError):
    """Raised when a transition is not permitted from the gate's current state."""


class StaleEvidenceError(ReviewGateError):
    """Raised when a transition is attempted against evidence that no longer matches
    the fingerprint the gate was created with -- the underlying evidence bundle changed
    (e.g. a different inference run) and this decision must not be applied to it."""


class InvalidActorError(ReviewGateError):
    """Raised when actor is missing, empty, or not a string."""


class InvalidReasonError(ReviewGateError):
    """Raised when reason is missing, empty, or not a string."""


MAX_ACTOR_LENGTH = 200
MAX_REASON_LENGTH = 2000


@dataclass(frozen=True)
class ReviewAuditEntry:
    actor: str
    timestamp: float
    previous_state: ReviewState
    new_state: ReviewState
    reason: str
    evidence_fingerprint: str

    def to_dict(self) -> dict:
        return {
            "actor": self.actor,
            "timestamp": self.timestamp,
            "previous_state": self.previous_state.value,
            "new_state": self.new_state.value,
            "reason": self.reason,
            "evidence_fingerprint": self.evidence_fingerprint,
        }


class ReviewGate:
    """One study's simulated review workflow, scoped to one evidence fingerprint."""

    def __init__(self, evidence_fingerprint: str):
        self._fingerprint = evidence_fingerprint
        self._state = ReviewState.PENDING_REVIEW
        self._history: list[ReviewAuditEntry] = []

    @property
    def evidence_fingerprint(self) -> str:
        return self._fingerprint

    @property
    def state(self) -> ReviewState:
        return self._state

    @property
    def history(self) -> Tuple[ReviewAuditEntry, ...]:
        """Append-only audit trail, oldest first."""
        return tuple(self._history)

    def transition(
        self,
        new_state: ReviewState,
        *,
        actor: str,
        reason: str,
        evidence_fingerprint: str,
    ) -> ReviewAuditEntry:
        """Attempt a state transition. Raises StaleEvidenceError if evidence_fingerprint
        does not match this gate's fingerprint, InvalidTransitionError if new_state is
        not reachable from the current state, or InvalidActorError/InvalidReasonError
        if actor/reason are missing, empty, non-string, or exceed their length bounds.
        Validation runs before the transition/fingerprint checks so a malformed actor
        or reason is always rejected regardless of what else is wrong with the call.
        On success, appends to history and returns the new entry; history is never
        mutated afterward.
        """
        if not isinstance(actor, str) or not actor.strip():
            raise InvalidActorError("actor must be a non-empty string identifying who/what requested this transition.")
        if len(actor) > MAX_ACTOR_LENGTH:
            raise InvalidActorError(f"actor exceeds max length {MAX_ACTOR_LENGTH}.")
        if not isinstance(reason, str) or not reason.strip():
            raise InvalidReasonError("reason must be a non-empty string explaining this transition.")
        if len(reason) > MAX_REASON_LENGTH:
            raise InvalidReasonError(f"reason exceeds max length {MAX_REASON_LENGTH}.")

        if evidence_fingerprint != self._fingerprint:
            raise StaleEvidenceError(
                f"Evidence fingerprint {evidence_fingerprint!r} does not match this gate's "
                f"fingerprint {self._fingerprint!r}; the underlying evidence has changed "
                "since this review gate was created."
            )
        allowed = VALID_TRANSITIONS.get(self._state, frozenset())
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {self._state.value} to {new_state.value}; "
                f"allowed: {sorted(s.value for s in allowed)}."
            )
        entry = ReviewAuditEntry(
            actor=actor,
            timestamp=time.time(),
            previous_state=self._state,
            new_state=new_state,
            reason=reason,
            evidence_fingerprint=evidence_fingerprint,
        )
        self._history.append(entry)
        self._state = new_state
        return entry
