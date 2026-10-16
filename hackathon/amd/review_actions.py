"""Three named human-review actions on top of the shared `mri_core.agents.review_gate.ReviewGate`.

The shared `ReviewGate` (not modified here) only knows three states: PENDING_REVIEW,
MORE_EVIDENCE_REQUESTED, REVIEWED -- it has no separate "escalated" state. Rather than add
one to shared mri_core code (out of scope for this app), "escalate" is implemented here as a
transition to REVIEWED whose audit `reason` is always prefixed "ESCALATED:", so it is
distinguishable from a plain approval in the audit trail without touching the shared contract.
This is a deliberate, documented limitation -- see hackathon/amd/README.md.

Every action goes through `ReviewGate.transition()` unchanged: invalid transitions (e.g.
approving an already-REVIEWED gate) and stale-evidence mismatches still raise exactly as they
do there. Nothing here ever marks a study reviewed/approved automatically -- every transition
requires an explicit actor and reason supplied by the caller (the Streamlit UI).
"""

from __future__ import annotations

from mri_core.agents.contracts import ReviewState
from mri_core.agents.review_gate import ReviewAuditEntry, ReviewGate

ESCALATE_REASON_PREFIX = "ESCALATED: "


def approve(gate: ReviewGate, *, actor: str, reason: str) -> ReviewAuditEntry:
    """Move the gate to REVIEWED. No clinician approval is implied -- see ReviewGate's own
    docstring; this only records that a named actor took this action, with their reason."""
    return gate.transition(
        ReviewState.REVIEWED, actor=actor, reason=reason, evidence_fingerprint=gate.evidence_fingerprint
    )


def request_more_evidence(gate: ReviewGate, *, actor: str, reason: str) -> ReviewAuditEntry:
    """Move the gate to MORE_EVIDENCE_REQUESTED."""
    return gate.transition(
        ReviewState.MORE_EVIDENCE_REQUESTED, actor=actor, reason=reason, evidence_fingerprint=gate.evidence_fingerprint
    )


def escalate(gate: ReviewGate, *, actor: str, reason: str) -> ReviewAuditEntry:
    """Move the gate to REVIEWED with an "ESCALATED:" reason prefix.

    The shared ReviewGate has no distinct escalated state (see module docstring), so this is
    the same underlying transition as approve() -- the two are distinguished only by the
    audit-trail reason text and actor, never silently merged or hidden from the history.
    """
    return gate.transition(
        ReviewState.REVIEWED, actor=actor, reason=f"{ESCALATE_REASON_PREFIX}{reason}",
        evidence_fingerprint=gate.evidence_fingerprint,
    )


def is_escalation(entry: ReviewAuditEntry) -> bool:
    """True if this audit entry was recorded by escalate() rather than approve()."""
    return entry.new_state == ReviewState.REVIEWED and entry.reason.startswith(ESCALATE_REASON_PREFIX)
