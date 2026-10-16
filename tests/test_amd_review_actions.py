"""Tests for hackathon/amd/review_actions.py: approve/request-more-evidence/escalate on top
of the shared, unmodified mri_core.agents.review_gate.ReviewGate.
"""

import pytest

from hackathon.amd.review_actions import approve, escalate, is_escalation, request_more_evidence
from mri_core.agents.contracts import ReviewState
from mri_core.agents.review_gate import InvalidTransitionError, ReviewGate, StaleEvidenceError


FINGERPRINT = "deadbeef" * 8


def _gate() -> ReviewGate:
    return ReviewGate(FINGERPRINT)


class TestApprove:
    def test_moves_pending_to_reviewed(self):
        gate = _gate()
        entry = approve(gate, actor="reviewer_a", reason="looks complete")
        assert gate.state == ReviewState.REVIEWED
        assert entry.previous_state == ReviewState.PENDING_REVIEW
        assert entry.new_state == ReviewState.REVIEWED
        assert not is_escalation(entry)

    def test_cannot_approve_twice(self):
        gate = _gate()
        approve(gate, actor="a", reason="r")
        with pytest.raises(InvalidTransitionError):
            approve(gate, actor="a", reason="r2")


class TestRequestMoreEvidence:
    def test_moves_pending_to_more_evidence_requested(self):
        gate = _gate()
        entry = request_more_evidence(gate, actor="reviewer_b", reason="need another series")
        assert gate.state == ReviewState.MORE_EVIDENCE_REQUESTED
        assert entry.new_state == ReviewState.MORE_EVIDENCE_REQUESTED

    def test_can_return_to_pending_then_be_approved(self):
        gate = _gate()
        request_more_evidence(gate, actor="a", reason="r1")
        gate.transition(
            ReviewState.PENDING_REVIEW, actor="a", reason="evidence added",
            evidence_fingerprint=gate.evidence_fingerprint,
        )
        approve(gate, actor="a", reason="now complete")
        assert gate.state == ReviewState.REVIEWED


class TestEscalate:
    def test_moves_pending_to_reviewed_with_escalated_prefix(self):
        gate = _gate()
        entry = escalate(gate, actor="reviewer_c", reason="needs senior sign-off")
        assert gate.state == ReviewState.REVIEWED
        assert entry.reason == "ESCALATED: needs senior sign-off"
        assert is_escalation(entry)

    def test_distinguishable_from_plain_approval_in_history(self):
        gate = _gate()
        escalate(gate, actor="a", reason="r")
        [entry] = gate.history
        assert is_escalation(entry)

        other_gate = _gate()
        approve(other_gate, actor="a", reason="r")
        [other_entry] = other_gate.history
        assert not is_escalation(other_entry)

    def test_cannot_escalate_an_already_reviewed_gate(self):
        gate = _gate()
        approve(gate, actor="a", reason="r")
        with pytest.raises(InvalidTransitionError):
            escalate(gate, actor="a", reason="too late")


class TestAuditTrailIsAppendOnlyAndHonorsStaleEvidence:
    def test_history_grows_by_exactly_one_per_action_and_is_never_reordered(self):
        gate = _gate()
        request_more_evidence(gate, actor="a", reason="r1")
        gate.transition(
            ReviewState.PENDING_REVIEW, actor="a", reason="r2", evidence_fingerprint=gate.evidence_fingerprint
        )
        approve(gate, actor="a", reason="r3")
        assert len(gate.history) == 3
        assert [e.reason for e in gate.history] == ["r1", "r2", "r3"]

    def test_action_against_a_different_fingerprint_is_rejected(self):
        gate = _gate()
        with pytest.raises(StaleEvidenceError):
            gate.transition(
                ReviewState.REVIEWED, actor="a", reason="r", evidence_fingerprint="some-other-fingerprint",
            )

    def test_every_action_requires_a_non_empty_reason_and_actor(self):
        gate = _gate()
        with pytest.raises(Exception):
            approve(gate, actor="", reason="r")
        with pytest.raises(Exception):
            approve(gate, actor="a", reason="")
