"""Tests for Amazon Alexa+ human approval safeguard.

Ensures the human-approval state is non-bypassable:
- Copilot cannot approve cases
- Only named human reviewers can transition approval state
- All cases require human review
"""

import pytest

from hackathon.amazon_alexa.tools import (
    ApprovalState,
    ReviewStatus,
    analyze_mri_evidence,
    request_human_approval,
)


class TestApprovalSafeguard:
    """Test that human approval cannot be bypassed."""

    def test_all_cases_require_approval(self):
        """Every case requires human approval."""
        for case_id in ["case-001-consistent", "case-002-change", "case-003-uncertain"]:
            evidence = analyze_mri_evidence(case_id)
            approval = request_human_approval(case_id, "Dr. Reviewer")
            assert approval.approval_state == ApprovalState.PENDING
            # Copilot cannot transition to APPROVED
            assert approval.approval_state != ApprovalState.APPROVED

    def test_consistent_case_still_requires_approval(self):
        """Even CONSISTENT status requires human approval."""
        evidence = analyze_mri_evidence("case-001-consistent")
        assert evidence.status == ReviewStatus.CONSISTENT
        approval = request_human_approval(evidence.case_id, "Dr. Smith")
        assert approval.approval_state == ApprovalState.PENDING
        # The copilot does not have a method to approve

    def test_copilot_cannot_approve(self):
        """The copilot tool layer cannot approve cases."""
        approval = request_human_approval("case-001-consistent", "Dr. Smith")
        # ApprovalRequest has no approve() method on the copilot side
        # Only PENDING state is accessible
        assert approval.approval_state == ApprovalState.PENDING
        assert not hasattr(approval, "approve") or callable(approval.approve) is False

    def test_reviewer_name_required_for_approval_request(self):
        """A named human reviewer is required to create an approval request."""
        with pytest.raises(ValueError):
            request_human_approval("case-001-consistent", "")
        with pytest.raises(ValueError):
            request_human_approval("case-001-consistent", "   ")

    def test_approval_state_immutable(self):
        """ApprovalRequest approval_state cannot be changed by copilot."""
        approval = request_human_approval("case-001-consistent", "Dr. Jones")
        original_state = approval.approval_state
        # Try to mutate (will fail because ApprovalRequest is frozen)
        with pytest.raises(AttributeError):
            approval.approval_state = ApprovalState.APPROVED
        assert approval.approval_state == original_state

    def test_review_needed_cases_require_approval(self):
        """Cases that need review still require human approval."""
        evidence = analyze_mri_evidence("case-002-change")
        assert evidence.status == ReviewStatus.NEEDS_REVIEW
        approval = request_human_approval(evidence.case_id, "Dr. Lee")
        assert approval.approval_state == ApprovalState.PENDING

    def test_uncertain_cases_require_approval(self):
        """Cases with uncertainty still require human approval."""
        evidence = analyze_mri_evidence("case-003-uncertain")
        assert evidence.status == ReviewStatus.UNCERTAIN
        approval = request_human_approval(evidence.case_id, "Dr. Brown")
        assert approval.approval_state == ApprovalState.PENDING

    def test_approval_request_tracks_reviewer(self):
        """Approval request records the reviewer but doesn't grant approval."""
        reviewer_name = "Dr. Specific Reviewer"
        approval = request_human_approval("case-001-consistent", reviewer_name)
        assert approval.reviewer_name == reviewer_name
        assert approval.approval_state == ApprovalState.PENDING
        # Recording the reviewer name does not grant approval

    def test_comment_does_not_affect_approval_state(self):
        """Reviewer comment does not affect approval state."""
        a1 = request_human_approval("case-001-consistent", "Dr. X", "")
        a2 = request_human_approval("case-001-consistent", "Dr. X", "detailed review")
        # Both have same approval state despite different comments
        assert a1.approval_state == a2.approval_state == ApprovalState.PENDING
