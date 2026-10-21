"""Conversational command router for the simulated Alexa+ MRI copilot.

Maps transcribed or typed text to the existing deterministic tools in
`tools.py`. Every response is built from those tools' real outputs for the
selected demonstration case; nothing is generated or inferred.

Safety: this module has no code path that produces APPROVED_BY_HUMAN. Approval
phrases ("approve this MRI") are refused, and "request human review" can only
create a PENDING request for a reviewer named explicitly (not via speech).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from hackathon.amazon_alexa.tools import (
    NON_DIAGNOSTIC_NOTICE,
    ApprovalRequest,
    ApprovalState,
    MRIEvidence,
    analyze_mri_evidence,
    explain_review_status,
    request_human_approval,
)


class Intent(str, Enum):
    ANALYZE = "analyze"
    EXPLAIN = "explain_status"
    CHANGES = "what_changed"
    REPORT = "evidence_report"
    REQUEST_REVIEW = "request_human_review"
    APPROVAL_STATUS = "approval_status"
    APPROVAL_REFUSED = "approval_refused"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CommandResult:
    intent: Intent
    spoken: str  # short text suitable for speech
    display: str  # full text/markdown shown in the UI
    case_id: str
    ok: bool = True
    approval_state: ApprovalState = ApprovalState.PENDING


@dataclass
class VoiceSession:
    """Per-session state: pending review requests and an audit trail."""

    requests: Dict[str, ApprovalRequest] = field(default_factory=dict)
    audit: List[dict] = field(default_factory=list)

    def log(self, event: str, case_id: str, **details: object) -> None:
        self.audit.append(
            {
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "event": event,
                "case_id": case_id,
                **details,
            }
        )


_SUPPORTED = (
    '"Analyze this MRI", "Explain the review status", "What changed between the images?", '
    '"Generate an evidence report", "Request human review", "What is the approval status?"'
)

# Order matters: approval-status questions and approval attempts are classified
# before everything else so they cannot be mistaken for a benign command.
_STATUS_Q = re.compile(
    r"\b(approval|approved|sign[- ]?off)\b.*\b(status|state)\b"
    r"|\b(status|state)\b.*\b(approval|approved|sign[- ]?off)\b"
    r"|\b(is|has|was)\b.*\b(it|this|the (case|mri|scan))\b.*\bapproved\b"
    r"|\bbeen approved\b|\bpending approval\b",
)
_APPROVE_ATTEMPT = re.compile(
    r"\b(approve|approves|approved|approving|authori[sz]e|sign[- ]?off|rubber[- ]?stamp"
    r"|accept|clear|certify|confirm|mark\b.*\b(approved|reviewed|done|complete))\b"
    r"|\bgive (it )?(the )?(go[- ]?ahead|green light)\b|\bgood to go\b|\blooks good\b"
)
_REQUEST_REVIEW = re.compile(
    r"\b(request|ask for|send|need|escalate|flag|route)\b.*\b(human|reviewer|review|approval)\b"
    r"|\bhuman review\b|\bsend (it |this )?for review\b"
)
_REPORT = re.compile(r"\b(report|dossier|summary of the evidence)\b")
_CHANGES = re.compile(
    r"\bwhat (has |have )?changed\b|\bchanges?\b|\bdifferences?\b|\bdiffer\b|\bcompare\b|\bcomparison\b"
)
_EXPLAIN = re.compile(r"\b(explain|review status|status|describe|tell me about|summari[sz]e)\b")
_ANALYZE = re.compile(r"\b(analy[sz]e|analysis|evaluate|assess|check|look at|examine|process)\b")


def classify(text: str) -> Intent:
    t = re.sub(r"[^a-z0-9\s'-]", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return Intent.UNKNOWN
    if _STATUS_Q.search(t):
        return Intent.APPROVAL_STATUS
    if _APPROVE_ATTEMPT.search(t):
        return Intent.APPROVAL_REFUSED
    if _REQUEST_REVIEW.search(t):
        return Intent.REQUEST_REVIEW
    if _REPORT.search(t):
        return Intent.REPORT
    if _CHANGES.search(t):
        return Intent.CHANGES
    if _ANALYZE.search(t):
        return Intent.ANALYZE
    if _EXPLAIN.search(t):
        return Intent.EXPLAIN
    return Intent.UNKNOWN


def _approval_sentence(session: VoiceSession, case_id: str) -> str:
    req = session.requests.get(case_id)
    if req is None:
        return (
            "No human review has been requested yet. The approval state is "
            f"{ApprovalState.PENDING.value}. The assistant cannot approve cases."
        )
    return (
        f"A human review was requested for reviewer {req.reviewer_name}. The approval "
        f"state is {req.approval_state.value}. It has not been approved; only the named "
        "human reviewer can decide, outside this assistant."
    )


def build_evidence_report(
    case_id: str, session: VoiceSession, evidence: Optional[MRIEvidence] = None
) -> str:
    """Build a report from supplied local evidence or a synthetic case."""
    ev = evidence if evidence is not None else analyze_mri_evidence(case_id)
    ex = explain_review_status(ev)
    evidence_source = "local image-derived" if evidence is not None else "synthetic"
    lines = [
        f"# Evidence Report — {ev.case_id}",
        "",
        f"**Review status:** {ev.status.value}",
        f"**Summary:** {ex.summary}",
        "",
        f"## Measured evidence ({evidence_source})",
        f"- Baseline date: {ev.baseline_date}",
        f"- Current date: {ev.current_date}",
        f"- Quality score: {ev.quality_score:.2f}",
        f"- Foreground: {ev.foreground_percentage:.1f}%",
        f"- Mean intensity: {ev.mean_intensity:.1f}",
        "",
        "## Quality notes",
        *[f"- {n}" for n in ex.quality_notes],
        "",
        "## Change notes",
        *[f"- {n}" for n in ex.change_notes],
        "",
        "## Human approval",
        f"- {_approval_sentence(session, case_id)}",
        "",
        f"_{NON_DIAGNOSTIC_NOTICE}_",
    ]
    return "\n".join(lines)


def route_command(
    text: str,
    case_id: str,
    session: VoiceSession,
    reviewer_name: str = "",
    comment: str = "",
    source: str = "typed",
    evidence: Optional[MRIEvidence] = None,
) -> CommandResult:
    """Execute a supported command against the existing MRI tools.

    `reviewer_name` must come from an explicit UI field, never from speech.
    """
    intent = classify(text)
    effective_case_id = evidence.case_id if evidence is not None else case_id
    command_details = {"intent": intent.value, "source": source}
    # Local-image audit records intentionally exclude free text: it could contain
    # a filename, measurements, or other user-supplied image detail.
    if evidence is None:
        command_details["text"] = text.strip()[:200]
    session.log("command", effective_case_id, **command_details)

    def done(intent_: Intent, spoken: str, display: Optional[str] = None, ok: bool = True) -> CommandResult:
        return CommandResult(intent_, spoken, display or spoken, effective_case_id, ok=ok)

    if intent == Intent.UNKNOWN:
        msg = f"Sorry, I did not understand that. Supported commands: {_SUPPORTED}."
        return done(intent, "Sorry, I did not understand that command.", msg, ok=False)

    if intent == Intent.APPROVAL_REFUSED:
        session.log("approval_refused", effective_case_id)
        msg = (
            "I can't approve cases. Approval requires a named human reviewer and cannot be "
            "granted by voice or by this assistant. The case remains "
            f"{ApprovalState.PENDING.value}. You can say \"request human review\" instead."
        )
        return done(intent, msg, ok=False)

    try:
        ev = evidence if evidence is not None else analyze_mri_evidence(case_id)
    except ValueError as exc:
        return done(intent, str(exc), ok=False)

    if intent == Intent.ANALYZE:
        ex = explain_review_status(ev)
        spoken = (
            f"Case {ev.case_id}: review status {ev.status.value.replace('_', ' ').lower()}. "
            f"Quality score {ev.quality_score:.2f}. Foreground {ev.foreground_percentage:.1f} percent. "
            f"{ex.summary} Human approval is required. This is not a diagnosis."
        )
        return done(intent, spoken)

    if intent == Intent.EXPLAIN:
        ex = explain_review_status(ev)
        spoken = " ".join([ex.summary, *ex.quality_notes, *ex.change_notes]) + (
            " Human approval is required. This is not a diagnosis."
        )
        return done(intent, spoken)

    if intent == Intent.CHANGES:
        ex = explain_review_status(ev)
        spoken = (
            f"Between {ev.baseline_date} and {ev.current_date}: "
            + " ".join(ex.change_notes)
            + " This is a measurement comparison, not a diagnosis."
        )
        return done(intent, spoken)

    if intent == Intent.REPORT:
        report = build_evidence_report(effective_case_id, session, ev)
        ex = explain_review_status(ev)
        spoken = (
            f"Evidence report generated for {ev.case_id}. Status {ev.status.value.replace('_', ' ').lower()}. "
            f"{ex.summary} The full report is on screen. This is not a diagnosis."
        )
        session.log("report_generated", effective_case_id)
        return done(intent, spoken, report)

    if intent == Intent.APPROVAL_STATUS:
        return done(intent, _approval_sentence(session, effective_case_id))

    # Intent.REQUEST_REVIEW
    name = (reviewer_name or "").strip()
    if not name:
        msg = (
            "A named human reviewer is required. Type the reviewer's name in the "
            "Reviewer Name field, then request human review again."
        )
        return done(intent, msg, ok=False)
    req = request_human_approval(effective_case_id, name, comment)
    session.requests[effective_case_id] = req
    session.log("review_requested", effective_case_id, reviewer=req.reviewer_name)
    spoken = (
        f"Human review requested for reviewer {req.reviewer_name}. The approval state is "
        f"{req.approval_state.value}. The assistant cannot approve; only the named reviewer can."
    )
    return CommandResult(intent, spoken, spoken, effective_case_id, approval_state=req.approval_state)
