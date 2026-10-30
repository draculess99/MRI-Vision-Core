"""Top-level orchestrator: run three-agent pipeline sequentially."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from .contracts import (
    ExtractionResult,
    VerificationResult,
    AdjudicationResult,
    extract_from_report,
    verify_extraction,
    adjudicate,
)


class ProcessingResult:
    """Complete three-agent pipeline output."""

    def __init__(
        self,
        study_uid: str,
        report_text: str,
        extraction: ExtractionResult,
        verification: VerificationResult,
        adjudication: AdjudicationResult,
    ):
        self.study_uid = study_uid
        self.report_text = report_text
        self.extraction = extraction
        self.verification = verification
        self.adjudication = adjudication
        self.pipeline_timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict:
        """Serialize all three agent outputs."""
        return {
            "study_uid": self.study_uid,
            "pipeline_timestamp": self.pipeline_timestamp,
            "extraction": self.extraction.to_dict(),
            "verification": self.verification.to_dict(),
            "adjudication": self.adjudication.to_dict(),
        }

    def to_json(self) -> str:
        """JSON-serialize the complete result."""
        return json.dumps(self.to_dict(), indent=2, default=str)


def process_report(
    report_text: str,
    study_uid: Optional[str] = None,
    backend: str = "rule-based",
) -> ProcessingResult:
    """
    Run three-agent pipeline sequentially.

    1. Agent 1 extracts findings from report
    2. Agent 2 verifies each extraction against source
    3. Agent 3 reconciles and escalates disagreements

    Args:
        report_text: Raw MRI report
        study_uid: Optional study identifier

    Returns:
        ProcessingResult with all three agent outputs

    Raises:
        NotImplementedError: If agents are not yet implemented
    """
    # Agent 1: Extract
    extraction = extract_from_report(report_text, study_uid=study_uid)

    # Agent 2: Verify
    verification = verify_extraction(report_text, extraction)

    # Agent 3: Adjudicate
    adjudication = adjudicate(report_text, extraction, verification)

    return ProcessingResult(
        study_uid=study_uid or extraction.study_uid,
        report_text=report_text,
        extraction=extraction,
        verification=verification,
        adjudication=adjudication,
    )
