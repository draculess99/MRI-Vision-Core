"""Multi-agent report labeling system: extractor, verifier, adjudicator.

Research prototype for Phase 20B. Offline-only processing (rule-based + optional local LLM).
All agents operate on report text and produce structured label outputs.
"""

from .contracts import (
    LabelState,
    VerificationStatus,
    AdjudicationConsensus,
    Finding,
    ExtractionResult,
    VerificationResult,
    AdjudicationResult,
    extract_from_report,
    verify_extraction,
    adjudicate,
)
from .extractor import extract_from_report_rule_based
from .verifier import verify_extraction as verify_impl
from .adjudicator import adjudicate as adjudicate_impl
from .orchestrator import process_report, ProcessingResult

__all__ = [
    # Enums
    "LabelState",
    "VerificationStatus",
    "AdjudicationConsensus",
    # Data structures
    "Finding",
    "ExtractionResult",
    "VerificationResult",
    "AdjudicationResult",
    # Agent functions
    "extract_from_report",
    "verify_extraction",
    "adjudicate",
    "extract_from_report_rule_based",
    # Pipeline
    "process_report",
    "ProcessingResult",
]
