"""Label contracts for report-derived (weak) RSNA knee labels.

Every target of every report is exactly one of four states, so "the report does not say" can
never be confused with "the finding is absent":

  positive       the report states the finding is present
  negative       the report explicitly states the finding is absent / the structure is normal
  uncertain      hedged, historical, mild/ambiguous, conflicting, or location-ambiguous mention
  not_mentioned  nothing usable was said (this includes reports in an unsupported language)

Only positive and negative ever produce a training target. uncertain and not_mentioned are masked
out by default; the numeric value stored under a zero mask is NaN so it cannot be read as a zero label.

Extractors are interchangeable: anything with `extract(report_text, study_uid=None) -> ReportLabels`
satisfies `ReportLabelExtractor` (a deterministic rule extractor today, an offline multilingual model later).
This module has no dependency on MRI inference or on any LLM provider.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence

import numpy as np

from .rsna_knee_dataset import TARGET_COLUMNS

EVIDENCE_MAX_CHARS = 160
UNSUPPORTED_METHOD = "unsupported_language"


class LabelState(str, enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"
    NOT_MENTIONED = "not_mentioned"


DECIDED_STATES = frozenset({LabelState.POSITIVE, LabelState.NEGATIVE})


class ReportLabelError(ValueError):
    pass


@dataclass(frozen=True)
class TargetLabel:
    target: str
    state: LabelState
    language: str
    method: str
    extractor_version: str
    evidence: Optional[str] = None  # a short sentence from the report; may be omitted from persisted manifests
    section: Optional[str] = None  # e.g. "findings", "impression", "body"
    reason: Optional[str] = None  # short machine-readable note, e.g. "conflicting_mentions", "historical"

    def __post_init__(self):
        if self.target not in TARGET_COLUMNS:
            raise ReportLabelError(f"Unknown target: {self.target!r}")
        if not isinstance(self.state, LabelState):
            raise ReportLabelError("state must be a LabelState")
        if self.evidence is not None and len(self.evidence) > EVIDENCE_MAX_CHARS:
            object.__setattr__(self, "evidence", self.evidence[:EVIDENCE_MAX_CHARS])

    @property
    def evidence_sha256(self) -> Optional[str]:
        return None if self.evidence is None else hashlib.sha256(self.evidence.encode("utf-8")).hexdigest()

    def to_dict(self, include_evidence: bool = False) -> dict:
        row = {"state": self.state.value, "section": self.section, "reason": self.reason,
               "evidence_sha256": self.evidence_sha256}
        if include_evidence:
            row["evidence"] = self.evidence
        return row


@dataclass(frozen=True)
class ReportLabels:
    study_uid: Optional[str]
    language: str
    supported: bool  # False -> every label is not_mentioned and must not be used for training
    method: str
    extractor_version: str
    labels: tuple  # one TargetLabel per TARGET_COLUMNS entry, in order

    def __post_init__(self):
        if tuple(label.target for label in self.labels) != tuple(TARGET_COLUMNS):
            raise ReportLabelError("labels must cover every target exactly once, in TARGET_COLUMNS order")

    def state(self, target: str) -> LabelState:
        return self.labels[TARGET_COLUMNS.index(target)].state

    def decided_count(self) -> int:
        return sum(label.state in DECIDED_STATES for label in self.labels)

    def to_dict(self, include_evidence: bool = False) -> dict:
        return {"study_uid": self.study_uid, "language": self.language, "supported": self.supported,
                "method": self.method, "extractor_version": self.extractor_version,
                "targets": {label.target: label.to_dict(include_evidence) for label in self.labels}}


class ReportLabelExtractor(Protocol):
    version: str

    def extract(self, report_text: str, study_uid: Optional[str] = None) -> ReportLabels: ...


def unsupported_labels(language: str, method: str, version: str, study_uid: Optional[str] = None) -> ReportLabels:
    """Labels for a report we refuse to classify: everything is not_mentioned and flagged unsupported."""
    return ReportLabels(
        study_uid, language, False, UNSUPPORTED_METHOD, version,
        tuple(TargetLabel(t, LabelState.NOT_MENTIONED, language, UNSUPPORTED_METHOD, version,
                          reason="unsupported_language") for t in TARGET_COLUMNS),
    )


def to_training_arrays(labels: ReportLabels, *, uncertain: str = "mask", uncertain_value: float = 0.5):
    """Return (values[12], mask[12]) for masked multi-label training.

    mask is 1 only where a target is trainable. Masked values are NaN, never 0.
      uncertain="mask" (default): uncertain targets are masked out.
      uncertain="soft": uncertain targets get `uncertain_value` with mask 1 (explicit opt-in).
    not_mentioned and unsupported-language labels are always masked.
    """
    if uncertain not in ("mask", "soft"):
        raise ReportLabelError("uncertain must be 'mask' or 'soft'")
    values = np.full(len(TARGET_COLUMNS), np.nan, dtype=np.float32)
    mask = np.zeros(len(TARGET_COLUMNS), dtype=np.float32)
    if not labels.supported:
        return values, mask
    for i, label in enumerate(labels.labels):
        if label.state is LabelState.POSITIVE:
            values[i], mask[i] = 1.0, 1.0
        elif label.state is LabelState.NEGATIVE:
            values[i], mask[i] = 0.0, 1.0
        elif label.state is LabelState.UNCERTAIN and uncertain == "soft":
            values[i], mask[i] = uncertain_value, 1.0
    return values, mask


def states_by_target(labels: ReportLabels) -> Mapping[str, str]:
    return {label.target: label.state.value for label in labels.labels}


def decided_targets(labels_seq: Sequence[ReportLabels]) -> np.ndarray:
    """(n_reports, 12) boolean array: which targets are positive/negative."""
    return np.array([[label.state in DECIDED_STATES for label in r.labels] for r in labels_seq], dtype=bool)
