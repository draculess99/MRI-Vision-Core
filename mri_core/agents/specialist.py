"""Specialist layer: deterministic grouping of the twelve RSNA targets.

This never invents a clinical reading. Each group simply restates the subset of raw,
uncalibrated model scores belonging to it, verbatim, with an explicit reminder that a
score near 0.5 does not mean an abnormality was detected. No positive/negative decision
is made anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from mri_core.rsna_knee_dataset import TARGET_COLUMNS

from .contracts import EvidenceBundle

# Every one of the 12 TARGET_COLUMNS must appear in exactly one group below.
GROUPS: Dict[str, Tuple[str, ...]] = {
    "Ligaments": ("ACL", "MCL"),
    "Menisci": ("Medial Meniscus", "Lateral Meniscus"),
    "Osteoarthritis": ("Medial OA", "Lateral OA", "PF OA"),
    "Other": ("Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"),
}

_NOTE_WITH_SCORES = (
    "Uncalibrated research scores shown exactly as produced by the model. A score near "
    "0.5 does not imply an abnormality was detected, and no positive/negative decision "
    "is made by this layer."
)
_NOTE_NO_SCORES = "No probabilities available for this study; inference did not succeed."


@dataclass(frozen=True)
class SpecialistSummary:
    group_name: str
    target_names: Tuple[str, ...]
    scores: Dict[str, float]  # subset of bundle.probabilities, or {} when unavailable
    note: str

    def to_dict(self) -> dict:
        return {
            "group_name": self.group_name,
            "target_names": list(self.target_names),
            "scores": dict(self.scores),
            "note": self.note,
        }


def _validate_group_coverage() -> None:
    covered = [t for targets in GROUPS.values() for t in targets]
    if sorted(covered) != sorted(TARGET_COLUMNS) or len(covered) != len(TARGET_COLUMNS):
        raise AssertionError(
            f"GROUPS must cover exactly the 12 TARGET_COLUMNS once each; "
            f"got {sorted(covered)!r} vs {sorted(TARGET_COLUMNS)!r}"
        )


_validate_group_coverage()


def summarize_by_group(bundle: EvidenceBundle) -> Tuple[SpecialistSummary, ...]:
    """Group the bundle's twelve targets into four deterministic specialist summaries.

    Every target from TARGET_COLUMNS is traceable to exactly one group; values are
    copied verbatim from bundle.probabilities, never recomputed or rounded.
    """
    summaries = []
    for group_name, targets in GROUPS.items():
        if bundle.probabilities is None:
            scores: Dict[str, float] = {}
            note = _NOTE_NO_SCORES
        else:
            scores = {target: bundle.probabilities[target] for target in targets}
            note = _NOTE_WITH_SCORES
        summaries.append(SpecialistSummary(group_name=group_name, target_names=targets, scores=scores, note=note))
    return tuple(summaries)
