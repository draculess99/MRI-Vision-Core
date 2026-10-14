"""Critic layer: deterministic, rule-based validation of an EvidenceBundle.

This never certifies anything as "clinically safe" -- a successful inference status is
not treated as a safety signal. Every check here is a structural/consistency check on
the evidence itself (shape, range, provenance, quality-gate honesty), returned as
explicit, typed findings with a severity and an evidence reference where applicable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from mri_core.rsna_inference_engine import STATUS_ERROR, STATUS_OK, STATUS_UNABLE_TO_EVALUATE
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

from .contracts import EvidenceBundle, QualityGateStatus, Severity

# Finding codes that represent an internal inconsistency in the inference engine's own
# output (as opposed to a quality or provenance concern about the underlying study).
INTERNAL_INCONSISTENCY_CODES = frozenset(
    {
        "LABEL_ORDER_MISMATCH",
        "NON_FINITE_PROBABILITY",
        "PROBABILITY_OUT_OF_RANGE",
        "STATUS_PROBABILITY_MISMATCH",
        "MISSING_PROVENANCE",
        "MISSING_PROVENANCE_HASH",
        "UNEXPECTED_ENSEMBLE_SIZE",
    }
)


@dataclass(frozen=True)
class CriticFinding:
    severity: Severity
    code: str
    message: str
    evidence_reference: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "evidence_reference": self.evidence_reference,
        }


def review(bundle: EvidenceBundle) -> Tuple[CriticFinding, ...]:
    """Run every deterministic check against bundle and return the findings, most
    severe first is not guaranteed -- callers that need ordering should sort by
    severity themselves; this function preserves check order for reproducibility.
    """
    findings: list[CriticFinding] = []

    if bundle.inference_status not in (STATUS_OK, STATUS_UNABLE_TO_EVALUATE, STATUS_ERROR):
        findings.append(
            CriticFinding(
                Severity.ERROR, "UNKNOWN_INFERENCE_STATUS",
                f"inference_status {bundle.inference_status!r} is not a recognized engine status.",
            )
        )

    if bundle.inference_status == STATUS_OK and bundle.probabilities is None:
        findings.append(
            CriticFinding(
                Severity.ERROR, "STATUS_PROBABILITY_MISMATCH",
                "inference_status is OK but probabilities is None.",
            )
        )
    if bundle.inference_status != STATUS_OK and bundle.probabilities is not None:
        findings.append(
            CriticFinding(
                Severity.WARNING, "STATUS_PROBABILITY_MISMATCH",
                f"inference_status is {bundle.inference_status!r} but probabilities are present.",
            )
        )

    if bundle.probabilities is not None:
        keys = list(bundle.probabilities.keys())
        if keys != list(TARGET_COLUMNS):
            findings.append(
                CriticFinding(
                    Severity.ERROR, "LABEL_ORDER_MISMATCH",
                    f"Expected exactly the 12 TARGET_COLUMNS in order; got {keys!r}.",
                )
            )
        for target, value in bundle.probabilities.items():
            is_finite = isinstance(value, float) and value == value and value not in (float("inf"), float("-inf"))
            if not is_finite:
                findings.append(
                    CriticFinding(Severity.ERROR, "NON_FINITE_PROBABILITY", f"{target}: {value!r}", target)
                )
            elif not (0.0 <= value <= 1.0):
                findings.append(
                    CriticFinding(Severity.ERROR, "PROBABILITY_OUT_OF_RANGE", f"{target}: {value!r}", target)
                )

    if bundle.inference_status == STATUS_OK:
        if bundle.model_count == 0:
            findings.append(CriticFinding(Severity.ERROR, "MISSING_PROVENANCE", "No checkpoint provenance recorded."))
        else:
            for index, provenance in enumerate(bundle.checkpoint_provenance):
                if not provenance.get("checkpoint_sha256"):
                    findings.append(
                        CriticFinding(
                            Severity.ERROR, "MISSING_PROVENANCE_HASH",
                            f"checkpoint_provenance[{index}] is missing a checkpoint_sha256.",
                            provenance.get("checkpoint_path"),
                        )
                    )
        if bundle.is_ensemble and bundle.model_count != 5:
            findings.append(
                CriticFinding(
                    Severity.WARNING, "UNEXPECTED_ENSEMBLE_SIZE",
                    f"Ensemble reported but model_count={bundle.model_count}, expected 5.",
                )
            )
        if not bundle.is_ensemble and bundle.model_count != 1:
            findings.append(
                CriticFinding(
                    Severity.WARNING, "UNEXPECTED_SINGLE_MODEL_COUNT",
                    f"Single-model inference reported but model_count={bundle.model_count}, expected 1.",
                )
            )

    if bundle.quality_gate_status == QualityGateStatus.NOT_ASSESSED:
        findings.append(
            CriticFinding(
                Severity.INFO, "QUALITY_NOT_ASSESSED",
                "Deterministic image-quality assessment was not performed for this study.",
            )
        )
    elif bundle.quality_gate_status == QualityGateStatus.INVALID:
        findings.append(
            CriticFinding(Severity.ERROR, "QUALITY_INVALID", "Deterministic quality status is INVALID.")
        )
    elif bundle.quality_gate_status == QualityGateStatus.REVIEW:
        findings.append(
            CriticFinding(Severity.WARNING, "QUALITY_REVIEW", "Deterministic quality status flags REVIEW.")
        )

    if bundle.warnings:
        findings.append(
            CriticFinding(Severity.WARNING, "ENGINE_WARNINGS_PRESENT", "; ".join(bundle.warnings))
        )

    if bundle.inference_status == STATUS_OK and not bundle.evidence_references:
        findings.append(
            CriticFinding(
                Severity.INFO, "NO_EVIDENCE_REFERENCES",
                "No evidence_references were attached to this bundle (e.g. checkpoint or report paths).",
            )
        )

    return tuple(findings)
