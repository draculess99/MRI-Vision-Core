"""Loader/validator for benchmarks/amd/public_demo_cases.json -- the five synthetic, non-patient cases
used by the public (Railway) synthetic-demo mode.

Loading and validating needs only the standard library. `build_case_inputs()` imports the shared
mri_core result types lazily; it builds a StudyInferenceResult directly from the case's synthetic
numbers -- no model, checkpoint, DICOM or network is ever involved. Malformed or missing data raises
PublicDemoCasesError (shown as a clean UI error); nothing is ever fabricated as a fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_CASES_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "amd" / "public_demo_cases.json"
EXPECTED_CASE_COUNT = 5
TARGETS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)
INFERENCE_STATUSES = ("OK", "UNABLE_TO_EVALUATE", "ERROR")
QUALITY_STATUSES = ("NOT_ASSESSED", "OK", "REVIEW", "INVALID")
WORKFLOW_STATUSES = (
    "EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW", "MORE_EVIDENCE_REQUIRED", "INFERENCE_UNAVAILABLE", "INCONSISTENT_EVIDENCE",
)
_REQUIRED = (
    "case_id", "title", "synthetic", "inference_status", "quality_gate_status", "scores", "warnings",
    "evidence_references", "expected_workflow_status", "expected_reason",
)
_FORBIDDEN_FRAGMENTS = ("1.2.826", ".dcm", ".ckpt", "train_series", "StudyInstanceUID", ":\\", "/Users/")


class PublicDemoCasesError(Exception):
    """Every message is written to be shown directly in the UI."""


@dataclass(frozen=True)
class PublicDemoCase:
    case_id: str
    title: str
    inference_status: str
    quality_gate_status: str
    scores: Optional[Dict[str, float]]
    warnings: Tuple[str, ...]
    evidence_references: Tuple[str, ...]
    expected_workflow_status: str
    expected_reason: str


@dataclass(frozen=True)
class PublicDemoCaseSet:
    schema_version: str
    disclaimer: str
    cases: Tuple[PublicDemoCase, ...]


def case_label(case: PublicDemoCase) -> str:
    return f"{case.case_id} — {case.title}"


def _fail(path: Path, message: str) -> PublicDemoCasesError:
    return PublicDemoCasesError(f"{message} ({path.name})")


def _parse_case(raw: object, index: int, path: Path) -> PublicDemoCase:
    if not isinstance(raw, dict):
        raise _fail(path, f"case {index} is not a JSON object")
    missing = [f for f in _REQUIRED if f not in raw]
    if missing:
        raise _fail(path, f"case {index} is missing field(s) {missing}")
    if raw["synthetic"] is not True:
        raise _fail(path, f"case {index} must be marked synthetic: true")
    for key in ("case_id", "title", "expected_reason"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise _fail(path, f"case {index} field {key!r} must be non-empty text")
    if raw["inference_status"] not in INFERENCE_STATUSES:
        raise _fail(path, f"case {raw['case_id']} has an unknown inference_status")
    if raw["quality_gate_status"] not in QUALITY_STATUSES:
        raise _fail(path, f"case {raw['case_id']} has an unknown quality_gate_status")
    if raw["expected_workflow_status"] not in WORKFLOW_STATUSES:
        raise _fail(path, f"case {raw['case_id']} has an unknown expected_workflow_status")
    scores = raw["scores"]
    if scores is not None:
        if not isinstance(scores, dict) or list(scores) != list(TARGETS):
            raise _fail(path, f"case {raw['case_id']} scores must be null or exactly the 12 targets in order")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0 for v in scores.values()):
            raise _fail(path, f"case {raw['case_id']} scores must be numbers in [0, 1]")
    for key in ("warnings", "evidence_references"):
        if not isinstance(raw[key], list) or not all(isinstance(v, str) for v in raw[key]):
            raise _fail(path, f"case {raw['case_id']} field {key!r} must be a list of text")
    blob = json.dumps(raw)
    if any(fragment in blob for fragment in _FORBIDDEN_FRAGMENTS):
        raise _fail(path, f"case {raw['case_id']} contains something that looks like real study data or a file path")
    return PublicDemoCase(
        case_id=raw["case_id"], title=raw["title"], inference_status=raw["inference_status"],
        quality_gate_status=raw["quality_gate_status"],
        scores=None if scores is None else {k: float(v) for k, v in scores.items()},
        warnings=tuple(raw["warnings"]), evidence_references=tuple(raw["evidence_references"]),
        expected_workflow_status=raw["expected_workflow_status"], expected_reason=raw["expected_reason"],
    )


def load_public_demo_cases(path: Optional[Path] = None) -> PublicDemoCaseSet:
    cases_path = Path(path) if path is not None else DEFAULT_CASES_PATH
    if not cases_path.is_file():
        raise PublicDemoCasesError(f"Public demonstration case file not found: {cases_path.name}")
    try:
        data = json.loads(cases_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PublicDemoCasesError(f"Public demonstration case file is not valid JSON: {cases_path.name}") from exc
    if not isinstance(data, dict):
        raise _fail(cases_path, "top level is not a JSON object")
    for key in ("schema_version", "disclaimer", "cases"):
        if key not in data:
            raise _fail(cases_path, f"missing top-level field {key!r}")
    if not isinstance(data["disclaimer"], str) or "not clinical" not in data["disclaimer"]:
        raise _fail(cases_path, "disclaimer must state the cases are not clinical advice")
    if not isinstance(data["cases"], list) or len(data["cases"]) != EXPECTED_CASE_COUNT:
        raise _fail(cases_path, f"expected exactly {EXPECTED_CASE_COUNT} cases")
    cases: List[PublicDemoCase] = [_parse_case(raw, i, cases_path) for i, raw in enumerate(data["cases"])]
    if len({c.case_id for c in cases}) != len(cases) or len({c.title for c in cases}) != len(cases):
        raise _fail(cases_path, "case_id and title values must be unique")
    return PublicDemoCaseSet(str(data["schema_version"]), data["disclaimer"], tuple(cases))


def build_case_inputs(case: PublicDemoCase):
    """(StudyInferenceResult, QualityGateStatus, evidence_references) for the unmodified
    mri_core.agents pipeline. Deterministic; built only from the case's own synthetic fields."""
    from hackathon.amd.synthetic_demo import SYNTHETIC_CHECKPOINT_PATH, SYNTHETIC_DISCLAIMER
    from mri_core.agents.contracts import QualityGateStatus
    from mri_core.rsna_inference_engine import (
        STATUS_ERROR, STATUS_OK, STATUS_UNABLE_TO_EVALUATE, ModelProvenance, StudyInferenceResult,
    )
    from mri_core.rsna_knee_dataset import TARGET_COLUMNS

    status = {"OK": STATUS_OK, "UNABLE_TO_EVALUATE": STATUS_UNABLE_TO_EVALUATE, "ERROR": STATUS_ERROR}[case.inference_status]
    provenance = ModelProvenance(
        checkpoint_path=SYNTHETIC_CHECKPOINT_PATH, checkpoint_sha256="0" * 64, epoch=None, fold=None,
        target_columns=tuple(TARGET_COLUMNS),
    )
    result = StudyInferenceResult(
        study_uid=f"{case.case_id}-SYNTHETIC-NOT-A-REAL-STUDY", status=status,
        probabilities=None if case.scores is None else dict(case.scores), is_ensemble=False,
        model_provenance=[provenance], latency_seconds=0.0,
        warnings=list(case.warnings), disclaimer=SYNTHETIC_DISCLAIMER,
    )
    return result, QualityGateStatus(case.quality_gate_status), case.evidence_references
