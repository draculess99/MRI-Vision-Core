"""Deterministic synthetic workflow-validation harness for the evidence dossier.

Metrics here validate the dossier WORKFLOW on curated synthetic scenarios.
They are NOT clinical accuracy, sensitivity, specificity or medical performance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cv2
import numpy as np

from hackathon.nebius_nvidia.evidence_dossier import (
    ApprovalState,
    EvidenceDossier,
    EvidenceStatus,
    build_dossier_from_images,
)

METRICS_LABEL = (
    "Synthetic workflow-validation metrics only; not clinical accuracy "
    "or medical performance."
)

QUALITY_FIELDS = (
    "width",
    "height",
    "mean_intensity",
    "std_deviation",
    "foreground_percentage",
    "flags",
)
TOP_LEVEL_FIELDS = (
    "status",
    "requires_human_approval",
    "approval",
    "quality",
    "changeguard",
    "review_notes",
    "notice",
)


def _disc(radius: int, size: int = 128) -> np.ndarray:
    img = np.full((size, size), 20, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, 220, -1)
    return img


@dataclass(frozen=True)
class Scenario:
    name: str
    image: np.ndarray
    baseline: Optional[np.ndarray]
    expected_status: EvidenceStatus
    reviewer_decision: bool = False


def default_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario("consistent_baseline_current", _disc(30), _disc(30), EvidenceStatus.CONSISTENT),
        Scenario("large_change_review", _disc(50), _disc(15), EvidenceStatus.REVIEW),
        Scenario(
            "low_information_uncertain",
            np.full((128, 128), 100, dtype=np.uint8),
            None,
            EvidenceStatus.UNCERTAIN,
        ),
        Scenario(
            "reviewer_approval",
            _disc(50),
            _disc(15),
            EvidenceStatus.REVIEW,
            reviewer_decision=True,
        ),
    )


def _completeness(dossier: EvidenceDossier, has_baseline: bool) -> float:
    data = dossier.to_dict()
    expected = len(TOP_LEVEL_FIELDS) + len(QUALITY_FIELDS)
    present = sum(1 for k in TOP_LEVEL_FIELDS if k in data)
    present += sum(1 for k in QUALITY_FIELDS if k in data.get("quality", {}))
    # null changeguard is only complete when no baseline was supplied
    if (data.get("changeguard") is None) == has_baseline:
        present -= 1
    if not data.get("review_notes"):
        present -= 1
    return max(present, 0) / expected


def _gate_enforced(scenario: Scenario, dossier: EvidenceDossier) -> bool:
    ok = dossier.requires_human_approval and dossier.approval.state is ApprovalState.PENDING
    try:
        dossier.decide(True, reviewer=" ")
        ok = False
    except ValueError:
        pass
    if scenario.reviewer_decision:
        decided = dossier.decide(True, reviewer="Synthetic Reviewer")
        ok = (
            ok
            and decided.approval.state is ApprovalState.APPROVED
            and decided.requires_human_approval
            and dossier.approval.state is ApprovalState.PENDING
        )
    return ok


def run_evaluation(
    scenarios: Optional[tuple[Scenario, ...]] = None,
    timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    scenarios = scenarios if scenarios is not None else default_scenarios()
    agree = gate = 0
    completeness: list[float] = []
    latencies_ms: list[float] = []
    per_scenario = []

    for sc in scenarios:
        start = timer()
        dossier = build_dossier_from_images(sc.image, baseline_image=sc.baseline)
        latencies_ms.append((timer() - start) * 1000.0)

        matched = dossier.status is sc.expected_status
        enforced = _gate_enforced(sc, dossier)
        comp = _completeness(dossier, sc.baseline is not None)
        agree += matched
        gate += enforced
        completeness.append(comp)
        per_scenario.append(
            {
                "name": sc.name,
                "expected_status": sc.expected_status.value,
                "actual_status": dossier.status.value,
                "status_match": matched,
                "completeness": comp,
                "gate_enforced": enforced,
            }
        )

    n = len(scenarios)
    return {
        "label": METRICS_LABEL,
        "scenarios_evaluated": n,
        "expected_status_agreement": agree / n if n else 0.0,
        "evidence_field_completeness": sum(completeness) / n if n else 0.0,
        "human_approval_gate_enforcement_rate": gate / n if n else 0.0,
        "mean_dossier_build_latency_ms": sum(latencies_ms) / n if n else 0.0,
        "per_scenario": per_scenario,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_evaluation(), indent=2))
