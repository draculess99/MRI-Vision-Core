"""Deterministic terminal demo: MRI Evidence Dossier workflow on synthetic scenarios.

Run: python -m hackathon.nebius_nvidia.demo
"""

from __future__ import annotations

import time
from typing import Callable

from hackathon.nebius_nvidia.eval_harness import (
    METRICS_LABEL,
    Scenario,
    default_scenarios,
    run_evaluation,
)
from hackathon.nebius_nvidia.evidence_dossier import (
    EvidenceDossier,
    build_dossier_from_images,
)

RULE = "=" * 64
THIN = "-" * 64


def _scenario_block(index: int, total: int, sc: Scenario) -> list[str]:
    d: EvidenceDossier = build_dossier_from_images(sc.image, baseline_image=sc.baseline)
    q = d.quality
    lines = [
        THIN,
        f"Scenario {index}/{total}: {sc.name}",
        THIN,
        f"Status            : {d.status.value}",
        f"Image             : {q.width}x{q.height}, mean={q.mean_intensity:.1f}, "
        f"std={q.std_deviation:.1f}, foreground={q.foreground_percentage:.1f}%",
        f"Quality flags     : {', '.join(q.flags) if q.flags else 'none'}",
    ]
    cg = d.changeguard
    if cg is None:
        lines.append("Change evidence   : no baseline supplied")
    else:
        deltas = ", ".join(f"{k}={v:+.2f}" for k, v in cg.deltas.items())
        lines.append(f"Change evidence   : {cg.status.value} ({deltas})")
    lines.append(f"Review note       : {d.notice}")
    lines.append(
        f"Approval          : {d.approval.state.value} "
        f"(human approval required: {'yes' if d.requires_human_approval else 'no'})"
    )
    if sc.reviewer_decision:
        decided = d.decide(True, reviewer="Synthetic Reviewer", comment="demo review")
        lines.append(
            f"Reviewer decision : {d.approval.state.value} -> {decided.approval.state.value} "
            f"by {decided.approval.reviewer}"
        )
        lines.append(
            "Safeguard         : named human reviewer required; human approval "
            f"requirement retained ({'yes' if decided.requires_human_approval else 'no'})"
        )
    return lines


def render_demo(timer: Callable[[], float] = time.perf_counter) -> str:
    scenarios = default_scenarios()
    out = [
        RULE,
        "MRI Evidence Dossier demo - Nebius x NVIDIA",
        "Non-diagnostic, deterministic, synthetic scenarios only.",
        RULE,
    ]
    for i, sc in enumerate(scenarios, 1):
        out.extend(_scenario_block(i, len(scenarios), sc))
    m = run_evaluation(scenarios, timer=timer)
    out += [
        RULE,
        "Evaluation metrics",
        METRICS_LABEL,
        THIN,
        f"Scenarios evaluated            : {m['scenarios_evaluated']}",
        f"Expected-status agreement      : {m['expected_status_agreement']:.0%}",
        f"Evidence-field completeness    : {m['evidence_field_completeness']:.0%}",
        f"Human-approval gate enforcement: {m['human_approval_gate_enforcement_rate']:.0%}",
        f"Mean dossier-build latency     : {m['mean_dossier_build_latency_ms']:.1f} ms",
        RULE,
    ]
    return "\n".join(out)


def main() -> None:
    print(render_demo())


if __name__ == "__main__":
    main()
