"""CLI demo: processes the synthetic workflow-validation scenarios and prints the
findings. No real data, no network calls, no training.

Run with:
    python -m mri_core.agents.cli_demo
"""

from __future__ import annotations

import json

from .evaluation import run_ablation, run_evaluation


def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    print("mri_core.agents — Phase 8 synthetic workflow-validation demo")
    print("Research prototype. No real patient data, no network calls, no training.")

    _print_header("SCENARIOS")
    report = run_evaluation()
    for outcome in report.outcomes:
        status_label = "PASS" if outcome.passed else "FAIL"
        expected = outcome.expected_workflow_status.value if outcome.expected_workflow_status else "n/a"
        actual = outcome.actual_workflow_status.value if outcome.actual_workflow_status else "n/a"
        print(f"[{status_label}] {outcome.name}")
        print(f"       expected={expected}  actual={actual}")
        print(f"       detail: {outcome.detail}")
        print(f"       latency: {outcome.latency_seconds * 1000:.2f} ms")

    _print_header("SYNTHETIC WORKFLOW-VALIDATION METRICS (not clinical accuracy)")
    print(f"scenario_count:            {report.scenario_count}")
    print(f"passed_count:              {report.passed_count}")
    print(f"failed_count:              {report.failed_count}")
    print(f"workflow_status_agreement: {report.workflow_status_agreement:.3f}")
    print(f"total_latency_seconds:     {report.total_latency_seconds:.6f}")
    print(f"mean_latency_seconds:      {report.mean_latency_seconds:.6f}")

    _print_header("ABLATION: A (raw evidence) vs B (+specialist/critic) vs C (+consensus/review gate)")
    for level in run_ablation():
        print(json.dumps(level.to_dict(), indent=2))

    _print_header("DONE")
    print("All metrics above are synthetic workflow-validation results only.")
    print("Not a clinical accuracy or diagnostic performance measurement.")
    return 0 if report.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
