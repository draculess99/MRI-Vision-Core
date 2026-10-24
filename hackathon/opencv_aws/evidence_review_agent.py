"""Deterministic Evidence Review Agent.

Non-diagnostic, research-only. This agent consumes OpenCV evidence manifest
and ChangeGuard baseline comparison outputs, applies rule-based checks,
and produces one of three deterministic outcomes.

All decisions are based on existing evidence; no recomputation, no LLM.
"""

from typing import Any, Dict, Optional, Tuple
from .review_contracts import (
    AgentDecision,
    AgentTrace,
    ReviewAction,
    ReviewOutcome,
    ToolStep,
)


def _validate_manifest(manifest: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    """
    Validate manifest structure and completeness.

    Returns:
        (status: "PASS" | "FLAG", findings: tuple of observation strings)
    """
    findings = []

    # Check required fields
    required = [
        "source_hash_sha256",
        "processing_steps",
        "generated_artifacts",
        "input_width",
        "input_height",
        "input_dtype",
    ]
    for field in required:
        if field not in manifest or manifest[field] is None:
            findings.append(f"Missing or None: {field}")

    # Validate processing_steps
    if "processing_steps" in manifest:
        steps = manifest["processing_steps"]
        if not isinstance(steps, list):
            findings.append("processing_steps is not a list")
        else:
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    findings.append(f"processing_steps[{i}] is not a dict")
                elif step.get("deterministic") is not True:
                    findings.append(
                        f"processing_steps[{i}] ('{step.get('step', '?')}') "
                        "is not marked deterministic"
                    )

    # Validate generated_artifacts
    if "generated_artifacts" in manifest:
        artifacts = manifest["generated_artifacts"]
        expected = {"original", "enhanced", "edges", "overlay"}
        if not isinstance(artifacts, list):
            findings.append("generated_artifacts is not a list")
        elif set(artifacts) != expected:
            findings.append(
                f"generated_artifacts {set(artifacts)} does not match expected {expected}"
            )

    # Validate dimensions
    width = manifest.get("input_width", 0)
    height = manifest.get("input_height", 0)
    if width <= 0 or width > 10000:
        findings.append(f"input_width {width} is invalid (must be 1-10000)")
    if height <= 0 or height > 10000:
        findings.append(f"input_height {height} is invalid (must be 1-10000)")

    # Validate dtype
    dtype = manifest.get("input_dtype", "")
    if dtype not in ("uint8", "float32", "float64"):
        findings.append(f"input_dtype '{dtype}' is unexpected (expected uint8 or float32)")

    status = "PASS" if not findings else "FLAG"
    return status, tuple(findings)


def _assess_change_guard(
    comparison: Optional[Dict[str, Any]],
) -> Tuple[str, Tuple[str, ...]]:
    """
    Assess ChangeGuard baseline comparison result.

    Returns:
        (status: "PASS" | "FLAG", findings: tuple of observation strings)
    """
    findings = []

    if comparison is None:
        findings.append("No ChangeGuard comparison result provided")
        return "FLAG", tuple(findings)

    status_val = comparison.get("status")
    if status_val not in ("STABLE", "REVIEW", "INVALID"):
        findings.append(f"ChangeGuard status '{status_val}' is not recognized")
        return "FLAG", tuple(findings)

    if status_val == "REVIEW":
        foreground = comparison.get("foreground_delta")
        mean_intensity = comparison.get("mean_intensity_delta")
        findings.append(
            f"ChangeGuard triggered REVIEW: foreground_delta={foreground}%, "
            f"mean_intensity_delta={mean_intensity}"
        )
        return "FLAG", tuple(findings)

    if status_val == "INVALID":
        reason = comparison.get("reason", "Unknown reason")
        # First-run condition: no baseline available is recoverable
        if "No baseline" in reason:
            findings.append("First-run condition: no baseline available (expected)")
            return "PASS", tuple(findings)
        # Other invalid conditions are unrecoverable
        findings.append(f"ChangeGuard INVALID: {reason}")
        return "FLAG", tuple(findings)

    # STABLE
    findings.append("ChangeGuard STABLE: baseline comparison passed")
    return "PASS", tuple(findings)


def _choose_next_action(
    manifest_status: str,
    manifest_findings: Tuple[str, ...],
    comparison_status: str,
    comparison_findings: Tuple[str, ...],
) -> Tuple[ReviewOutcome, str, ReviewAction, bool]:
    """
    Deterministic decision logic: combine manifest and comparison results.

    Returns:
        (outcome, reasoning, next_action, human_approval_required)
    """
    # Categorize manifest issues as recoverable (request new input) or unrecoverable (need approval)
    unrecoverable_manifest_keywords = [
        "input_width",
        "input_height",
        "input_dtype",
    ]
    recoverable_manifest_keywords = [
        "generated_artifacts",
        "processing_steps",
    ]

    unrecoverable_findings = [
        f for f in manifest_findings
        if any(k in f for k in unrecoverable_manifest_keywords)
    ]
    recoverable_findings = [
        f for f in manifest_findings
        if f not in unrecoverable_findings
    ]

    # === Rule 1: Pending human approval if unrecoverable manifest issues exist ===
    if unrecoverable_findings:
        reasoning = f"Manifest validation failed: {'; '.join(unrecoverable_findings[:2])}"
        return (
            ReviewOutcome.PENDING_HUMAN_APPROVAL,
            reasoning,
            ReviewAction(
                action_type="halt_pending_approval",
                user_facing_text="Manual review required: image metadata is invalid or missing. "
                "Please check that your input image is valid.",
                is_recoverable=False,
            ),
            True,
        )

    # === Rule 2: Request more evidence if recoverable manifest issues or ChangeGuard flagged ===
    if recoverable_findings or comparison_status == "FLAG":
        # Check if it's the first-run no-baseline case
        if comparison_status == "FLAG" and any("No baseline" in f for f in comparison_findings):
            # First-run is fine; check manifest only
            if recoverable_findings:
                reasoning = f"Evidence review incomplete: {'; '.join(recoverable_findings[:2])}"
            else:
                # No issues; proceed to finalize
                reasoning = "All artifacts generated deterministically. Evidence manifest complete. Baseline stable (or first run)."
                return (
                    ReviewOutcome.FINALIZE_EVIDENCE_BUNDLE,
                    reasoning,
                    ReviewAction(
                        action_type="finalize",
                        user_facing_text="Evidence bundle finalized. All processing steps recorded deterministically.",
                        is_recoverable=True,
                    ),
                    False,
                )
        else:
            # ChangeGuard REVIEW or recoverable manifest issues
            if comparison_status == "FLAG":
                reasoning = f"Evidence review incomplete: {'; '.join(comparison_findings[:2])}"
            else:
                reasoning = f"Evidence review incomplete: {'; '.join(recoverable_findings[:2])}"

        return (
            ReviewOutcome.REQUEST_MORE_EVIDENCE,
            reasoning,
            ReviewAction(
                action_type="request_input",
                user_facing_text="Please upload a reference image or a different study image.",
                is_recoverable=True,
            ),
            False,
        )

    # === Rule 3: Finalize if both manifest and comparison passed ===
    if manifest_status == "PASS" and comparison_status == "PASS":
        reasoning = "All artifacts generated deterministically. Evidence manifest complete. Baseline stable (or first run)."
        return (
            ReviewOutcome.FINALIZE_EVIDENCE_BUNDLE,
            reasoning,
            ReviewAction(
                action_type="finalize",
                user_facing_text="Evidence bundle finalized. All processing steps recorded deterministically.",
                is_recoverable=True,
            ),
            False,
        )

    # === Fallback: Unclassified state ===
    reasoning = f"Unclassified state: manifest={manifest_status}, comparison={comparison_status}"
    return (
        ReviewOutcome.PENDING_HUMAN_APPROVAL,
        reasoning,
        ReviewAction(
            action_type="halt_pending_approval",
            user_facing_text="An unexpected condition occurred. Please contact support.",
            is_recoverable=False,
        ),
        True,
    )


def review_evidence(
    manifest: Dict[str, Any],
    comparison: Optional[Dict[str, Any]],
) -> AgentDecision:
    """
    Run the Evidence Review Agent on OpenCV manifest and ChangeGuard comparison.

    Args:
        manifest: Dict from EvidenceManifest.to_dict()
        comparison: Dict from ComparisonResult.to_dict() or None

    Returns:
        AgentDecision with outcome, reasoning, next_action, and auditable trace
    """
    source_hash = manifest.get("source_hash_sha256", "unknown")

    # === Tool Step 1: Validate Manifest ===
    manifest_status, manifest_findings = _validate_manifest(manifest)
    step1 = ToolStep(
        tool_name="validate_manifest",
        status=manifest_status,
        findings=manifest_findings,
    )

    # === Tool Step 2: Assess ChangeGuard ===
    comparison_status, comparison_findings = _assess_change_guard(comparison)
    step2 = ToolStep(
        tool_name="assess_change_guard",
        status=comparison_status,
        findings=comparison_findings,
    )

    # === Tool Step 3: Choose Next Action ===
    outcome, reasoning, next_action, human_approval = _choose_next_action(
        manifest_status,
        manifest_findings,
        comparison_status,
        comparison_findings,
    )
    step3 = ToolStep(
        tool_name="choose_next_action",
        status="PASS",
        findings=(f"Decided: {outcome.value}",),
    )

    # === Build Trace ===
    observations = {
        "manifest_status": manifest_status,
        "manifest_fields_present": sorted(manifest.keys()),
        "comparison_status": comparison_status,
        "change_guard_result": comparison if comparison else None,
    }

    trace = AgentTrace(
        trace_version="1.0",
        source_hash_sha256=source_hash,
        observations=observations,
        tool_steps=(step1, step2, step3),
        outcome=outcome.value,
    )

    return AgentDecision(
        outcome=outcome,
        reasoning=reasoning,
        next_action=next_action,
        human_approval_required=human_approval,
        trace=trace,
    )
