"""Reproducible automated benchmark runner over the 22 curated AMD demo scenarios
(`benchmarks/amd/amd_demo_scenarios.json`, loaded through `benchmark_scenarios.py`).

Reuses the existing real-inference building blocks directly; this module does not implement
a parallel model or preprocessing path:
  - `hackathon.amd.benchmark_scenarios.load_scenarios()`              -- the 22 scenarios
  - `mri_core.rsna_integration.load_rsna_metadata_safe()` /
    `discover_available_studies()`                                    -- local availability
    (the same two calls `hackathon/amd/streamlit_app.py`'s `_list_available_studies()` makes)
  - `hackathon.amd.run_real_checkpoint_smoke.pick_study_and_series()` /
    `sha256_of_file()` / `NON_DIAGNOSTIC_NOTICE`
  - `hackathon.amd.kaggle_mil_bridge.load_bridge_checkpoint()` /
    `run_study_inference()` / `BridgeError`                            -- the real forward pass

For a scenario whose `study_id` is not present locally, no inference is attempted and no
prediction is fabricated -- it is recorded as `run_status="unavailable"`. A scenario that
raises during inference is recorded as `run_status="inference_error"` with the error message,
and the run continues with the remaining scenarios. Nothing here stores raw DICOM pixel data
or file paths; only study/series identifiers, probabilities, and timing are kept.

Aggregate metrics (ROC-AUC / PR-AUC per target) are computed only over scenarios that were
*successfully inferred* -- never over unavailable or failed ones -- and explicitly report
"not computable" with a reason when a target lacks both a positive and a negative
successfully-inferred example, rather than printing a misleading number from too few samples.
This is a research/demo benchmark; nothing here is a clinical-accuracy or medical-performance
claim.
"""

from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from mri_core.rsna_integration import RSNADiscoveryError, discover_available_studies, load_rsna_metadata_safe

from .benchmark_scenarios import ScenarioSet
from .kaggle_mil_bridge import BridgeError, load_bridge_checkpoint, run_study_inference
from .run_real_checkpoint_smoke import NON_DIAGNOSTIC_NOTICE, pick_study_and_series, sha256_of_file

SCHEMA_VERSION = "amd-benchmark-runner/v1"
DISCLAIMER = "Research/demo benchmark only; not clinical accuracy or medical performance."
COVERAGE_NOTE = (
    "Metrics are computed only over locally available scenarios. Unavailable scenarios are "
    "reported separately and are never assigned fabricated predictions."
)
DEFAULT_THRESHOLD = 0.5
MIN_SAMPLES_FOR_METRIC = 5
"""Below this many successfully-inferred examples for a target, ROC-AUC/PR-AUC are reported
as "insufficient local sample size" rather than a computed number -- even when there happens
to be one positive and one negative example, a 2-sample AUC of 0.0 or 1.0 is not a meaningful
ranking metric. See the user-facing requirement this encodes: never let a tiny local sample
look like a real evaluation result."""

RUN_STATUS_SUCCESS = "success"
RUN_STATUS_UNAVAILABLE = "unavailable"
RUN_STATUS_ERROR = "inference_error"


@dataclass
class ScenarioRunResult:
    scenario_id: str
    title: str
    study_id: str
    available_local: bool
    run_status: str  # one of RUN_STATUS_SUCCESS / RUN_STATUS_UNAVAILABLE / RUN_STATUS_ERROR
    positive_targets: List[str]
    label_vector: Optional[List[int]]
    model_probabilities: Optional[Dict[str, float]]
    predicted_positive_targets: Optional[List[str]]
    latency_seconds: Optional[float]
    series_uid: Optional[str]
    plane: Optional[str]
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _local_availability(data_root: Path) -> set:
    """The same two calls streamlit_app.py's `_list_available_studies()` makes."""
    metadata = load_rsna_metadata_safe(data_root)
    return set(discover_available_studies(data_root, metadata))


def dry_run_report(scenario_set: ScenarioSet, data_root: Optional[Path]) -> List[dict]:
    """Load scenarios and check availability only -- never loads a checkpoint, never runs
    inference. Returns one row per scenario, in file order."""
    available_study_ids: set = set()
    availability_error: Optional[str] = None
    if data_root is not None:
        try:
            available_study_ids = _local_availability(data_root)
        except (RSNADiscoveryError, OSError) as exc:
            availability_error = f"{type(exc).__name__}: {exc}"

    rows = []
    for scenario in scenario_set.scenarios:
        available = scenario["study_id"] in available_study_ids if data_root is not None else False
        rows.append({
            "scenario_id": scenario["scenario_id"],
            "title": scenario["title"],
            "study_id": scenario["study_id"],
            "available_local": available,
            "would_run": available,
            "availability_check_error": availability_error,
        })
    return rows


def _run_one_scenario(
    scenario: dict, *, data_root: Path, model, available_study_ids: set, threshold: float,
) -> ScenarioRunResult:
    scenario_id = scenario["scenario_id"]
    title = scenario["title"]
    study_id = scenario["study_id"]
    positive_targets = list(scenario["positive_targets"])
    label_vector = list(scenario["label_vector"]) if "label_vector" in scenario else None

    if study_id not in available_study_ids:
        return ScenarioRunResult(
            scenario_id=scenario_id, title=title, study_id=study_id, available_local=False,
            run_status=RUN_STATUS_UNAVAILABLE, positive_targets=positive_targets,
            label_vector=label_vector, model_probabilities=None, predicted_positive_targets=None,
            latency_seconds=None, series_uid=None, plane=None,
            warnings=[
                "Benchmark metadata loaded. The underlying MRI study is not available on this "
                "machine, so live inference cannot be run for this scenario."
            ],
        )

    try:
        resolved_study_uid, series_uid, plane = pick_study_and_series(data_root, study_id)
        series_dir = data_root / "raw" / "train_series" / resolved_study_uid / series_uid
        started = time.monotonic()
        inference = run_study_inference(model, series_dir)
        latency_seconds = time.monotonic() - started
    except Exception as exc:  # noqa: BLE001 -- one bad scenario must never abort the run
        return ScenarioRunResult(
            scenario_id=scenario_id, title=title, study_id=study_id, available_local=True,
            run_status=RUN_STATUS_ERROR, positive_targets=positive_targets,
            label_vector=label_vector, model_probabilities=None, predicted_positive_targets=None,
            latency_seconds=None, series_uid=None, plane=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    probabilities = inference["probabilities"]
    predicted_positive = sorted(target for target, prob in probabilities.items() if prob >= threshold)
    return ScenarioRunResult(
        scenario_id=scenario_id, title=title, study_id=resolved_study_uid, available_local=True,
        run_status=RUN_STATUS_SUCCESS, positive_targets=positive_targets, label_vector=label_vector,
        model_probabilities=probabilities, predicted_positive_targets=predicted_positive,
        latency_seconds=latency_seconds, series_uid=series_uid, plane=plane,
    )


def run_scenarios(
    scenario_set: ScenarioSet, *, data_root: Path, checkpoint_path: Path, device: str = "cpu",
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple:
    """Run every scenario in `scenario_set`, in file order (already AMD-001..022 order).

    Returns `(results, checkpoint_provenance, checkpoint_load_error)`. If the checkpoint
    cannot be loaded at all, every scenario whose study *is* locally available is still
    recorded -- as `inference_error` with the load failure -- rather than aborting the run;
    scenarios whose study is unavailable are recorded as `unavailable` exactly as they would
    be if the checkpoint had loaded.
    """
    try:
        available_study_ids = _local_availability(data_root)
    except (RSNADiscoveryError, OSError) as exc:
        available_study_ids = set()
        availability_error = f"{type(exc).__name__}: {exc}"
    else:
        availability_error = None

    model = None
    checkpoint_provenance: Optional[dict] = None
    checkpoint_load_error: Optional[str] = None
    if any(s["study_id"] in available_study_ids for s in scenario_set.scenarios):
        try:
            model, checkpoint_provenance = load_bridge_checkpoint(checkpoint_path, device=device)
        except BridgeError as exc:
            checkpoint_load_error = str(exc)

    results: List[ScenarioRunResult] = []
    for scenario in scenario_set.scenarios:
        if availability_error is not None:
            results.append(ScenarioRunResult(
                scenario_id=scenario["scenario_id"], title=scenario["title"], study_id=scenario["study_id"],
                available_local=False, run_status=RUN_STATUS_UNAVAILABLE,
                positive_targets=list(scenario["positive_targets"]),
                label_vector=list(scenario.get("label_vector", [])) or None,
                model_probabilities=None, predicted_positive_targets=None, latency_seconds=None,
                series_uid=None, plane=None, warnings=[f"Could not check local availability: {availability_error}"],
            ))
            continue
        if scenario["study_id"] in available_study_ids and model is None:
            results.append(ScenarioRunResult(
                scenario_id=scenario["scenario_id"], title=scenario["title"], study_id=scenario["study_id"],
                available_local=True, run_status=RUN_STATUS_ERROR,
                positive_targets=list(scenario["positive_targets"]),
                label_vector=list(scenario.get("label_vector", [])) or None,
                model_probabilities=None, predicted_positive_targets=None, latency_seconds=None,
                series_uid=None, plane=None, error=f"Checkpoint could not be loaded: {checkpoint_load_error}",
            ))
            continue
        results.append(_run_one_scenario(
            scenario, data_root=data_root, model=model, available_study_ids=available_study_ids, threshold=threshold,
        ))

    return results, checkpoint_provenance, checkpoint_load_error


def compute_metrics(results: List[ScenarioRunResult], targets: List[str], threshold: float) -> dict:
    """Aggregate metrics computed only over `run_status == "success"` scenarios."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    total = len(results)
    available = sum(1 for r in results if r.available_local)
    succeeded = [r for r in results if r.run_status == RUN_STATUS_SUCCESS]
    unavailable = [r for r in results if r.run_status == RUN_STATUS_UNAVAILABLE]
    failed = [r for r in results if r.run_status == RUN_STATUS_ERROR]

    coverage = {
        "total_curated_scenarios": total,
        "locally_available_scenarios": available,
        "successfully_inferred_scenarios": len(succeeded),
        "unavailable_scenarios": len(unavailable),
        "failed_scenarios": len(failed),
        "inference_coverage_pct": round(len(succeeded) / total * 100.0, 1) if total else 0.0,
    }

    per_target: Dict[str, dict] = {}
    for index, target in enumerate(targets):
        pairs = [
            (r.label_vector[index], r.model_probabilities[target])
            for r in succeeded
            if r.label_vector is not None and r.model_probabilities is not None
        ]
        labels = [p[0] for p in pairs]
        scores = [p[1] for p in pairs]
        n_positive = sum(1 for label in labels if label == 1)
        n_negative = len(labels) - n_positive
        entry: dict = {"n_samples": len(labels), "n_positive": n_positive, "n_negative": n_negative}
        if n_positive == 0 or n_negative == 0:
            entry["roc_auc"] = None
            entry["pr_auc"] = None
            entry["metric_status"] = "not_computable"
            entry["not_computable_reason"] = (
                "insufficient class variation among successfully-inferred scenarios: need at "
                f"least one positive and one negative example (n_positive={n_positive}, "
                f"n_negative={n_negative})"
            )
        elif len(labels) < MIN_SAMPLES_FOR_METRIC:
            entry["roc_auc"] = None
            entry["pr_auc"] = None
            entry["metric_status"] = "insufficient_sample_size"
            entry["not_computable_reason"] = (
                f"insufficient local sample size: only {len(labels)} successfully-inferred "
                f"example(s) for this target, fewer than the minimum of {MIN_SAMPLES_FOR_METRIC} "
                "required to report a ranking metric -- a tiny-sample AUC is not a meaningful result"
            )
        else:
            entry["roc_auc"] = float(roc_auc_score(labels, scores))
            entry["pr_auc"] = float(average_precision_score(labels, scores))
            entry["metric_status"] = "computed"
        per_target[target] = entry

    valid_roc = [e["roc_auc"] for e in per_target.values() if e["metric_status"] == "computed"]
    valid_pr = [e["pr_auc"] for e in per_target.values() if e["metric_status"] == "computed"]
    performance = {
        "threshold": threshold,
        "min_samples_for_metric": MIN_SAMPLES_FOR_METRIC,
        "per_target": per_target,
        "targets_with_valid_roc_auc": len(valid_roc),
        "targets_with_valid_pr_auc": len(valid_pr),
        "macro_roc_auc": float(np.mean(valid_roc)) if valid_roc else None,
        "macro_pr_auc": float(np.mean(valid_pr)) if valid_pr else None,
        "macro_roc_auc_not_computable_reason": None if valid_roc else (
            "not computable: no target had a sufficiently large successfully-inferred sample "
            f"(minimum {MIN_SAMPLES_FOR_METRIC}) with both a positive and a negative example "
            f"(only {len(succeeded)} scenario(s) were successfully inferred in total)"
        ),
        "macro_pr_auc_not_computable_reason": None if valid_pr else (
            "not computable: no target had a sufficiently large successfully-inferred sample "
            f"(minimum {MIN_SAMPLES_FOR_METRIC}) with at least one positive example "
            f"(only {len(succeeded)} scenario(s) were successfully inferred in total)"
        ),
    }

    latencies = [r.latency_seconds for r in succeeded if r.latency_seconds is not None]
    runtime = {
        "n_timed_runs": len(latencies),
        "mean_seconds": float(np.mean(latencies)) if latencies else None,
        "median_seconds": float(np.median(latencies)) if latencies else None,
        "min_seconds": float(min(latencies)) if latencies else None,
        "max_seconds": float(max(latencies)) if latencies else None,
        "total_wall_seconds": float(sum(latencies)) if latencies else None,
    }

    mean_probability_per_target = {}
    for target in targets:
        probs = [r.model_probabilities[target] for r in succeeded if r.model_probabilities is not None]
        mean_probability_per_target[target] = float(np.mean(probs)) if probs else None

    disagreements = 0
    for r in succeeded:
        if r.model_probabilities is None or r.label_vector is None:
            continue
        for index, target in enumerate(targets):
            predicted = r.model_probabilities[target] >= threshold
            actual = bool(r.label_vector[index])
            if predicted != actual:
                disagreements += 1

    model_output_summary = {
        "mean_probability_per_target": mean_probability_per_target,
        "label_prediction_disagreements_at_threshold": disagreements,
        "threshold": threshold,
        "calibration_summary_not_computable_reason": (
            None if len(succeeded) >= 10 else
            f"not computable: only {len(succeeded)} scenario(s) were successfully inferred, too "
            "few for a meaningful calibration-style summary"
        ),
    }

    return {"coverage": coverage, "performance": performance, "runtime": runtime, "model_output_summary": model_output_summary}


def build_report(
    *, scenario_set: ScenarioSet, results: List[ScenarioRunResult], checkpoint_path: Path,
    checkpoint_provenance: Optional[dict], checkpoint_load_error: Optional[str], data_root: Path,
    device: str, threshold: float,
) -> dict:
    metrics = compute_metrics(results, list(scenario_set.targets), threshold)
    checkpoint_section = {"checkpoint_path": str(checkpoint_path), "load_error": checkpoint_load_error}
    try:
        checkpoint_section["checkpoint_sha256"] = sha256_of_file(checkpoint_path) if checkpoint_path.is_file() else None
    except OSError:
        checkpoint_section["checkpoint_sha256"] = None
    if checkpoint_provenance:
        checkpoint_section.update(checkpoint_provenance)

    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_set": scenario_set.scenario_set,
        "benchmark_run_id": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "device": device,
        },
        "checkpoint": checkpoint_section,
        "data_root": str(data_root),
        "non_diagnostic_notice": NON_DIAGNOSTIC_NOTICE,
        "coverage": metrics["coverage"],
        "performance": metrics["performance"],
        "runtime": metrics["runtime"],
        "model_output_summary": metrics["model_output_summary"],
        "scenario_results": [r.to_dict() for r in results],
        "limitations": [
            "Only scenarios whose underlying study is present under the configured data_root "
            "are inferred; metrics never include unavailable or failed scenarios.",
            "ROC-AUC/PR-AUC per target require at least one positive and one negative "
            "successfully-inferred example; otherwise they are reported as not computable.",
            "The checkpoint is an externally trained, unevaluated research model bridged into "
            "this adapter (see hackathon/amd/kaggle_mil_bridge.py); probabilities are raw, "
            "uncalibrated sigmoid outputs.",
            COVERAGE_NOTE,
        ],
        "disclaimer": DISCLAIMER,
    }


def render_markdown(report: dict) -> str:
    coverage = report["coverage"]
    performance = report["performance"]
    runtime = report["runtime"]
    lines = [
        "# AMD curated benchmark results",
        "",
        f"*{report['disclaimer']}*",
        "",
        f"Run: `{report['benchmark_run_id']}`  ·  scenario set: `{report['scenario_set']}`  ·  "
        f"schema: `{report['schema_version']}`",
        "",
        "## Coverage",
        "",
        f"- Total curated scenarios: **{coverage['total_curated_scenarios']}**",
        f"- Locally available: **{coverage['locally_available_scenarios']}**",
        f"- Successfully inferred: **{coverage['successfully_inferred_scenarios']}**",
        f"- Unavailable: **{coverage['unavailable_scenarios']}**",
        f"- Failed: **{coverage['failed_scenarios']}**",
        f"- Inference coverage: **{coverage['inference_coverage_pct']}%**",
        "",
        COVERAGE_NOTE,
        "",
        "## Model ranking / classification metrics",
        "",
        f"Threshold: `{performance['threshold']}`  ·  denominator: "
        f"**{coverage['successfully_inferred_scenarios']}** successfully-inferred scenario(s).",
        "",
    ]
    if performance["macro_roc_auc"] is not None:
        lines.append(f"- Macro ROC-AUC ({performance['targets_with_valid_roc_auc']} valid targets): "
                      f"**{performance['macro_roc_auc']:.3f}**")
    else:
        lines.append(f"- Macro ROC-AUC: not computable -- {performance['macro_roc_auc_not_computable_reason']}")
    if performance["macro_pr_auc"] is not None:
        lines.append(f"- Macro PR-AUC ({performance['targets_with_valid_pr_auc']} valid targets): "
                      f"**{performance['macro_pr_auc']:.3f}**")
    else:
        lines.append(f"- Macro PR-AUC: not computable -- {performance['macro_pr_auc_not_computable_reason']}")
    lines += ["", "| Target | n | pos | neg | ROC-AUC | PR-AUC |", "| --- | --- | --- | --- | --- | --- |"]
    for target, entry in performance["per_target"].items():
        roc = f"{entry['roc_auc']:.3f}" if entry["roc_auc"] is not None else "not computable"
        pr = f"{entry['pr_auc']:.3f}" if entry["pr_auc"] is not None else "not computable"
        lines.append(f"| {target} | {entry['n_samples']} | {entry['n_positive']} | {entry['n_negative']} | {roc} | {pr} |")

    lines += ["", "## Runtime (successfully-inferred scenarios only)", ""]
    if runtime["n_timed_runs"]:
        lines += [
            f"- Mean latency: **{runtime['mean_seconds']:.3f}s**",
            f"- Median latency: **{runtime['median_seconds']:.3f}s**",
            f"- Min / max latency: **{runtime['min_seconds']:.3f}s** / **{runtime['max_seconds']:.3f}s**",
            f"- Total wall time: **{runtime['total_wall_seconds']:.3f}s** over {runtime['n_timed_runs']} run(s)",
        ]
    else:
        lines.append("- not computable: no scenario was successfully inferred")

    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    lines += ["", f"*{DISCLAIMER}*", ""]
    return "\n".join(lines)
