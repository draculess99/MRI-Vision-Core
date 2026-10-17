"""Loader/validator for `benchmarks/amd/amd_demo_scenarios.json` -- the 22 curated AMD demo
scenarios (see `benchmarks/amd/CONSUMER.md` for the authoritative contract this module
implements).

Deliberately imports nothing beyond the standard library (`json`, `pathlib`, `dataclasses`,
`typing`) -- no `torch`, `pydicom`, `streamlit`, or any inference code. Browsing scenario
metadata must never require those optional, heavier dependencies to be installed or
importable; only *running* inference on a resolved study needs them, and that happens
elsewhere, through the existing real-inference code path -- this module never calls it.

Nothing here reads, writes, or knows about DICOM files, model checkpoints, or the local RSNA
data root. It resolves exactly one file (the scenario JSON) and returns plain data. Whether a
given scenario's `study_id` is actually present on this machine is a separate question this
module does not answer -- see `hackathon/amd/streamlit_app.py`'s use of the existing
`mri_core.rsna_integration.discover_available_studies()` for that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS_PATH = _REPO_ROOT / "benchmarks" / "amd" / "amd_demo_scenarios.json"

REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "scenario_set", "scenario_count", "disclaimer", "targets", "scenarios"}
)
# Matches tests/test_amd_demo_scenarios.py's own REQUIRED set and benchmarks/amd/CONSUMER.md.
REQUIRED_SCENARIO_FIELDS = frozenset(
    {
        "scenario_id", "title", "study_id", "scenario_type", "why_selected", "positive_targets",
        "negative_target_count", "positive_target_count", "series_count", "dicom_count",
        "planes", "tags", "presentation_tag", "expected_demo_behavior", "data_source",
    }
)


class BenchmarkScenarioError(Exception):
    """Base class for scenario-loading failures. Every message here is written to be shown
    directly in the Streamlit UI -- never a raw traceback, never an unhandled exception type."""


class ScenarioFileMissingError(BenchmarkScenarioError):
    """The scenario JSON file does not exist (or could not be read) at the expected path."""


class ScenarioFileMalformedError(BenchmarkScenarioError):
    """The file exists but is not valid JSON, or does not match the required schema."""


@dataclass(frozen=True)
class ScenarioSet:
    schema_version: str
    scenario_set: str
    scenario_count: int
    disclaimer: str
    targets: Tuple[str, ...]
    scenarios: Tuple[dict, ...]  # each already validated against REQUIRED_SCENARIO_FIELDS


def load_scenarios(path: Optional[Path] = None) -> ScenarioSet:
    """Load and validate the scenario file.

    Raises `ScenarioFileMissingError` if the file is absent or unreadable, or
    `ScenarioFileMalformedError` if it is not valid JSON, is not a JSON object, is missing a
    required top-level or per-scenario field, or has duplicate `scenario_id`/`title` values.
    Never raises `json.JSONDecodeError`, `KeyError`, or any other raw exception type to the
    caller -- every failure path is one of the two controlled errors above.
    """
    scenarios_path = path if path is not None else DEFAULT_SCENARIOS_PATH

    if not scenarios_path.is_file():
        raise ScenarioFileMissingError(f"Benchmark scenario file not found: {scenarios_path}")

    try:
        raw_text = scenarios_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioFileMissingError(
            f"Could not read benchmark scenario file {scenarios_path}: {exc}"
        ) from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ScenarioFileMalformedError(
            f"Benchmark scenario file is not valid JSON: {scenarios_path} ({exc})"
        ) from exc

    if not isinstance(data, dict):
        raise ScenarioFileMalformedError(
            f"Benchmark scenario file's top level is not a JSON object: {scenarios_path}"
        )

    missing_top = REQUIRED_TOP_LEVEL_FIELDS - set(data.keys())
    if missing_top:
        raise ScenarioFileMalformedError(
            f"Benchmark scenario file is missing required top-level field(s) "
            f"{sorted(missing_top)}: {scenarios_path}"
        )

    scenarios = data["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ScenarioFileMalformedError(
            f"Benchmark scenario file's 'scenarios' field is not a non-empty list: {scenarios_path}"
        )

    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ScenarioFileMalformedError(
                f"Scenario at index {index} is not a JSON object: {scenarios_path}"
            )
        missing = REQUIRED_SCENARIO_FIELDS - set(scenario.keys())
        if missing:
            raise ScenarioFileMalformedError(
                f"Scenario at index {index} ({scenario.get('scenario_id', '?')!r}) is missing "
                f"required field(s) {sorted(missing)}: {scenarios_path}"
            )

    scenario_ids = [s["scenario_id"] for s in scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ScenarioFileMalformedError(f"Duplicate scenario_id values found: {scenarios_path}")

    titles = [s["title"] for s in scenarios]
    if len(set(titles)) != len(titles):
        raise ScenarioFileMalformedError(f"Duplicate title values found: {scenarios_path}")

    return ScenarioSet(
        schema_version=str(data["schema_version"]),
        scenario_set=str(data["scenario_set"]),
        scenario_count=int(data["scenario_count"]),
        disclaimer=str(data["disclaimer"]),
        targets=tuple(data["targets"]),
        scenarios=tuple(scenarios),
    )


def scenario_label(scenario: dict) -> str:
    """The exact selectbox label format this module's consumers must use:
    'AMD-001 — <title>'. An em dash, matching benchmarks/amd/CONSUMER.md's own example."""
    return f"{scenario['scenario_id']} — {scenario['title']}"


def build_labels(scenario_set: ScenarioSet) -> List[str]:
    """One label per scenario, in file order -- suitable to pass directly to a selectbox."""
    return [scenario_label(s) for s in scenario_set.scenarios]
