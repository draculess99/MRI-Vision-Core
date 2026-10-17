"""Tests for hackathon/amd/benchmark_scenarios.py: the reusable loader/validator for
benchmarks/amd/amd_demo_scenarios.json. Deliberately does not import torch/pydicom/streamlit
at all -- this file proves the loader itself has no such dependency, by never needing one.
"""

from __future__ import annotations

import json

import pytest

from hackathon.amd.benchmark_scenarios import (
    REQUIRED_SCENARIO_FIELDS,
    REQUIRED_TOP_LEVEL_FIELDS,
    ScenarioFileMalformedError,
    ScenarioFileMissingError,
    build_labels,
    load_scenarios,
    scenario_label,
)


def _valid_doc(scenarios=None):
    scenario = {
        "scenario_id": "AMD-001", "title": "Example scenario", "study_id": "1.2.3",
        "scenario_type": "target_anchor", "why_selected": "because", "positive_targets": ["Effusion"],
        "negative_target_count": 11, "positive_target_count": 1, "series_count": 5,
        "dicom_count": 100, "planes": ["Axial"], "tags": ["low_label_count"],
        "presentation_tag": "Simple", "expected_demo_behavior": "demo", "data_source": "RSNA Knee Abnormality Detection",
    }
    return {
        "schema_version": "1.0", "scenario_set": "amd_mri_core_demo_v1", "scenario_count": 1,
        "disclaimer": "Research/demo benchmark only; not clinical accuracy or medical performance.",
        "targets": ["ACL"], "scenarios": scenarios if scenarios is not None else [scenario],
    }


class TestRealScenarioFile:
    def test_loads_successfully(self):
        scenario_set = load_scenarios()
        assert scenario_set.schema_version
        assert scenario_set.scenarios

    def test_exactly_22_scenarios(self):
        scenario_set = load_scenarios()
        assert scenario_set.scenario_count == 22
        assert len(scenario_set.scenarios) == 22

    def test_scenario_ids_are_unique(self):
        scenario_set = load_scenarios()
        ids = [s["scenario_id"] for s in scenario_set.scenarios]
        assert len(ids) == len(set(ids)) == 22

    def test_scenario_ids_are_amd_001_through_022_in_order(self):
        scenario_set = load_scenarios()
        ids = [s["scenario_id"] for s in scenario_set.scenarios]
        assert ids == [f"AMD-{i:03d}" for i in range(1, 23)]

    def test_every_scenario_has_every_required_field(self):
        scenario_set = load_scenarios()
        for scenario in scenario_set.scenarios:
            assert REQUIRED_SCENARIO_FIELDS <= set(scenario.keys())

    def test_display_labels_are_correct_format(self):
        scenario_set = load_scenarios()
        labels = build_labels(scenario_set)
        assert len(labels) == 22
        for label, scenario in zip(labels, scenario_set.scenarios):
            assert label == f"{scenario['scenario_id']} — {scenario['title']}"
        assert labels[0].startswith("AMD-001 — ")

    def test_labels_are_unique(self):
        scenario_set = load_scenarios()
        labels = build_labels(scenario_set)
        assert len(labels) == len(set(labels))

    def test_disclaimer_mentions_research_demo_only(self):
        scenario_set = load_scenarios()
        assert "research" in scenario_set.disclaimer.lower() or "demo" in scenario_set.disclaimer.lower()

    def test_loading_touches_only_the_json_file_not_any_optional_dependency(self):
        """Regression-style guard: importing/using this module must never require torch,
        pydicom, or streamlit. The real guarantee is structural (see the module's own
        imports, checked below) -- this does not pop torch/pydicom/streamlit out of
        sys.modules, since doing so without restoring them afterward corrupts process-wide
        state (e.g. streamlit's internal DeltaGeneratorSingleton) for any AppTest-based test
        that runs later in the same pytest session, such as tests/test_amd_streamlit_app.py
        when the whole test suite is collected together."""
        load_scenarios()
        import hackathon.amd.benchmark_scenarios as module
        import inspect

        source = inspect.getsource(module)
        for forbidden in ("import torch", "import pydicom", "import streamlit"):
            assert forbidden not in source


class TestScenarioLabel:
    def test_format(self):
        scenario = {"scenario_id": "AMD-005", "title": "Some title"}
        assert scenario_label(scenario) == "AMD-005 — Some title"


class TestMissingFile:
    def test_missing_file_raises_controlled_error(self, tmp_path):
        with pytest.raises(ScenarioFileMissingError):
            load_scenarios(tmp_path / "does_not_exist.json")

    def test_missing_file_error_message_is_user_friendly(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(ScenarioFileMissingError, match=r"not found"):
            load_scenarios(missing)


class TestMalformedJson:
    def test_invalid_json_syntax_raises_controlled_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ScenarioFileMalformedError, match="not valid JSON"):
            load_scenarios(bad)

    def test_top_level_not_an_object_raises_controlled_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ScenarioFileMalformedError):
            load_scenarios(bad)

    def test_missing_top_level_field_raises_controlled_error(self, tmp_path):
        doc = _valid_doc()
        del doc["disclaimer"]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ScenarioFileMalformedError, match="disclaimer"):
            load_scenarios(path)

    def test_missing_per_scenario_field_raises_controlled_error(self, tmp_path):
        doc = _valid_doc()
        del doc["scenarios"][0]["why_selected"]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ScenarioFileMalformedError, match="why_selected"):
            load_scenarios(path)

    def test_empty_scenarios_list_raises_controlled_error(self, tmp_path):
        doc = _valid_doc(scenarios=[])
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ScenarioFileMalformedError):
            load_scenarios(path)

    def test_duplicate_scenario_ids_raise_controlled_error(self, tmp_path):
        doc = _valid_doc()
        doc["scenarios"].append(dict(doc["scenarios"][0]))
        doc["scenarios"][1]["title"] = "A different title"
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ScenarioFileMalformedError, match="Duplicate scenario_id"):
            load_scenarios(path)

    def test_duplicate_titles_raise_controlled_error(self, tmp_path):
        doc = _valid_doc()
        duplicate = dict(doc["scenarios"][0])
        duplicate["scenario_id"] = "AMD-002"
        doc["scenarios"].append(duplicate)
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ScenarioFileMalformedError, match="Duplicate title"):
            load_scenarios(path)

    def test_valid_minimal_document_loads_cleanly(self, tmp_path):
        doc = _valid_doc()
        path = tmp_path / "good.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        scenario_set = load_scenarios(path)
        assert scenario_set.scenario_count == 1
        assert len(scenario_set.scenarios) == 1
