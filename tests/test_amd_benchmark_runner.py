"""Tests for hackathon/amd/benchmark_runner.py: the automated 22-scenario benchmark runner.

Uses a small synthetic RSNA-shaped data root (same `tests/dicom_factory.py` helper as
`test_amd_demo_scenarios.py`) and mocks the model-loading/inference calls -- this file never
needs a real checkpoint, torch weights, GPU, network access, or an API key. It proves the
runner's *bookkeeping* (availability gating, never fabricating, continuing past one failure,
denominator correctness, "not computable" vs "insufficient sample size" metric gating, schema
stability) independently of whether the real bridged model happens to be present on a given
machine.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mri_core.rsna_knee_dataset import SERIES_COLUMNS, SUBMISSION_COLUMNS, TARGET_COLUMNS

from .dicom_factory import AXIAL, write_series

import hackathon.amd.benchmark_runner as runner
from hackathon.amd.benchmark_scenarios import ScenarioSet
from hackathon.amd.kaggle_mil_bridge import BridgeError

AVAILABLE_STUDY = "1.2.826.0.1.3680043.8.498.0001"
UNAVAILABLE_STUDY = "1.2.826.0.1.3680043.8.498.9999"
AVAILABLE_SERIES = f"{AVAILABLE_STUDY}.s0"


def _scenario(scenario_id: str, study_id: str, label_vector: list, positive_targets: list) -> dict:
    return {
        "scenario_id": scenario_id, "title": f"Title for {scenario_id}", "study_id": study_id,
        "scenario_type": "target_anchor", "why_selected": "test fixture", "positive_targets": positive_targets,
        "negative_target_count": len(TARGET_COLUMNS) - len(positive_targets),
        "positive_target_count": len(positive_targets), "series_count": 1, "dicom_count": 4,
        "planes": ["Axial"], "tags": [], "presentation_tag": "Simple", "expected_demo_behavior": "demo",
        "data_source": "RSNA Knee Abnormality Detection", "label_vector": label_vector,
    }


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "rsna"
    write_series(
        root / "raw" / "train_series" / AVAILABLE_STUDY / AVAILABLE_SERIES, count=4, orientation=AXIAL,
        series_uid=AVAILABLE_SERIES, study_uid=AVAILABLE_STUDY, rows=8, columns=8,
    )
    label_values = {t: 0.0 for t in TARGET_COLUMNS}
    label_values["Effusion"] = 1.0
    train = pd.DataFrame({"StudyInstanceUID": [AVAILABLE_STUDY], "Report": ["r"], **{t: [v] for t, v in label_values.items()}})
    train.to_csv(root / "train.csv", index=False)
    pd.DataFrame([[AVAILABLE_STUDY, AVAILABLE_SERIES, 0, 0, "Axial"]], columns=SERIES_COLUMNS).to_csv(
        root / "train_series.csv", index=False
    )
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(root / "test.csv", index=False)
    pd.DataFrame(columns=SERIES_COLUMNS).to_csv(root / "test_series.csv", index=False)
    pd.DataFrame(columns=SUBMISSION_COLUMNS).to_csv(root / "sample_submission.csv", index=False)
    return root


@pytest.fixture
def scenario_set() -> ScenarioSet:
    available_labels = [0] * len(TARGET_COLUMNS)
    available_labels[TARGET_COLUMNS.index("Effusion")] = 1
    unavailable_labels = [0] * len(TARGET_COLUMNS)
    unavailable_labels[TARGET_COLUMNS.index("ACL")] = 1
    scenarios = (
        _scenario("AMD-001", AVAILABLE_STUDY, available_labels, ["Effusion"]),
        _scenario("AMD-002", UNAVAILABLE_STUDY, unavailable_labels, ["ACL"]),
    )
    return ScenarioSet(
        schema_version="1.0", scenario_set="amd_mri_core_demo_v1_test", scenario_count=2,
        disclaimer="Research/demo benchmark only; not clinical accuracy or medical performance.",
        targets=tuple(TARGET_COLUMNS), scenarios=scenarios,
    )


def _fake_probabilities(positive=()) -> dict:
    return {t: (0.9 if t in positive else 0.1) for t in TARGET_COLUMNS}


@pytest.fixture
def mock_model(monkeypatch):
    """Replaces the real checkpoint load + forward pass with deterministic fakes -- no torch
    weights, no timm model, no GPU required."""
    sentinel_model = object()
    sentinel_provenance = {"architecture": "fake", "source": "test fixture"}

    def fake_load(path, *, device="cpu"):
        return sentinel_model, sentinel_provenance

    def fake_run(model, series_dir):
        assert model is sentinel_model
        return {"probabilities": _fake_probabilities(positive=("Effusion",)), "n_slices": 4, "image_size": 8}

    monkeypatch.setattr(runner, "load_bridge_checkpoint", fake_load)
    monkeypatch.setattr(runner, "run_study_inference", fake_run)
    return sentinel_model, sentinel_provenance


class TestLoadsAllScenarios:
    def test_loads_all_22_real_scenarios_in_order(self):
        from hackathon.amd.benchmark_scenarios import load_scenarios

        real_set = load_scenarios()
        assert real_set.scenario_count == 22
        ids = [s["scenario_id"] for s in real_set.scenarios]
        assert ids == [f"AMD-{i:03d}" for i in range(1, 23)]


class TestUnavailableStudyNeverFabricates:
    def test_unavailable_study_recorded_not_fabricated(self, data_root, scenario_set, mock_model):
        results, _, checkpoint_error = runner.run_scenarios(
            scenario_set, data_root=data_root, checkpoint_path=data_root / "fake.ckpt",
        )
        assert checkpoint_error is None
        by_id = {r.scenario_id: r for r in results}
        unavailable = by_id["AMD-002"]
        assert unavailable.available_local is False
        assert unavailable.run_status == runner.RUN_STATUS_UNAVAILABLE
        assert unavailable.model_probabilities is None
        assert unavailable.predicted_positive_targets is None
        assert unavailable.latency_seconds is None
        assert any("not available on this machine" in w for w in unavailable.warnings)

    def test_scenario_ordering_is_preserved(self, data_root, scenario_set, mock_model):
        results, _, _ = runner.run_scenarios(scenario_set, data_root=data_root, checkpoint_path=data_root / "fake.ckpt")
        assert [r.scenario_id for r in results] == ["AMD-001", "AMD-002"]


class TestInferenceFailureDoesNotAbortRun:
    def test_one_scenario_failing_does_not_stop_the_others(self, data_root, scenario_set, monkeypatch):
        sentinel_model = object()

        def fake_load(path, *, device="cpu"):
            return sentinel_model, {}

        def flaky_run(model, series_dir):
            raise BridgeError("simulated decode failure")

        monkeypatch.setattr(runner, "load_bridge_checkpoint", fake_load)
        monkeypatch.setattr(runner, "run_study_inference", flaky_run)

        results, _, checkpoint_error = runner.run_scenarios(scenario_set, data_root=data_root, checkpoint_path=data_root / "fake.ckpt")
        assert checkpoint_error is None
        assert len(results) == 2
        failed = next(r for r in results if r.scenario_id == "AMD-001")
        assert failed.run_status == runner.RUN_STATUS_ERROR
        assert "simulated decode failure" in failed.error
        assert failed.model_probabilities is None


class TestSuccessfulMockInference:
    def test_records_probabilities_and_latency(self, data_root, scenario_set, mock_model):
        results, provenance, checkpoint_error = runner.run_scenarios(
            scenario_set, data_root=data_root, checkpoint_path=data_root / "fake.ckpt",
        )
        assert checkpoint_error is None
        assert provenance == mock_model[1]
        success = next(r for r in results if r.scenario_id == "AMD-001")
        assert success.run_status == runner.RUN_STATUS_SUCCESS
        assert success.model_probabilities["Effusion"] == 0.9
        assert success.predicted_positive_targets == ["Effusion"]
        assert isinstance(success.latency_seconds, float) and success.latency_seconds >= 0.0


class TestDenominatorsAndMetricGating:
    def test_denominator_counts_are_correct(self, data_root, scenario_set, mock_model):
        results, _, _ = runner.run_scenarios(scenario_set, data_root=data_root, checkpoint_path=data_root / "fake.ckpt")
        metrics = runner.compute_metrics(results, list(scenario_set.targets), runner.DEFAULT_THRESHOLD)
        coverage = metrics["coverage"]
        assert coverage["total_curated_scenarios"] == 2
        assert coverage["locally_available_scenarios"] == 1
        assert coverage["successfully_inferred_scenarios"] == 1
        assert coverage["unavailable_scenarios"] == 1
        assert coverage["failed_scenarios"] == 0

    def test_unavailable_case_excluded_from_performance_metrics(self, data_root, scenario_set, mock_model):
        results, _, _ = runner.run_scenarios(scenario_set, data_root=data_root, checkpoint_path=data_root / "fake.ckpt")
        metrics = runner.compute_metrics(results, list(scenario_set.targets), runner.DEFAULT_THRESHOLD)
        acl_entry = metrics["performance"]["per_target"]["ACL"]
        # AMD-002 (the only ACL-positive scenario) was never inferred -- ACL must show 0 positives.
        assert acl_entry["n_samples"] == 1
        assert acl_entry["n_positive"] == 0

    def test_insufficient_sample_size_reports_not_computable_not_a_number(self, data_root, scenario_set, mock_model):
        results, _, _ = runner.run_scenarios(scenario_set, data_root=data_root, checkpoint_path=data_root / "fake.ckpt")
        metrics = runner.compute_metrics(results, list(scenario_set.targets), runner.DEFAULT_THRESHOLD)
        performance = metrics["performance"]
        assert performance["macro_roc_auc"] is None
        assert "insufficient" in performance["macro_roc_auc_not_computable_reason"] or \
               "not computable" in performance["macro_roc_auc_not_computable_reason"]
        for entry in performance["per_target"].values():
            assert entry["roc_auc"] is None
            assert entry["metric_status"] in ("not_computable", "insufficient_sample_size")


class TestOutputSchemaStability:
    def test_json_report_has_the_documented_top_level_keys(self, data_root, scenario_set, mock_model):
        results, provenance, checkpoint_error = runner.run_scenarios(
            scenario_set, data_root=data_root, checkpoint_path=data_root / "fake.ckpt",
        )
        report = runner.build_report(
            scenario_set=scenario_set, results=results, checkpoint_path=data_root / "fake.ckpt",
            checkpoint_provenance=provenance, checkpoint_load_error=checkpoint_error, data_root=data_root,
            device="cpu", threshold=runner.DEFAULT_THRESHOLD,
        )
        for key in (
            "schema_version", "scenario_set", "benchmark_run_id", "environment", "checkpoint", "data_root",
            "non_diagnostic_notice", "coverage", "performance", "runtime", "model_output_summary",
            "scenario_results", "limitations", "disclaimer",
        ):
            assert key in report
        assert report["disclaimer"] == "Research/demo benchmark only; not clinical accuracy or medical performance."
        # Round-trips through JSON cleanly (no NaN/Infinity, no non-serializable objects).
        reserialized = json.loads(json.dumps(report))
        assert reserialized["scenario_set"] == report["scenario_set"]

    def test_markdown_summary_includes_denominator_and_disclaimer(self, data_root, scenario_set, mock_model):
        results, provenance, checkpoint_error = runner.run_scenarios(
            scenario_set, data_root=data_root, checkpoint_path=data_root / "fake.ckpt",
        )
        report = runner.build_report(
            scenario_set=scenario_set, results=results, checkpoint_path=data_root / "fake.ckpt",
            checkpoint_provenance=provenance, checkpoint_load_error=checkpoint_error, data_root=data_root,
            device="cpu", threshold=runner.DEFAULT_THRESHOLD,
        )
        markdown = runner.render_markdown(report)
        assert "Research/demo benchmark only; not clinical accuracy or medical performance." in markdown
        assert "**1** successfully-inferred scenario(s)" in markdown
        assert "Locally available: **1**" in markdown

    def test_no_raw_dicom_data_is_serialized(self, data_root, scenario_set, mock_model):
        results, provenance, checkpoint_error = runner.run_scenarios(
            scenario_set, data_root=data_root, checkpoint_path=data_root / "fake.ckpt",
        )
        report = runner.build_report(
            scenario_set=scenario_set, results=results, checkpoint_path=data_root / "fake.ckpt",
            checkpoint_provenance=provenance, checkpoint_load_error=checkpoint_error, data_root=data_root,
            device="cpu", threshold=runner.DEFAULT_THRESHOLD,
        )
        text = json.dumps(report)
        assert ".dcm" not in text
        assert "PixelData" not in text
        assert "base64" not in text.lower()


class TestDryRun:
    def test_dry_run_never_calls_inference(self, data_root, scenario_set, monkeypatch):
        def must_not_be_called(*args, **kwargs):
            raise AssertionError("dry run must never load a checkpoint or run inference")

        monkeypatch.setattr(runner, "load_bridge_checkpoint", must_not_be_called)
        monkeypatch.setattr(runner, "run_study_inference", must_not_be_called)

        rows = runner.dry_run_report(scenario_set, data_root)
        assert len(rows) == 2
        by_id = {row["scenario_id"]: row for row in rows}
        assert by_id["AMD-001"]["available_local"] is True
        assert by_id["AMD-001"]["would_run"] is True
        assert by_id["AMD-002"]["available_local"] is False
        assert by_id["AMD-002"]["would_run"] is False

    def test_dry_run_works_with_no_data_root_configured(self, scenario_set):
        rows = runner.dry_run_report(scenario_set, None)
        assert all(row["available_local"] is False for row in rows)
        assert all(row["would_run"] is False for row in rows)


class TestSingleScenarioMode:
    def test_running_only_one_scenario_id_works(self, data_root, scenario_set, mock_model):
        from dataclasses import replace

        only_first = replace(scenario_set, scenarios=(scenario_set.scenarios[0],), scenario_count=1)
        results, _, _ = runner.run_scenarios(only_first, data_root=data_root, checkpoint_path=data_root / "fake.ckpt")
        assert len(results) == 1
        assert results[0].scenario_id == "AMD-001"
        assert results[0].run_status == runner.RUN_STATUS_SUCCESS


class TestNoNetworkOrCredentialsRequired:
    def test_module_imports_no_llm_or_provider_code(self):
        import inspect

        source = inspect.getsource(runner)
        for forbidden in ("llm_config", "llm_providers", "providers", "import requests"):
            assert forbidden not in source

    def test_run_scenarios_needs_no_api_key_env_vars(self, data_root, scenario_set, mock_model, monkeypatch):
        for key in ("GROQ_API_KEY", "AMD_TOKENFACTORY_API_KEY", "AMD_TOKENFACTORY_BASE_URL"):
            monkeypatch.delenv(key, raising=False)
        results, _, checkpoint_error = runner.run_scenarios(scenario_set, data_root=data_root, checkpoint_path=data_root / "fake.ckpt")
        assert checkpoint_error is None
        assert any(r.run_status == runner.RUN_STATUS_SUCCESS for r in results)


class TestCheckpointLoadFailure:
    def test_checkpoint_load_failure_is_recorded_not_raised(self, data_root, scenario_set, monkeypatch):
        def failing_load(path, *, device="cpu"):
            raise BridgeError("no such checkpoint")

        monkeypatch.setattr(runner, "load_bridge_checkpoint", failing_load)
        results, provenance, checkpoint_error = runner.run_scenarios(
            scenario_set, data_root=data_root, checkpoint_path=data_root / "fake.ckpt",
        )
        assert checkpoint_error == "no such checkpoint"
        assert provenance is None
        available_scenario = next(r for r in results if r.scenario_id == "AMD-001")
        assert available_scenario.run_status == runner.RUN_STATUS_ERROR
        assert "Checkpoint could not be loaded" in available_scenario.error
        unavailable_scenario = next(r for r in results if r.scenario_id == "AMD-002")
        assert unavailable_scenario.run_status == runner.RUN_STATUS_UNAVAILABLE
