"""Smoke tests for the AMD adapter (hackathon/amd).

The default tests use the synthetic checkpoint + tiny synthetic DICOM study from the shared
engine tests, to verify plumbing and output structure only -- the numbers they produce are
never real MRI predictions or benchmark results. The real-data test at the bottom runs only
when AMD_CHECKPOINT and AMD_RSNA_ROOT point at an actual trained checkpoint and downloaded
RSNA data; it skips otherwise.
"""

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pandas")

from hackathon.amd import benchmark, rocm_env, run_inference
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

from .test_rsna_inference_engine import STUDY, make_checkpoint, rsna  # noqa: F401  (fixture)


class TestRocmEnv:
    def test_report_is_self_consistent(self):
        report = rocm_env.check_environment()
        assert report["amd_gpu_available"] == (
            report["is_rocm_build"] and report["accelerator_available"] and report["device_count"] > 0
        )
        if not report["amd_gpu_available"]:
            assert report["unavailable_reasons"]
        json.dumps(report)

    def test_require_amd_exit_code_reflects_reality(self):
        expected = 0 if rocm_env.check_environment()["amd_gpu_available"] else 2
        assert rocm_env.main(["--require-amd"]) == expected


class TestRunInference:
    def test_writes_separated_sections(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        out = tmp_path / "out" / "prediction.json"
        code = run_inference.main([
            "--data-root", str(rsna.meta), "--checkpoint", str(checkpoint),
            "--dicom-root", str(rsna.raw), "--study-uid", STUDY, "--output", str(out),
        ])
        assert code == 0
        report = json.loads(out.read_text())

        prediction = report["prediction"]
        assert prediction["status"] == "ok"
        assert list(prediction["probabilities"]) == list(TARGET_COLUMNS)

        evidence = report["evidence_review"]
        assert "probabilities" not in evidence
        assert evidence["quality_gate_status"] == "NOT_ASSESSED"
        assert evidence["uncertainty"]["status"] == "not_estimated"

        gate = report["human_review_gate"]
        assert gate["state"] == "PENDING_REVIEW" and gate["history"] == [] and gate["simulated"] is True
        assert gate["evidence_fingerprint"] == evidence["evidence_fingerprint"]

    def test_missing_checkpoint_fails_without_output(self, rsna, tmp_path):
        out = tmp_path / "prediction.json"
        code = run_inference.main([
            "--data-root", str(rsna.meta), "--checkpoint", str(tmp_path / "nope.pt"),
            "--dicom-root", str(rsna.raw), "--study-uid", STUDY, "--output", str(out),
        ])
        assert code == 1 and not out.exists()


class TestBenchmark:
    def test_cpu_measured_and_gpu_reported_honestly(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        out = tmp_path / "benchmark.json"
        code = benchmark.main([
            "--data-root", str(rsna.meta), "--checkpoint", str(checkpoint), "--dicom-root", str(rsna.raw), "--study-uid", STUDY,
            "--iterations", "3", "--warmup", "1", "--batch-sizes", "1", "2", "--output", str(out),
        ])
        assert code == 0
        report = json.loads(out.read_text())

        cpu_b1 = report["cpu"]["batches"]["1"]
        assert cpu_b1["latency"]["median_ms"] > 0 and cpu_b1["throughput_studies_per_s"] > 0
        assert set(report["cpu"]["batches"]) == {"1", "2"}
        assert report["workload"]["checkpoint_sha256"]

        if report["environment"]["amd_gpu_available"]:
            assert report["gpu"]["is_amd_rocm"] is True
            assert report["gpu"]["gpu"]["peak_allocated_bytes"] >= 0
            assert report["comparison"]["gpu_speedup_vs_cpu"] > 0
        else:
            assert report["gpu"]["status"] == "unavailable" and report["gpu"]["reasons"]
            assert "batches" not in report["gpu"]
            assert report["comparison"]["status"] == "not_computed"

    def test_require_gpu_exit_code(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        code = benchmark.main([
            "--data-root", str(rsna.meta), "--checkpoint", str(checkpoint), "--dicom-root", str(rsna.raw), "--study-uid", STUDY,
            "--iterations", "1", "--warmup", "0", "--require-gpu", "--output", str(tmp_path / "b.json"),
        ])
        assert code == (0 if rocm_env.check_environment()["amd_gpu_available"] else 2)


@pytest.mark.skipif(
    not (os.environ.get("AMD_CHECKPOINT") and os.environ.get("AMD_RSNA_ROOT")),
    reason="set AMD_CHECKPOINT and AMD_RSNA_ROOT to run against a real trained checkpoint and real MRI data",
)
def test_real_checkpoint_real_study(tmp_path):
    out = tmp_path / "prediction.json"
    code = run_inference.main([
        "--data-root", os.environ["AMD_RSNA_ROOT"], "--checkpoint", os.environ["AMD_CHECKPOINT"],
        "--output", str(out),
    ])
    assert code == 0
    report = json.loads(out.read_text())
    assert report["prediction"]["status"] == "ok"
    assert len(report["prediction"]["probabilities"]) == 12
