"""Deterministic tests for hackathon/amd/benchmark_kaggle_bridge.py.

Uses a synthetic DICOM study (tests/dicom_factory.py, the same generator the shared engine's
own tests use) and a freshly-instantiated, untrained KneeMILClassifier saved in the exact
Lightning checkpoint shape load_bridge_checkpoint expects -- no real Kaggle checkpoint or real
RSNA download is needed. Numbers from these fixtures are plumbing checks only, never real
predictions or real benchmarks (see tests/test_amd_kaggle_bridge.py for the real-asset-gated
version of that).
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("pydicom")
pd = pytest.importorskip("pandas")

from hackathon.amd import benchmark_kaggle_bridge, rocm_env
from hackathon.amd.kaggle_mil_bridge import BACKBONE_NAME, KneeMILClassifier, TARGET_COLUMNS

from .dicom_factory import SAGITTAL, write_series

STUDY = "1.2.826.1"
SERIES = "1.2.826.2.3"


@pytest.fixture
def bridge_data_root(tmp_path):
    """A data_root laid out exactly like a real RSNA knee dir: train.csv at the root,
    raw/train_series/<study>/<series>/*.dcm nested underneath -- the layout
    pick_study_and_series (reused from run_real_checkpoint_smoke) expects."""
    root = tmp_path / "data"
    root.mkdir()
    pd.DataFrame([{"StudyInstanceUID": STUDY, "Report": "r", **{t: i % 2 for i, t in enumerate(TARGET_COLUMNS)}}]) \
        .to_csv(root / "train.csv", index=False)
    pd.DataFrame(
        [(STUDY, SERIES, 1, 0, "Sagittal")],
        columns=("StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"),
    ).to_csv(root / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(root / "test.csv", index=False)
    pd.DataFrame(columns=("StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane")) \
        .to_csv(root / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [], **{t: [] for t in TARGET_COLUMNS}}).to_csv(root / "sample_submission.csv", index=False)
    write_series(root / "raw" / "train_series" / STUDY / SERIES, count=3, orientation=SAGITTAL, series_uid=SERIES, study_uid=STUDY)
    return root


def make_bridge_checkpoint(path):
    """A minimal Lightning-shaped checkpoint for an untrained KneeMILClassifier -- correctly
    shaped weights, not trained ones. Only used to exercise the benchmark plumbing."""
    model = KneeMILClassifier(model_name=BACKBONE_NAME, num_labels=len(TARGET_COLUMNS))
    torch.save(
        {
            "epoch": 0,
            "global_step": 1,
            "pytorch-lightning_version": "2.6.5",
            "state_dict": model.state_dict(),
            "hyper_parameters": {"model_name": BACKBONE_NAME, "num_labels": len(TARGET_COLUMNS)},
        },
        path,
    )
    return path


class TestBenchmarkKaggleBridge:
    def test_cpu_measured_with_provenance_and_gpu_reported_honestly(self, bridge_data_root, tmp_path):
        checkpoint = make_bridge_checkpoint(tmp_path / "fold0-knee-mil-best.ckpt")
        out = tmp_path / "benchmark.json"
        code = benchmark_kaggle_bridge.main([
            "--data-root", str(bridge_data_root), "--checkpoint", str(checkpoint), "--study-uid", STUDY,
            "--iterations", "3", "--warmup", "1", "--batch-sizes", "1", "2", "--output", str(out),
        ])
        assert code == 0
        report = json.loads(out.read_text())

        cpu_b1 = report["cpu"]["batches"]["1"]
        assert cpu_b1["latency"]["median_ms"] > 0 and cpu_b1["throughput_studies_per_s"] > 0
        assert set(report["cpu"]["batches"]) == {"1", "2"}

        workload = report["workload"]
        assert workload["checkpoint_sha256"]
        assert "efficientnet_b0" in workload["model_provenance"]["architecture"]
        assert "attention-MIL" in workload["model_provenance"]["architecture"]
        assert "not native mri_core.RSNAKneeCNN" in workload["model_provenance"]["source"]

        if report["environment"]["amd_gpu_available"]:
            assert report["gpu"]["is_amd_rocm"] is True
            assert report["gpu"]["gpu"]["peak_allocated_bytes"] >= 0
            assert report["comparison"]["gpu_speedup_vs_cpu"] > 0
        else:
            # The core honesty requirement: a CPU-only run must never be reported as a GPU benchmark.
            assert report["gpu"]["status"] == "unavailable" and report["gpu"]["reasons"]
            assert "batches" not in report["gpu"]
            assert "is_amd_rocm" not in report["gpu"]
            assert report["comparison"]["status"] == "not_computed"

    def test_require_gpu_exit_code_reflects_reality(self, bridge_data_root, tmp_path):
        checkpoint = make_bridge_checkpoint(tmp_path / "ckpt.ckpt")
        code = benchmark_kaggle_bridge.main([
            "--data-root", str(bridge_data_root), "--checkpoint", str(checkpoint), "--study-uid", STUDY,
            "--iterations", "1", "--warmup", "0", "--require-gpu", "--output", str(tmp_path / "b.json"),
        ])
        assert code == (0 if rocm_env.check_environment()["amd_gpu_available"] else 2)

    def test_missing_checkpoint_fails_without_writing_output(self, bridge_data_root, tmp_path):
        out = tmp_path / "benchmark.json"
        code = benchmark_kaggle_bridge.main([
            "--data-root", str(bridge_data_root), "--checkpoint", str(tmp_path / "nope.ckpt"), "--study-uid", STUDY,
            "--iterations", "1", "--warmup", "0", "--output", str(out),
        ])
        assert code == 1 and not out.exists()

    def test_benchmark_py_native_path_is_untouched(self):
        """This module must not alter benchmark.py's own public behavior; it only reuses its
        generic, model-agnostic timing/memory helpers."""
        from hackathon.amd import benchmark

        assert hasattr(benchmark, "run_benchmark")
        assert hasattr(benchmark, "bench_device")
        assert benchmark.SCHEMA == "amd-benchmark/v1"

    def test_gpu_path_composition_is_mocked_deterministically(self, monkeypatch, bridge_data_root, tmp_path):
        """Exercises the gpu_ok branch and is_amd_rocm/comparison wiring without touching real
        CUDA/ROCm -- bench_device itself is mocked for the non-cpu call so this runs on any
        machine, including one with no accelerator at all."""
        checkpoint = make_bridge_checkpoint(tmp_path / "ckpt.ckpt")
        fake_env = {
            "amd_gpu_available": True, "accelerator_available": True, "device_count": 1,
            "unavailable_reasons": [], "platform": "mock", "python": "mock", "torch_version": "mock",
            "torch_hip_version": "mock", "torch_cuda_version": None, "is_rocm_build": True,
            "devices": ["mock-amd-gpu"], "smi_tools_on_path": {"rocm-smi": None, "amd-smi": None},
            "generated_utc": "mock",
        }
        monkeypatch.setattr(benchmark_kaggle_bridge, "check_environment", lambda: fake_env)

        real_bench_device = benchmark_kaggle_bridge.bench_device
        calls = []

        def fake_bench_device(checkpoint_path, volume, device_name, **kwargs):
            calls.append(device_name)
            if device_name == "cpu":
                return real_bench_device(checkpoint_path, volume, device_name, **kwargs)
            fake_latency = {"mean_ms": 1.0, "median_ms": 1.0, "p95_ms": 1.0, "min_ms": 1.0, "max_ms": 1.0, "stdev_ms": 0.0}
            fake_result = {
                "device": device_name,
                "batches": {"1": {"iterations": kwargs["iterations"], "warmup": kwargs["warmup"],
                                   "latency": fake_latency, "throughput_studies_per_s": 1000.0}},
                "gpu": {"name": "mock-amd-gpu", "gcn_arch": "mock", "total_memory_bytes": 1,
                        "free_memory_bytes_after": 1, "peak_allocated_bytes": 1, "peak_reserved_bytes": 1,
                        "smi_snapshot": {}},
            }
            return fake_result, [0.5] * len(TARGET_COLUMNS), {"architecture": "mock"}

        monkeypatch.setattr(benchmark_kaggle_bridge, "bench_device", fake_bench_device)

        report = benchmark_kaggle_bridge.run_benchmark(
            bridge_data_root, checkpoint, STUDY, iterations=2, warmup=0, batch_sizes=[1], gpu_device="cuda:0",
        )

        assert calls == ["cpu", "cuda:0"]
        assert report["gpu"]["is_amd_rocm"] is True
        assert "status" not in report["gpu"]
        assert report["comparison"]["gpu_speedup_vs_cpu"] > 0

    def test_non_amd_accelerator_is_labeled_is_amd_rocm_false(self, monkeypatch, bridge_data_root, tmp_path):
        """allow_non_amd must still tell the truth about whether the accelerator was AMD."""
        checkpoint = make_bridge_checkpoint(tmp_path / "ckpt.ckpt")
        fake_env = {
            "amd_gpu_available": False, "accelerator_available": True, "device_count": 1,
            "unavailable_reasons": ["not a ROCm build"], "platform": "mock", "python": "mock",
            "torch_version": "mock", "torch_hip_version": None, "torch_cuda_version": "mock",
            "is_rocm_build": False, "devices": ["mock-non-amd-gpu"],
            "smi_tools_on_path": {"rocm-smi": None, "amd-smi": None}, "generated_utc": "mock",
        }
        monkeypatch.setattr(benchmark_kaggle_bridge, "check_environment", lambda: fake_env)

        real_bench_device = benchmark_kaggle_bridge.bench_device

        def fake_bench_device(checkpoint_path, volume, device_name, **kwargs):
            if device_name == "cpu":
                return real_bench_device(checkpoint_path, volume, device_name, **kwargs)
            fake_latency = {"mean_ms": 1.0, "median_ms": 1.0, "p95_ms": 1.0, "min_ms": 1.0, "max_ms": 1.0, "stdev_ms": 0.0}
            fake_result = {"device": device_name,
                           "batches": {"1": {"iterations": kwargs["iterations"], "warmup": kwargs["warmup"],
                                              "latency": fake_latency, "throughput_studies_per_s": 1000.0}},
                           "gpu": {"name": "mock-non-amd-gpu", "gcn_arch": None, "total_memory_bytes": 1,
                                   "free_memory_bytes_after": 1, "peak_allocated_bytes": 1, "peak_reserved_bytes": 1,
                                   "smi_snapshot": {}}}
            return fake_result, [0.5] * len(TARGET_COLUMNS), {"architecture": "mock"}

        monkeypatch.setattr(benchmark_kaggle_bridge, "bench_device", fake_bench_device)

        report = benchmark_kaggle_bridge.run_benchmark(
            bridge_data_root, checkpoint, STUDY, iterations=2, warmup=0, batch_sizes=[1], gpu_device="cuda:0",
            allow_non_amd=True,
        )
        assert report["gpu"]["is_amd_rocm"] is False
