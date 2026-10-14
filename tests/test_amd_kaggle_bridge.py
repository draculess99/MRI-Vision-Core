"""Smoke test for hackathon/amd/kaggle_mil_bridge.py and run_real_checkpoint_smoke.py.

Unlike tests/test_amd_adapter.py, there is no synthetic-fixture path here: the bridged
architecture only means anything against a real Kaggle `foldN-knee-mil-best.ckpt`. This test
runs only when AMD_KAGGLE_CHECKPOINT and AMD_KAGGLE_RSNA_ROOT point at real local paths
(an external Kaggle checkpoint, a real downloaded RSNA knee data_root); it skips otherwise.
"""

import json
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("pydicom")

from hackathon.amd import run_real_checkpoint_smoke
from hackathon.amd.kaggle_mil_bridge import TARGET_COLUMNS

requires_real_assets = pytest.mark.skipif(
    not (os.environ.get("AMD_KAGGLE_CHECKPOINT") and os.environ.get("AMD_KAGGLE_RSNA_ROOT")),
    reason="set AMD_KAGGLE_CHECKPOINT and AMD_KAGGLE_RSNA_ROOT to run against the real bridged checkpoint and real MRI data",
)


@requires_real_assets
def test_real_checkpoint_real_study_produces_structured_result(tmp_path):
    out = tmp_path / "prediction.json"
    code = run_real_checkpoint_smoke.main([
        "--data-root", os.environ["AMD_KAGGLE_RSNA_ROOT"],
        "--checkpoint", os.environ["AMD_KAGGLE_CHECKPOINT"],
        "--out", str(out),
    ])
    assert code == 0
    report = json.loads(out.read_text())

    prediction = report["prediction"]
    assert list(prediction["probabilities"].keys()) == list(TARGET_COLUMNS)
    assert all(0.0 <= v <= 1.0 for v in prediction["probabilities"].values())
    assert prediction["n_slices_sampled"] == 24
    assert 0.0 < prediction["latency_seconds"] < 300.0

    provenance = report["provenance"]
    assert provenance["checkpoint_path"] == os.environ["AMD_KAGGLE_CHECKPOINT"]
    assert len(provenance["checkpoint_sha256"]) == 64
    assert "not native mri_core.RSNAKneeCNN" in provenance["source"]

    assert "not a diagnostic device" in report["non_diagnostic_notice"]


@requires_real_assets
def test_result_is_reproducible_across_runs(tmp_path):
    first_out = tmp_path / "first.json"
    second_out = tmp_path / "second.json"
    for out in (first_out, second_out):
        code = run_real_checkpoint_smoke.main([
            "--data-root", os.environ["AMD_KAGGLE_RSNA_ROOT"],
            "--checkpoint", os.environ["AMD_KAGGLE_CHECKPOINT"],
            "--out", str(out),
        ])
        assert code == 0

    first = json.loads(first_out.read_text())["prediction"]["probabilities"]
    second = json.loads(second_out.read_text())["prediction"]["probabilities"]
    assert first == second


@requires_real_assets
def test_missing_checkpoint_fails_without_writing_output(tmp_path):
    out = tmp_path / "prediction.json"
    code = run_real_checkpoint_smoke.main([
        "--data-root", os.environ["AMD_KAGGLE_RSNA_ROOT"],
        "--checkpoint", str(tmp_path / "does-not-exist.ckpt"),
        "--out", str(out),
    ])
    assert code == 1
    assert not out.exists()
