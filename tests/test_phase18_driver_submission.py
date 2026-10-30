"""Executes the submission section of the Phase 18 kernel driver with fake predictions: ensemble averaging, baseline-only Synovitis, schema, fallback, abort."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mri_core.rsna_knee_dataset import SUBMISSION_COLUMNS, TARGET_COLUMNS
from mri_core.weak_label_submit import SubmissionError, combine, validate_submission

SOURCE = Path("scripts/kaggle_phase18_driver.py").read_text(encoding="utf-8")
SECTION = SOURCE[SOURCE.index("# ---- 5. hidden-test inference and submission"):]
TARGETS11 = [t for t in TARGET_COLUMNS if t != "Synovitis"]


def run_section(tmp_path, order, fake, max_failed="0.05"):
    class Stop(Exception):
        pass

    def stop(reason):
        raise Stop(reason)

    summary = {"wall_seconds": 0}
    ns = {"os": SimpleNamespace(environ={"P18_MAX_FAILED": max_failed}), "np": np, "pd": pd, "sys": SimpleNamespace(exit=lambda c: None), "json": json,
          "metadata": SimpleNamespace(sample_submission=pd.DataFrame({"StudyInstanceUID": order, **{t: 0.5 for t in TARGET_COLUMNS}})),
          "baseline_ckpts": ["b0", "b1"], "models": {"s701": "m0", "s702": "m1"}, "device": "cpu", "TARGETS11": TARGETS11, "TARGET_COLUMNS": TARGET_COLUMNS,
          "predict_test": fake, "combine": combine, "validate_submission": validate_submission, "SubmissionError": SubmissionError,
          "SUMMARY": summary, "WORKING": tmp_path, "stop": stop, "save": lambda: None, "sha": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    exec(SECTION, ns)
    return summary, Stop


def fake_factory(order, values, failed=()):
    def fake(path, metadata, device):
        v = values[path]
        return {u: np.full(12, v) for u in order}, set(failed)
    return fake


def test_candidate_columns_are_the_seed_mean_and_synovitis_is_the_baseline_mean(tmp_path):
    order = ["a", "b", "c"]
    fake = fake_factory(order, {"b0": 0.2, "b1": 0.4, "m0": 0.6, "m1": 0.8})
    summary, _ = run_section(tmp_path, order, fake)
    frame = pd.read_csv(tmp_path / "submission.csv")
    assert tuple(frame.columns) == SUBMISSION_COLUMNS and list(frame["StudyInstanceUID"]) == order
    assert np.allclose(frame["Synovitis"], 0.3) and all(np.allclose(frame[t], 0.7) for t in TARGETS11)
    assert summary["submission"]["rows"] == 3 and summary["submission"]["targets_from_baseline"] == ["Synovitis"] and summary["stage"] == "done"


def test_unreadable_studies_get_the_mean_fill_within_the_limit_and_abort_beyond_it(tmp_path):
    order = [f"s{i}" for i in range(40)]
    values = {"b0": 0.2, "b1": 0.4, "m0": 0.6, "m1": 0.8}
    summary, _ = run_section(tmp_path, order, fake_factory(order, values, failed={"s3"}))
    frame = pd.read_csv(tmp_path / "submission.csv")
    assert summary["submission"]["unreadable_test_studies"] == 1 and np.isfinite(frame[list(TARGET_COLUMNS)].to_numpy()).all()
    out2 = tmp_path / "second"
    out2.mkdir()
    with pytest.raises(Exception, match="unreadable"):
        run_section(out2, order, fake_factory(order, values, failed={f"s{i}" for i in range(10)}))
    assert not (out2 / "submission.csv").exists()


def test_missing_prediction_for_a_template_study_stops_without_writing_a_submission(tmp_path):
    order = ["a", "b"]

    def fake(path, metadata, device):
        return {"a": np.full(12, 0.5)}, set()
    with pytest.raises(Exception) as e:
        run_section(tmp_path, order, fake)
    assert "no test metadata" in str(e.value) and not (tmp_path / "submission.csv").exists()
