"""AMD demo scenario generator: determinism, selection quality, serialization hygiene and the consumer contract."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mri_core.rsna_knee_dataset import SERIES_COLUMNS, TARGET_COLUMNS

from .dicom_factory import AXIAL, CORONAL, SAGITTAL, write_series

SPEC = importlib.util.spec_from_file_location("build_amd_demo_scenarios", Path(__file__).resolve().parents[1] / "scripts" / "build_amd_demo_scenarios.py")
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)
ORIENT = {"Axial": AXIAL, "Coronal": CORONAL, "Sagittal": SAGITTAL}
REQUIRED = {"scenario_id", "title", "study_id", "scenario_type", "why_selected", "positive_targets", "negative_target_count", "positive_target_count", "series_count",
            "dicom_count", "planes", "tags", "presentation_tag", "expected_demo_behavior", "data_source"}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("rsna")
    meta, raw = root / "meta", root / "raw"
    meta.mkdir()
    rng = np.random.default_rng(7)
    n = 34
    ids = [f"1.2.826.0.1.3680043.8.498.{i:04d}" for i in range(n)]
    labels = (rng.random((n, 12)) < rng.uniform(0.1, 0.6, 12)).astype(int)
    labels[np.arange(12), np.arange(12)] = 1  # every target has at least one positive study
    series_rows = []
    for i, uid in enumerate(ids):
        planes = ["Axial", "Coronal", "Sagittal"] + [("Axial", "Coronal", "Sagittal")[int(rng.integers(3))] for _ in range(int(rng.integers(0, 4)))]
        for k, plane in enumerate(planes):
            series_uid = f"{uid}.s{k}"
            series_rows.append((uid, series_uid, int(rng.integers(0, 2)), int(rng.integers(0, 2)), plane))
            write_series(raw / "train_series" / uid / series_uid, count=int(rng.integers(2, 6)), orientation=ORIENT[plane], spacing=3.0, series_uid=series_uid,
                         study_uid=uid, rows=8, columns=8)
    train = pd.DataFrame({"StudyInstanceUID": ids, "Report": "synthetic report", **{t: labels[:, j].astype(float) for j, t in enumerate(TARGET_COLUMNS)}})
    unlabeled = pd.DataFrame({"StudyInstanceUID": [f"1.2.826.0.1.3680043.8.498.9{i:03d}" for i in range(3)], "Report": "r", **{t: np.nan for t in TARGET_COLUMNS}})
    pd.concat([train, unlabeled]).to_csv(meta / "train.csv", index=False)
    pd.DataFrame(series_rows, columns=SERIES_COLUMNS).to_csv(meta / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(meta / "test.csv", index=False)
    pd.DataFrame(columns=SERIES_COLUMNS).to_csv(meta / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [], **{t: [] for t in TARGET_COLUMNS}}).to_csv(meta / "sample_submission.csv", index=False)
    return {"meta": meta, "raw": raw, "ids": ids, "labels": labels, "root": root}


def run(world, out_name):
    out = world["root"] / out_name
    assert gen.main(["--data-dir", str(world["meta"]), "--dicom-root", str(world["raw"]), "--out-dir", str(out)]) == 0
    return out


def load(out):
    return json.loads((out / "amd_demo_scenarios.json").read_text())


def test_generator_is_deterministic_and_byte_identical(world):
    a, b = run(world, "run_a"), run(world, "run_b")
    for name in ("amd_demo_scenarios.json", "amd_demo_scenarios_summary.json", "README.md"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    assert gen.main(["--data-dir", str(world["meta"]), "--dicom-root", str(world["raw"]), "--out-dir", str(a), "--check"]) == 0


def test_exactly_22_unique_scenarios_from_locally_available_labelled_studies(world):
    doc = load(run(world, "run_c"))
    scenarios = doc["scenarios"]
    assert doc["scenario_count"] == len(scenarios) == 22
    assert len({s["scenario_id"] for s in scenarios}) == 22 and len({s["study_id"] for s in scenarios}) == 22 and len({s["title"] for s in scenarios}) == 22
    assert [s["scenario_id"] for s in scenarios] == [f"AMD-{i:03d}" for i in range(1, 23)]
    assert set(s["study_id"] for s in scenarios) <= set(world["ids"])  # labelled and present locally (unlabelled studies were never candidates)
    assert not any(s["study_id"].startswith("1.2.826.0.1.3680043.8.498.9") for s in scenarios)


def test_all_twelve_targets_are_in_the_schema_and_covered(world):
    doc = load(run(world, "run_d"))
    assert doc["targets"] == list(TARGET_COLUMNS) and len(doc["targets"]) == 12
    for target in TARGET_COLUMNS:
        assert any(target in s["positive_targets"] for s in doc["scenarios"]), target


def test_positive_targets_and_counts_agree_with_the_source_labels(world):
    doc = load(run(world, "run_e"))
    index = {u: i for i, u in enumerate(world["ids"])}
    for s in doc["scenarios"]:
        row = world["labels"][index[s["study_id"]]]
        assert s["positive_targets"] == [t for t, v in zip(TARGET_COLUMNS, row) if v] and s["label_vector"] == row.tolist()
        assert s["positive_target_count"] == int(row.sum()) and s["negative_target_count"] == 12 - int(row.sum())


def test_structure_fields_match_the_metadata_and_dicom_files(world):
    doc = load(run(world, "run_f"))
    series = pd.read_csv(world["meta"] / "train_series.csv")
    for s in doc["scenarios"]:
        rows = series[series["StudyInstanceUID"] == s["study_id"]]
        files = sum(1 for p in (world["raw"] / "train_series" / s["study_id"]).rglob("*") if p.is_file())
        assert s["series_count"] == len(rows) and s["dicom_count"] == files and set(s["planes"]) == set(rows["Anatomical_Plane"])


def test_selection_is_diverse_with_no_duplicate_label_vectors(world):
    doc = load(run(world, "run_g"))
    vectors = {tuple(s["label_vector"]) for s in doc["scenarios"]}
    assert len(vectors) == 22
    counts = [s["positive_target_count"] for s in doc["scenarios"]]
    assert counts == sorted(counts) and max(counts) - min(counts) >= 4  # ordered simple to complex and spanning a range of burdens
    assert {s["scenario_type"] for s in doc["scenarios"]} >= {"target_anchor", "high_multilabel", "label_diversity", "structural_diversity", "low_label_count"}
    summary = json.loads((run(world, "run_g") / "amd_demo_scenarios_summary.json").read_text())
    assert summary["targets_with_a_positive_scenario"] == 12 and summary["scenario_count"] == 22 and summary["model_tags_present"] is False


def test_no_absolute_paths_blobs_or_model_tags_are_serialized(world):
    out = run(world, "run_h")
    for name in ("amd_demo_scenarios.json", "amd_demo_scenarios_summary.json", "README.md"):
        text = (out / name).read_text()
        assert not re.search(r"[A-Za-z]:[\\/]", text) and "/Users/" not in text and "/home/" not in text and "\\\\" not in text
        assert "DICM" not in text and "base64" not in text.lower() and "\x00" not in text and "synthetic report" not in text
    assert (out / "amd_demo_scenarios.json").stat().st_size < 200_000
    assert "model_signals" not in (out / "amd_demo_scenarios.json").read_text()


def test_json_has_the_fields_the_streamlit_consumer_needs_and_loads_without_ml_dependencies(world):
    out = run(world, "run_i")
    doc = load(out)
    assert doc["schema_version"] == "1.0" and doc["scenario_set"] == "amd_mri_core_demo_v1" and "not clinical" in doc["disclaimer"]
    for s in doc["scenarios"]:
        assert REQUIRED <= set(s) and s["data_source"] == "RSNA Knee Abnormality Detection"
    code = ("import json,sys; d=json.load(open(sys.argv[1])); labels=[f\"{x['scenario_id']} - {x['title']}\" for x in d['scenarios']]; "
            "bad=[m for m in ('torch','numpy','pandas','pydicom','sklearn') if m in sys.modules]; print(len(labels), bad)")
    result = subprocess.run([sys.executable, "-I", "-c", code, str(out / "amd_demo_scenarios.json")], capture_output=True, text=True)
    assert result.stdout.strip() == "22 []", result.stderr


def test_too_few_labelled_studies_is_refused(world):
    candidates = gen.discover_candidates(world["meta"], world["raw"])
    with pytest.raises(ValueError, match="only 5 labelled local studies"):
        gen.select(candidates[:5])


def test_committed_artifact_matches_the_local_source_when_the_cache_exists():
    data = Path("data/rsna-knee")
    committed = Path("benchmarks/amd/amd_demo_scenarios.json")
    if not committed.is_file():
        pytest.skip("scenario file not generated in this checkout")
    doc = json.loads(committed.read_text())
    assert doc["scenario_count"] == 22 and len(doc["scenarios"]) == 22
    if not (data / "train.csv").is_file() or not (data / "raw" / "train_series").is_dir():
        pytest.skip("local RSNA cache not present: JSON structure checked, labels not re-verified")
    assert gen.main(["--check"]) == 0
