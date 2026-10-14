"""Fixture-only tests for scripts/kaggle_rsna_runner.py (no Kaggle, no real RSNA data)."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_rsna_runner as kr  # noqa: E402

from mri_core.rsna_knee_dataset import SERIES_COLUMNS, TARGET_COLUMNS, load_rsna_metadata  # noqa: E402

from .dicom_factory import AXIAL, CORONAL, SAGITTAL, write_series  # noqa: E402

pytestmark = pytest.mark.rsna

STUDIES = [f"1.2.826.{i}" for i in range(1, 7)]
CONFIG = {**kr.DEFAULT_CONFIG, "image_size": 8, "slices_per_series": 2, "dropout": 0.0, "batch_size": 2,
          "num_workers": 0, "epochs": 2, "device": "cpu", "validation_fraction": 0.34}


def build_input(root: Path, *, name="some-renamed-comp", dicom_subdir=None, skip_study=None):
    meta = root / name
    raw = meta if dicom_subdir is None else meta / dicom_subdir
    meta.mkdir(parents=True)
    rows = [{"StudyInstanceUID": uid, "Report": "r", **{t: (i + j) % 2 for j, t in enumerate(TARGET_COLUMNS)}}
            for i, uid in enumerate(STUDIES)]
    pd.DataFrame(rows).to_csv(meta / "train.csv", index=False)
    series_rows = []
    for uid in STUDIES:
        for plane, orientation in (("Axial", AXIAL), ("Coronal", CORONAL), ("Sagittal", SAGITTAL)):
            series_uid = f"{uid}.series.{plane}"
            series_rows.append((uid, series_uid, 1, 0, plane))
            if uid != skip_study:
                write_series(raw / "train_series" / uid / series_uid, count=2, orientation=orientation,
                            series_uid=series_uid, study_uid=uid, rows=8, columns=8)
    pd.DataFrame(series_rows, columns=SERIES_COLUMNS).to_csv(meta / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(meta / "test.csv", index=False)
    pd.DataFrame(columns=SERIES_COLUMNS).to_csv(meta / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [], **{t: [] for t in TARGET_COLUMNS}}).to_csv(meta / "sample_submission.csv", index=False)
    return meta, raw


def snapshot(root: Path):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


class TestDiscovery:
    def test_finds_data_under_any_folder_name(self, tmp_path):
        meta, raw = build_input(tmp_path / "input", name="totally-different-name")
        assert kr.discover_data(tmp_path / "input") == (meta, raw)

    def test_dicom_root_in_subfolder(self, tmp_path):
        meta, raw = build_input(tmp_path / "input", dicom_subdir="images")
        assert kr.discover_data(tmp_path / "input") == (meta, raw)

    def test_ambiguous_or_missing_input_is_an_error(self, tmp_path):
        build_input(tmp_path / "input", name="a")
        build_input(tmp_path / "input", name="b")
        with pytest.raises(kr.DiscoveryError, match="exactly one"):
            kr.discover_data(tmp_path / "input")
        with pytest.raises(kr.DiscoveryError):
            kr.discover_data(tmp_path / "nope")


class TestInspectIsReadOnly:
    def test_reports_all_12_targets_and_changes_nothing(self, tmp_path, capsys):
        meta, raw = build_input(tmp_path / "input", skip_study=STUDIES[0])
        before = snapshot(tmp_path)
        assert kr.main(["inspect", "--data-dir", str(meta), "--dicom-root", str(raw)]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["target_count"] == 12 and len(report["targets"]) == 12
        assert report["complete_label_studies"] == 6
        assert report["labeled_studies_available_on_disk"] == 5  # the study with no DICOMs is excluded
        assert report["study_listing"]["shown"] == 5
        assert report["sampled_study_size"]["mean_files"] > 0
        assert snapshot(tmp_path) == before


class TestSelection:
    def test_subset_is_seeded_and_persisted(self, tmp_path):
        avail = sorted(STUDIES)
        assert kr.select_subset(avail, 3, 0) == kr.select_subset(avail, 3, 0)
        path = tmp_path / "sel.json"
        first = kr.load_or_create_selection(path, avail, 3, 0)
        assert kr.load_or_create_selection(path, avail, 5, 99) == first  # saved list wins
        with pytest.raises(RuntimeError):
            kr.load_or_create_selection(path, avail[:1], 3, 0)


class TestTrainResumeExport:
    def test_segmented_training_resumes_and_checkpoint_is_complete(self, tmp_path):
        meta, raw = build_input(tmp_path / "input")
        metadata = load_rsna_metadata(meta, raw)
        out = tmp_path / "work"
        first = kr.run_training(metadata, CONFIG, out, study_limit=None, resume_from=None, epochs_this_session=1)
        assert first["final_epoch"] == 1 and (out / "selection.json").is_file()
        second = kr.run_training(metadata, CONFIG, out, study_limit=None, resume_from=None, epochs_this_session=1)
        assert second["resumed_from_epoch"] == 1 and second["final_epoch"] == 2
        ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        for key in ("model", "optimizer", "scheduler", "scaler", "config", "epoch", "global_step", "metrics",
                    "torch_rng", "python_rng", "numpy_rng", "loader_rng", "train_study_ids", "valid_study_ids"):
            assert key in ckpt, key
        assert ckpt["epoch"] == 2 and not set(ckpt["train_study_ids"]) & set(ckpt["valid_study_ids"])
        assert len(ckpt["target_columns"]) == 12

    def test_resume_from_attached_output_and_no_overwrite(self, tmp_path):
        meta, raw = build_input(tmp_path / "input")
        metadata = load_rsna_metadata(meta, raw)
        prev = tmp_path / "prev_output"
        kr.run_training(metadata, CONFIG, prev, study_limit=None, resume_from=None, epochs_this_session=1)
        new = tmp_path / "new_session"
        result = kr.run_training(metadata, CONFIG, new, study_limit=None, resume_from=prev,
                                 epochs_this_session=0)
        assert result["resumed_from_epoch"] == 1 and result["final_epoch"] == 2
        with pytest.raises(FileExistsError):
            kr._stage_resume(prev, new)

    def test_benchmark_reports_timing(self, tmp_path):
        meta, raw = build_input(tmp_path / "input")
        summary = kr.run_benchmark(load_rsna_metadata(meta, raw), CONFIG, tmp_path / "bench", studies=4, epochs=1)
        assert summary["seconds_per_epoch"] >= 0 and (tmp_path / "bench" / "benchmark.json").is_file()

    def test_export_has_checksums_and_no_competition_data(self, tmp_path):
        meta, raw = build_input(tmp_path / "input")
        out = tmp_path / "work"
        kr.run_training(load_rsna_metadata(meta, raw), CONFIG, out, study_limit=None, resume_from=None,
                        epochs_this_session=1)
        (out / "stray.dcm").write_bytes(b"x")
        result = kr.export_artifacts(out, tmp_path / "export")
        manifest = result["files"]
        assert manifest["last.pt"]["sha256"] == hashlib.sha256((out / "last.pt").read_bytes()).hexdigest()
        with zipfile.ZipFile(result["archive"]) as z:
            names = z.namelist()
        assert "last.pt" in names and "MANIFEST.json" in names
        assert not any(n.endswith((".dcm", ".csv")) for n in names)

    def test_export_without_checkpoint_fails(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            kr.export_artifacts(tmp_path / "empty", tmp_path / "export")
