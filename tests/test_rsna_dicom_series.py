import dataclasses
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

from mri_core import DicomSeriesError, DicomSeriesWarning, MRIVolume
from mri_core.rsna_knee_dataset import (SERIES_COLUMNS, SUBMISSION_COLUMNS, TARGET_COLUMNS, load_rsna_metadata,
                                        load_series_volume, load_study_volumes, metadata_report)

from .dicom_factory import AXIAL, SAGITTAL, STUDY_UID, write_series

STUDY = STUDY_UID
AXIAL_SERIES, SAGITTAL_SERIES = "1.2.826.2.1", "1.2.826.2.2"
TEST_STUDY, TEST_SERIES = "1.2.826.9", "1.2.826.9.1"
CLI = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_rsna_knee.py"


@pytest.fixture
def rsna(tmp_path):
    """Metadata CSVs in meta/, DICOMs in a separate raw/ root, mirroring data/rsna-knee and data/rsna-knee/raw."""
    meta, raw = tmp_path / "meta", tmp_path / "raw"
    meta.mkdir()
    pd.DataFrame([{"StudyInstanceUID": STUDY, "Report": "r", **{t: i % 2 for i, t in enumerate(TARGET_COLUMNS)}}]) \
        .to_csv(meta / "train.csv", index=False)
    pd.DataFrame([(STUDY, AXIAL_SERIES, 1, 0, "Axial"), (STUDY, SAGITTAL_SERIES, 0, 1, "Sagittal")],
                 columns=SERIES_COLUMNS).to_csv(meta / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [TEST_STUDY]}).to_csv(meta / "test.csv", index=False)
    pd.DataFrame([(TEST_STUDY, TEST_SERIES, 1, 1, "Axial")], columns=SERIES_COLUMNS).to_csv(meta / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [TEST_STUDY], **{t: 0.5 for t in TARGET_COLUMNS}}) \
        .to_csv(meta / "sample_submission.csv", index=False)
    write_series(raw / "train_series" / STUDY / AXIAL_SERIES, count=4, orientation=AXIAL, series_uid=AXIAL_SERIES)
    write_series(raw / "train_series" / STUDY / SAGITTAL_SERIES, count=3, orientation=SAGITTAL, series_uid=SAGITTAL_SERIES,
                 spacing=4.0, rows=8, columns=10)
    write_series(raw / "test_series" / TEST_STUDY / TEST_SERIES, count=3, orientation=AXIAL, series_uid=TEST_SERIES,
                 study_uid=TEST_STUDY)
    return SimpleNamespace(meta=meta, raw=raw)


# ----------------------------------------------------------------------------- DICOM root configuration

def test_dicom_root_defaults_to_data_dir(rsna):
    metadata = load_rsna_metadata(rsna.meta)
    assert metadata.dicom_root is None
    assert metadata.dicom_split_dir("train") == rsna.meta / "train_series"
    assert metadata.dicom_series_dir(STUDY, AXIAL_SERIES) == rsna.meta / "train_series" / STUDY / AXIAL_SERIES
    assert metadata_report(metadata)["dicom_directories"]["train"] is False
    with pytest.raises(ValueError, match="train or test"):
        metadata.dicom_split_dir("validation")


@pytest.mark.parametrize("as_type", [Path, str])
def test_configured_dicom_root_needs_no_metadata_replacement(rsna, as_type):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=as_type(rsna.raw))
    assert metadata.data_dir == rsna.meta and metadata.dicom_root == rsna.raw
    assert metadata.dicom_split_dir("train") == rsna.raw / "train_series"
    assert metadata.dicom_split_dir("test") == rsna.raw / "test_series"
    assert metadata.dicom_series_dir(STUDY, SAGITTAL_SERIES) == rsna.raw / "train_series" / STUDY / SAGITTAL_SERIES
    directories = metadata_report(metadata)["dicom_directories"]
    assert directories["train"] is True and directories["test"] is True
    assert directories["train_path"] == str(rsna.raw / "train_series")


def test_cli_inspect_accepts_dicom_root(rsna, capsys):
    spec = importlib.util.spec_from_file_location("kaggle_rsna_knee", CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["inspect", "--data-dir", str(rsna.meta)]) == 0
    assert json.loads(capsys.readouterr().out)["dicom_directories"]["train"] is False
    assert cli.main(["inspect", "--data-dir", str(rsna.meta), "--dicom-root", str(rsna.raw)]) == 0
    directories = json.loads(capsys.readouterr().out)["dicom_directories"]
    assert directories["train"] is True and directories["train_path"] == str(rsna.raw / "train_series")


def test_existing_dicom_dataset_honours_dicom_root(rsna):
    pytest.importorskip("torch")
    from mri_core.rsna_knee_train import RSNAKneeDicomDataset
    config = {"image_size": 16, "slices_per_series": 2, "planes": ["Axial", "Sagittal"]}
    default = load_rsna_metadata(rsna.meta)
    with pytest.raises(FileNotFoundError, match="train_series"):
        RSNAKneeDicomDataset(default, default.train, config)
    configured = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    item = RSNAKneeDicomDataset(configured, configured.train, config)[0]
    assert tuple(item["images"].shape) == (2, 2, 1, 16, 16)
    assert bool(item["mask"].all())


def test_submission_uses_dicom_root(rsna, tmp_path):
    torch = pytest.importorskip("torch")
    from mri_core.rsna_knee_model import RSNAKneeCNN
    from mri_core.rsna_knee_submit import write_submission
    config = {"dropout": 0.0, "planes": ["Axial"], "image_size": 16, "slices_per_series": 2}
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": RSNAKneeCNN(1, 0.0).state_dict(), "config": config, "target_columns": list(TARGET_COLUMNS)}, checkpoint)
    with pytest.raises(FileNotFoundError, match="test_series"):
        write_submission(rsna.meta, checkpoint, tmp_path / "out.csv")
    result = write_submission(rsna.meta, checkpoint, tmp_path / "out.csv", dicom_root=rsna.raw)
    assert tuple(result.columns) == SUBMISSION_COLUMNS and list(result.StudyInstanceUID.astype(str)) == [TEST_STUDY]


# ----------------------------------------------------------------------------- series and study adapter

def test_load_series_volume_combines_dicom_and_csv_metadata(rsna):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    volume = load_series_volume(metadata, STUDY, SAGITTAL_SERIES)
    assert isinstance(volume, MRIVolume) and volume.shape == (3, 8, 10)
    assert volume.metadata["Series Instance UID"] == SAGITTAL_SERIES
    assert volume.metadata["Anatomical Plane"] == "Sagittal"
    assert volume.metadata["Acquisition Plane (DICOM orientation)"] == "Sagittal"
    assert volume.metadata["Fluid Sensitive"] is False and volume.metadata["Fat Suppression"] is True
    assert [float(volume.get_slice(i)[0, 0]) for i in range(3)] == [10.0, 20.0, 30.0]


def test_load_series_volume_supports_test_split(rsna):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    assert load_series_volume(metadata, TEST_STUDY, TEST_SERIES, split="test").shape == (3, 8, 8)


def test_load_study_volumes_keeps_metadata_order(rsna):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    volumes = load_study_volumes(metadata, STUDY)
    assert list(volumes) == [AXIAL_SERIES, SAGITTAL_SERIES]
    assert [v.shape for v in volumes.values()] == [(4, 8, 8), (3, 8, 10)]
    with pytest.raises(KeyError, match="No train series"):
        load_study_volumes(metadata, "1.2.826.404")


def test_missing_series_directory_explains_dicom_root(rsna):
    metadata = load_rsna_metadata(rsna.meta)
    with pytest.raises(FileNotFoundError) as excinfo:
        load_series_volume(metadata, STUDY, AXIAL_SERIES)
    assert "--dicom-root" in str(excinfo.value) and str(rsna.meta / "train_series") in str(excinfo.value)


def test_unknown_series_is_rejected(rsna):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    with pytest.raises(KeyError, match="found 0"):
        load_series_volume(metadata, STUDY, "1.2.826.404")


def test_plane_disagreement_between_csv_and_dicom_warns(rsna):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    frame = metadata.train_series.copy()
    frame.loc[frame.SeriesInstanceUID == AXIAL_SERIES, "Anatomical_Plane"] = "Coronal"
    with pytest.warns(DicomSeriesWarning, match="Coronal.*Axial"):
        volume = load_series_volume(dataclasses.replace(metadata, train_series=frame), STUDY, AXIAL_SERIES)
    assert volume.metadata["Anatomical Plane"] == "Coronal"


def test_series_directory_holding_another_series_is_rejected(rsna):
    metadata = load_rsna_metadata(rsna.meta, dicom_root=rsna.raw)
    alias = "1.2.826.2.9"
    shutil.copytree(rsna.raw / "train_series" / STUDY / AXIAL_SERIES, rsna.raw / "train_series" / STUDY / alias)
    frame = metadata.train_series.copy()
    frame.loc[frame.SeriesInstanceUID == AXIAL_SERIES, "SeriesInstanceUID"] = alias
    with pytest.raises(DicomSeriesError, match="does not match expected"):
        load_series_volume(dataclasses.replace(metadata, train_series=frame), STUDY, alias)
