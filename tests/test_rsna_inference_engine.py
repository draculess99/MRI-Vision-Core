"""Tests for the reusable RSNA inference engine, using synthetic checkpoints and a tiny
synthetic DICOM study. No real RSNA data or Kaggle access required here (see
test_rsna_inference_engine_real_data.py for the real-study smoke tests).
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from mri_core.rsna_inference_engine import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_UNABLE_TO_EVALUATE,
    InferenceEngineError,
    ModelProvenance,
    StudyInferenceResult,
    run_ensemble_model_inference,
    run_single_model_inference,
    sha256_of_file,
    verify_ensemble_compatibility,
)
from mri_core.rsna_knee_dataset import SERIES_COLUMNS, TARGET_COLUMNS, load_rsna_metadata
from mri_core.rsna_knee_model import RSNAKneeCNN

from .dicom_factory import AXIAL, CORONAL, SAGITTAL, write_series

STUDY = "1.2.826.1"
AXIAL_SERIES, CORONAL_SERIES, SAGITTAL_SERIES = "1.2.826.2.1", "1.2.826.2.2", "1.2.826.2.3"


@pytest.fixture
def rsna(tmp_path):
    meta, raw = tmp_path / "meta", tmp_path / "raw"
    meta.mkdir()
    pd.DataFrame([{"StudyInstanceUID": STUDY, "Report": "r", **{t: i % 2 for i, t in enumerate(TARGET_COLUMNS)}}]) \
        .to_csv(meta / "train.csv", index=False)
    pd.DataFrame(
        [(STUDY, AXIAL_SERIES, 1, 0, "Axial"), (STUDY, CORONAL_SERIES, 0, 1, "Coronal"),
         (STUDY, SAGITTAL_SERIES, 0, 1, "Sagittal")],
        columns=SERIES_COLUMNS,
    ).to_csv(meta / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(meta / "test.csv", index=False)
    pd.DataFrame(columns=SERIES_COLUMNS).to_csv(meta / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [], **{t: [] for t in TARGET_COLUMNS}}).to_csv(meta / "sample_submission.csv", index=False)
    write_series(raw / "train_series" / STUDY / AXIAL_SERIES, count=2, orientation=AXIAL, series_uid=AXIAL_SERIES, study_uid=STUDY)
    write_series(raw / "train_series" / STUDY / CORONAL_SERIES, count=2, orientation=CORONAL, series_uid=CORONAL_SERIES, study_uid=STUDY)
    write_series(raw / "train_series" / STUDY / SAGITTAL_SERIES, count=2, orientation=SAGITTAL, series_uid=SAGITTAL_SERIES, study_uid=STUDY)
    metadata = load_rsna_metadata(meta, dicom_root=raw)
    return SimpleNamespace(meta=meta, raw=raw, metadata=metadata)


def make_checkpoint(path, epoch=15, fold=0, target_columns=None):
    model = RSNAKneeCNN(num_planes=3, dropout=0.0)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"image_size": 16, "slices_per_series": 2, "planes": ["Axial", "Coronal", "Sagittal"], "dropout": 0.0, "cv_fold": fold},
            "target_columns": list(target_columns) if target_columns is not None else list(TARGET_COLUMNS),
            "epoch": epoch,
        },
        path,
    )
    return path


class TestSha256:
    def test_matches_known_hash(self, tmp_path):
        import hashlib
        path = tmp_path / "f.bin"
        path.write_bytes(b"hello world")
        assert sha256_of_file(path) == hashlib.sha256(b"hello world").hexdigest()


class TestSingleModelInference:
    def test_ok_result_has_twelve_ordered_probabilities(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        result = run_single_model_inference(rsna.metadata, checkpoint, STUDY)
        assert result.status == STATUS_OK
        assert list(result.probabilities.keys()) == list(TARGET_COLUMNS)
        assert len(result.probabilities) == 12
        assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in result.probabilities.values())
        assert not result.is_ensemble
        assert not result.review_required
        assert result.disclaimer

    def test_provenance_includes_sha256_and_epoch(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt", epoch=7, fold=3)
        result = run_single_model_inference(rsna.metadata, checkpoint, STUDY)
        (provenance,) = result.model_provenance
        assert provenance.checkpoint_sha256 == sha256_of_file(checkpoint)
        assert provenance.epoch == 7
        assert provenance.fold == 3
        assert provenance.target_columns == tuple(TARGET_COLUMNS)

    def test_latency_is_measured_and_positive(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        result = run_single_model_inference(rsna.metadata, checkpoint, STUDY)
        assert result.latency_seconds > 0.0

    def test_repeated_inference_is_deterministic(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        first = run_single_model_inference(rsna.metadata, checkpoint, STUDY)
        second = run_single_model_inference(rsna.metadata, checkpoint, STUDY)
        assert first.probabilities == second.probabilities

    def test_unknown_study_returns_unable_to_evaluate_not_a_raised_exception(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        result = run_single_model_inference(rsna.metadata, checkpoint, "1.2.826.999.unknown")
        assert result.status == STATUS_UNABLE_TO_EVALUATE
        assert result.probabilities is None
        assert result.review_required is True
        assert result.warnings

    def test_missing_checkpoint_raises_engine_error_not_unable_to_evaluate(self, rsna, tmp_path):
        with pytest.raises(InferenceEngineError):
            run_single_model_inference(rsna.metadata, tmp_path / "does_not_exist.pt", STUDY)

    def test_target_mismatch_raises_engine_error(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt", target_columns=["abnormal", "acl", "meniscus"])
        with pytest.raises(InferenceEngineError):
            run_single_model_inference(rsna.metadata, checkpoint, STUDY)


class TestEnsembleModelInference:
    def test_ok_result_averages_five_models(self, rsna, tmp_path):
        paths = [make_checkpoint(tmp_path / f"fold{i}.pt", fold=i) for i in range(5)]
        result = run_ensemble_model_inference(rsna.metadata, paths, STUDY)
        assert result.status == STATUS_OK
        assert result.is_ensemble
        assert len(result.model_provenance) == 5
        assert list(result.probabilities.keys()) == list(TARGET_COLUMNS)

    def test_missing_fold_raises_engine_error_no_partial_ensemble(self, rsna, tmp_path):
        paths = [make_checkpoint(tmp_path / f"fold{i}.pt", fold=i) for i in range(4)]
        paths.append(tmp_path / "missing_fold4.pt")
        with pytest.raises(InferenceEngineError):
            run_ensemble_model_inference(rsna.metadata, paths, STUDY)

    def test_unknown_study_returns_unable_to_evaluate(self, rsna, tmp_path):
        paths = [make_checkpoint(tmp_path / f"fold{i}.pt", fold=i) for i in range(5)]
        result = run_ensemble_model_inference(rsna.metadata, paths, "1.2.826.999.unknown")
        assert result.status == STATUS_UNABLE_TO_EVALUATE
        assert result.probabilities is None


class TestEnsembleCompatibilityVerification:
    def test_five_matching_checkpoints_are_compatible(self, tmp_path):
        paths = [make_checkpoint(tmp_path / f"fold{i}.pt", fold=i) for i in range(5)]
        report = verify_ensemble_compatibility(paths)
        assert report.compatible
        assert report.target_columns_match
        assert report.configs_match
        assert not report.mismatches

    def test_target_mismatch_is_detected_without_running_inference(self, tmp_path):
        paths = [make_checkpoint(tmp_path / f"fold{i}.pt", fold=i) for i in range(4)]
        paths.append(make_checkpoint(tmp_path / "fold4.pt", fold=4, target_columns=["abnormal", "acl", "meniscus"]))
        report = verify_ensemble_compatibility(paths)
        assert not report.compatible
        assert report.mismatches

    def test_missing_checkpoint_is_reported_not_raised(self, tmp_path):
        paths = [make_checkpoint(tmp_path / f"fold{i}.pt", fold=i) for i in range(4)]
        paths.append(tmp_path / "missing.pt")
        report = verify_ensemble_compatibility(paths)
        assert not report.compatible
        assert any("not found" in m for m in report.mismatches)


class TestResultSerialization:
    def test_to_dict_round_trips_key_fields(self, rsna, tmp_path):
        checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
        result = run_single_model_inference(rsna.metadata, checkpoint, STUDY)
        payload = result.to_dict()
        assert payload["study_uid"] == STUDY
        assert payload["status"] == STATUS_OK
        assert len(payload["probabilities"]) == 12
        assert payload["model_provenance"][0]["checkpoint_sha256"] == sha256_of_file(checkpoint)
        assert "disclaimer" in payload and payload["disclaimer"]
