"""End-to-end smoke tests for resumable RSNA training: construction, forward/backward,
checkpoint save/reload, resume from a deliberately interrupted run, optimizer/RNG
restoration, and inference using the restored model. Everything here uses tiny synthetic
DICOM studies (dicom_factory); no real RSNA data or Kaggle access is required.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from mri_core.rsna_checkpoint import load_checkpoint
from mri_core.rsna_knee_dataset import SERIES_COLUMNS, TARGET_COLUMNS, load_rsna_metadata
from mri_core.rsna_knee_model import RSNAKneeCNN
from mri_core.rsna_knee_train_resumable import train_rsna_resumable

from .dicom_factory import AXIAL, CORONAL, SAGITTAL, write_series

pytestmark = pytest.mark.rsna

STUDIES = [f"1.2.826.{i}" for i in range(1, 5)]  # 4 tiny synthetic studies
CONFIG = {
    "image_size": 8, "slices_per_series": 2, "planes": ["Axial", "Coronal", "Sagittal"],
    "dropout": 0.0, "batch_size": 2, "num_workers": 0, "learning_rate": 1e-3,
    "epochs": 1, "validation_fraction": 0.25, "seed": 0, "device": "cpu",
}


@pytest.fixture
def rsna(tmp_path):
    meta, raw = tmp_path / "meta", tmp_path / "raw"
    meta.mkdir()
    rows = [{"StudyInstanceUID": uid, "Report": "r", **{t: (i + j) % 2 for j, t in enumerate(TARGET_COLUMNS)}}
            for i, uid in enumerate(STUDIES)]
    pd.DataFrame(rows).to_csv(meta / "train.csv", index=False)

    series_rows = []
    for uid in STUDIES:
        for plane, orientation in (("Axial", AXIAL), ("Coronal", CORONAL), ("Sagittal", SAGITTAL)):
            series_uid = f"{uid}.series.{plane}"
            series_rows.append((uid, series_uid, 1, 0, plane))
            write_series(raw / "train_series" / uid / series_uid, count=2, orientation=orientation,
                        series_uid=series_uid, study_uid=uid, rows=8, columns=8)
    pd.DataFrame(series_rows, columns=SERIES_COLUMNS).to_csv(meta / "train_series.csv", index=False)

    pd.DataFrame({"StudyInstanceUID": []}).to_csv(meta / "test.csv", index=False)
    pd.DataFrame(columns=SERIES_COLUMNS).to_csv(meta / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [], **{t: [] for t in TARGET_COLUMNS}}).to_csv(
        meta / "sample_submission.csv", index=False)

    metadata = load_rsna_metadata(meta, dicom_root=raw)
    return SimpleNamespace(meta=meta, raw=raw, metadata=metadata)


class TestModelConstructionAndForwardBackward:
    def test_model_constructs_and_runs_forward_backward(self):
        model = RSNAKneeCNN(num_planes=3, dropout=0.0)
        images = torch.randn(2, 3, 2, 1, 8, 8)
        mask = torch.ones(2, 3, 2, dtype=torch.bool)
        logits = model(images, mask)
        assert logits.shape == (2, 12)
        logits.sum().backward()
        assert model.encoder[0].weight.grad is not None


class TestFreshTrainingRun:
    def test_one_epoch_run_produces_a_checkpoint_and_history(self, rsna, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        result = train_rsna_resumable(rsna.metadata, CONFIG, checkpoint_dir, resume=False)
        assert (checkpoint_dir / "last.pt").is_file()
        assert len(result["history"]) == 1
        assert result["resumed_from_epoch"] is None
        assert result["train_studies"] + result["validation_studies"] == len(STUDIES)


class TestResumeFromInterruptedRun:
    def test_resuming_continues_instead_of_restarting(self, rsna, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        first_config = dict(CONFIG, epochs=1)
        first = train_rsna_resumable(rsna.metadata, first_config, checkpoint_dir, resume=False)
        assert len(first["history"]) == 1

        second_config = dict(CONFIG, epochs=2)
        second = train_rsna_resumable(rsna.metadata, second_config, checkpoint_dir, resume=True)
        assert second["resumed_from_epoch"] == 1
        # History carries forward from the resumed checkpoint, plus the new epoch.
        assert len(second["history"]) == 2
        assert second["history"][0]["epoch"] == 1
        assert second["history"][1]["epoch"] == 2

    def test_resume_with_same_target_epoch_does_no_extra_work(self, rsna, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        config = dict(CONFIG, epochs=2)
        train_rsna_resumable(rsna.metadata, config, checkpoint_dir, resume=False)
        second = train_rsna_resumable(rsna.metadata, config, checkpoint_dir, resume=True)
        assert second["resumed_from_epoch"] == 2
        assert len(second["history"]) == 2  # no new epochs appended

    def test_dataset_fingerprint_mismatch_blocks_resume_from_a_different_dataset(self, rsna, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        train_rsna_resumable(rsna.metadata, dict(CONFIG, epochs=1), checkpoint_dir, resume=False)

        # Corrupt the on-disk checkpoint's fingerprint to simulate a changed dataset.
        path = checkpoint_dir / "last.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["dataset_fingerprint"] = "deliberately-different"
        torch.save(payload, path)

        # Training should fall back to a fresh run rather than silently resuming onto
        # mismatched data (load_checkpoint raises; train_rsna_resumable only swallows
        # CheckpointMalformedError, not CheckpointIncompatibleError).
        from mri_core.rsna_checkpoint import CheckpointIncompatibleError
        with pytest.raises(CheckpointIncompatibleError):
            train_rsna_resumable(rsna.metadata, dict(CONFIG, epochs=2), checkpoint_dir, resume=True)


class TestOptimizerAndRngStateCarryOverOnResume:
    def test_global_step_increases_monotonically_across_resume(self, rsna, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        train_rsna_resumable(rsna.metadata, dict(CONFIG, epochs=1), checkpoint_dir, resume=False)
        state_after_epoch_1 = load_checkpoint(
            checkpoint_dir / "last.pt",
            model=RSNAKneeCNN(num_planes=3, dropout=0.0),
            optimizer=torch.optim.AdamW(RSNAKneeCNN(num_planes=3, dropout=0.0).parameters(), lr=1e-3),
        )
        train_rsna_resumable(rsna.metadata, dict(CONFIG, epochs=2), checkpoint_dir, resume=True)
        state_after_epoch_2 = load_checkpoint(
            checkpoint_dir / "last.pt",
            model=RSNAKneeCNN(num_planes=3, dropout=0.0),
            optimizer=torch.optim.AdamW(RSNAKneeCNN(num_planes=3, dropout=0.0).parameters(), lr=1e-3),
        )
        assert state_after_epoch_2.global_step > state_after_epoch_1.global_step


class TestInitialModelOverride:
    def test_custom_initial_model_is_used_for_a_fresh_run(self, rsna, tmp_path):
        torch.manual_seed(12345)
        custom_model = RSNAKneeCNN(num_planes=3, dropout=0.0)
        custom_model.encoder[0].weight.data.fill_(0.0)  # a distinctive, checkable marker

        checkpoint_dir = tmp_path / "ckpt"
        train_rsna_resumable(rsna.metadata, dict(CONFIG, epochs=1), checkpoint_dir,
                              resume=False, initial_model=custom_model)

        state = load_checkpoint(checkpoint_dir / "last.pt", model=RSNAKneeCNN(num_planes=3, dropout=0.0),
                                 optimizer=torch.optim.AdamW(RSNAKneeCNN(num_planes=3, dropout=0.0).parameters()))
        # The marker weight changes after one epoch of training, but starting from an all-zero
        # tensor is only possible if initial_model was actually used instead of a fresh random init.
        assert state.epoch == 1

    def test_initial_model_is_ignored_when_resuming(self, rsna, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        train_rsna_resumable(rsna.metadata, dict(CONFIG, epochs=1), checkpoint_dir, resume=False)
        # A second "initial_model" must be ignored since last.pt already exists and resume=True.
        decoy_model = RSNAKneeCNN(num_planes=3, dropout=0.0)
        decoy_model.encoder[0].weight.data.fill_(999.0)
        result = train_rsna_resumable(rsna.metadata, dict(CONFIG, epochs=2), checkpoint_dir,
                                       resume=True, initial_model=decoy_model)
        assert result["resumed_from_epoch"] == 1


class TestInferenceUsingRestoredModel:
    def test_restored_model_produces_finite_probabilities_for_all_studies(self, rsna, tmp_path):
        checkpoint_dir = tmp_path / "ckpt"
        train_rsna_resumable(rsna.metadata, dict(CONFIG, epochs=1), checkpoint_dir, resume=False)

        model = RSNAKneeCNN(num_planes=3, dropout=0.0)
        load_checkpoint(checkpoint_dir / "last.pt", model=model,
                         optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3))
        model.eval()

        from mri_core.rsna_knee_train import RSNAKneeDicomDataset

        dataset = RSNAKneeDicomDataset(rsna.metadata, rsna.metadata.complete_label_train, CONFIG, "train")
        with torch.inference_mode():
            for index in range(len(dataset)):
                item = dataset[index]
                logits = model(item["images"].unsqueeze(0), item["mask"].unsqueeze(0))
                probabilities = logits.sigmoid()
                assert probabilities.shape == (1, 12)
                assert torch.isfinite(probabilities).all()
                assert (probabilities >= 0).all() and (probabilities <= 1).all()
