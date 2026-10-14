"""Smoke tests for resumable MRNet training: tiny synthetic data only, no real MRNet files."""

import pytest

torch = pytest.importorskip("torch")

from mri_core.cnn_train import smoke_fixture
from mri_core.cnn_train_resumable import TrainingInterrupted, train_mrnet_resumable
from mri_core.labels import SPLITS, load_labels
from mri_core.manifest import build_manifest

CONFIG = {
    "planes": ["axial"], "slices": 2, "resolution": 16, "epochs": 1, "batch_size": 2,
    "num_workers": 0, "seed": 0, "learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.0,
    "rotation_degrees": 0.0, "translation_fraction": 0.0, "contrast_fraction": 0.0,
}


@pytest.fixture
def manifest(tmp_path):
    labels_dir = smoke_fixture(tmp_path)
    labels = load_labels(labels_dir)
    return build_manifest(labels, tmp_path)


class TestFreshRun:
    def test_one_epoch_writes_checkpoint(self, manifest, tmp_path):
        result = train_mrnet_resumable(manifest, CONFIG, tmp_path / "ckpt", resume=False)
        assert (tmp_path / "ckpt" / "last.pt").is_file()
        assert len(result["history"]) == 1
        assert result["resumed_from_epoch"] is None


class TestResume:
    def test_resuming_does_not_repeat_epoch_one(self, manifest, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        train_mrnet_resumable(manifest, dict(CONFIG, epochs=1), ckpt_dir, resume=False)
        second = train_mrnet_resumable(manifest, dict(CONFIG, epochs=2), ckpt_dir, resume=True)
        assert second["resumed_from_epoch"] == 1
        assert [h["epoch"] for h in second["history"]] == [1, 2]


class TestInterruptHandler:
    def test_signal_during_training_stops_cleanly_without_partial_checkpoint(self, manifest, tmp_path):
        import signal as signal_module

        from mri_core.cnn_train_resumable import GracefulInterruptHandler

        ckpt_dir = tmp_path / "ckpt"
        # Pre-seed one real completed epoch so we can confirm it's untouched after interruption.
        train_mrnet_resumable(manifest, dict(CONFIG, epochs=1), ckpt_dir, resume=False)
        before_bytes = (ckpt_dir / "last.pt").read_bytes()

        original_init = GracefulInterruptHandler.__enter__

        def fire_immediately(self):
            original_init(self)
            self.requested = True  # simulate a signal having already arrived
            return self

        GracefulInterruptHandler.__enter__ = fire_immediately
        try:
            with pytest.raises(TrainingInterrupted):
                train_mrnet_resumable(manifest, dict(CONFIG, epochs=2), ckpt_dir, resume=True)
        finally:
            GracefulInterruptHandler.__enter__ = original_init

        after_bytes = (ckpt_dir / "last.pt").read_bytes()
        assert after_bytes == before_bytes  # no partial epoch 2 state was ever written
