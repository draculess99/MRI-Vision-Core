"""Smoke tests for resumable RSNA checkpoints: save/reload, RNG/optimizer restoration,
and rejection of incompatible checkpoints. All models here are tiny and synthetic.
"""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mri_core.rsna_checkpoint import (
    CheckpointIncompatibleError,
    CheckpointMalformedError,
    labeled_frame_fingerprint,
    load_checkpoint,
    save_checkpoint,
    studies_fingerprint,
)
from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from mri_core.rsna_knee_model import RSNAKneeCNN

CONFIG = {"image_size": 16, "slices_per_series": 2, "planes": ["Axial", "Coronal", "Sagittal"], "dropout": 0.0}


def make_model_and_optimizer(seed=0):
    torch.manual_seed(seed)
    model = RSNAKneeCNN(num_planes=3, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


class TestSaveAndReload:
    def test_round_trip_preserves_weights(self, tmp_path):
        model, optimizer = make_model_and_optimizer(seed=1)
        before = {k: v.clone() for k, v in model.state_dict().items()}

        path = tmp_path / "last.pt"
        save_checkpoint(
            path, model=model, optimizer=optimizer, config=CONFIG, target_columns=TARGET_COLUMNS,
            epoch=3, global_step=42, train_study_ids=["a", "b"], valid_study_ids=["c"],
            dataset_fingerprint="fp123", metrics={"history": []}, best_metric=0.6, is_best=True,
        )
        assert path.is_file()

        fresh_model, fresh_optimizer = make_model_and_optimizer(seed=999)  # different init
        state = load_checkpoint(path, model=fresh_model, optimizer=fresh_optimizer,
                                 expected_target_columns=TARGET_COLUMNS)

        for key, tensor in before.items():
            assert torch.equal(fresh_model.state_dict()[key], tensor)
        assert state.epoch == 3
        assert state.global_step == 42
        assert state.best_metric == 0.6
        assert state.is_best is True
        assert state.train_study_ids == ["a", "b"]
        assert state.valid_study_ids == ["c"]

    def test_atomic_write_leaves_no_temp_file_behind(self, tmp_path):
        model, optimizer = make_model_and_optimizer()
        path = tmp_path / "last.pt"
        save_checkpoint(
            path, model=model, optimizer=optimizer, config=CONFIG, target_columns=TARGET_COLUMNS,
            epoch=1, global_step=1, train_study_ids=[], valid_study_ids=[],
            dataset_fingerprint="fp", metrics={}, best_metric=None, is_best=False,
        )
        assert not path.with_suffix(path.suffix + ".tmp").exists()

    def test_best_checkpoint_is_a_separate_file(self, tmp_path):
        model, optimizer = make_model_and_optimizer()
        path = tmp_path / "last.pt"
        save_checkpoint(
            path, model=model, optimizer=optimizer, config=CONFIG, target_columns=TARGET_COLUMNS,
            epoch=1, global_step=1, train_study_ids=[], valid_study_ids=[],
            dataset_fingerprint="fp", metrics={}, best_metric=0.5, is_best=True,
        )
        assert (tmp_path / "last_best.pt").is_file()

    def test_non_best_epoch_does_not_overwrite_best(self, tmp_path):
        model, optimizer = make_model_and_optimizer()
        path = tmp_path / "last.pt"
        save_checkpoint(
            path, model=model, optimizer=optimizer, config=CONFIG, target_columns=TARGET_COLUMNS,
            epoch=1, global_step=1, train_study_ids=[], valid_study_ids=[],
            dataset_fingerprint="fp", metrics={}, best_metric=0.9, is_best=True,
        )
        best_bytes_before = (tmp_path / "last_best.pt").read_bytes()
        save_checkpoint(
            path, model=model, optimizer=optimizer, config=CONFIG, target_columns=TARGET_COLUMNS,
            epoch=2, global_step=2, train_study_ids=[], valid_study_ids=[],
            dataset_fingerprint="fp", metrics={}, best_metric=0.9, is_best=False,
        )
        assert (tmp_path / "last_best.pt").read_bytes() == best_bytes_before


class TestOptimizerAndRngRestoration:
    def test_optimizer_momentum_state_is_restored(self, tmp_path):
        model, optimizer = make_model_and_optimizer(seed=2)
        images = torch.randn(1, 3, 2, 1, 16, 16)
        mask = torch.ones(1, 3, 2, dtype=torch.bool)
        logits = model(images, mask)
        logits.sum().backward()
        optimizer.step()  # populate AdamW exp_avg/exp_avg_sq state
        before_state = optimizer.state_dict()

        path = tmp_path / "last.pt"
        save_checkpoint(
            path, model=model, optimizer=optimizer, config=CONFIG, target_columns=TARGET_COLUMNS,
            epoch=1, global_step=1, train_study_ids=[], valid_study_ids=[],
            dataset_fingerprint="fp", metrics={}, best_metric=None, is_best=False,
        )

        fresh_model, fresh_optimizer = make_model_and_optimizer(seed=2)
        load_checkpoint(path, model=fresh_model, optimizer=fresh_optimizer, expected_target_columns=TARGET_COLUMNS)
        after_state = fresh_optimizer.state_dict()
        assert after_state["state"].keys() == before_state["state"].keys()
        for param_id in before_state["state"]:
            assert torch.equal(after_state["state"][param_id]["exp_avg"], before_state["state"][param_id]["exp_avg"])

    def test_rng_restoration_reproduces_subsequent_random_draws(self, tmp_path):
        model, optimizer = make_model_and_optimizer()
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        path = tmp_path / "last.pt"
        save_checkpoint(
            path, model=model, optimizer=optimizer, config=CONFIG, target_columns=TARGET_COLUMNS,
            epoch=1, global_step=1, train_study_ids=[], valid_study_ids=[],
            dataset_fingerprint="fp", metrics={}, best_metric=None, is_best=False,
        )

        expected_python = random.random()
        expected_numpy = np.random.rand()
        expected_torch = torch.rand(1).item()

        # Perturb all RNGs, then restore from checkpoint and confirm the exact draws repeat.
        random.seed(999); np.random.seed(999); torch.manual_seed(999)
        fresh_model, fresh_optimizer = make_model_and_optimizer()
        load_checkpoint(path, model=fresh_model, optimizer=fresh_optimizer, expected_target_columns=TARGET_COLUMNS,
                         restore_rng=True)

        assert random.random() == expected_python
        assert np.random.rand() == expected_numpy
        assert torch.rand(1).item() == expected_torch


class TestRejectsIncompatibleCheckpoints:
    def test_missing_file_raises_malformed(self, tmp_path):
        model, optimizer = make_model_and_optimizer()
        with pytest.raises(CheckpointMalformedError):
            load_checkpoint(tmp_path / "missing.pt", model=model, optimizer=optimizer)

    def test_target_column_mismatch_is_rejected(self, tmp_path):
        model, optimizer = make_model_and_optimizer()
        path = tmp_path / "last.pt"
        save_checkpoint(
            path, model=model, optimizer=optimizer, config=CONFIG, target_columns=TARGET_COLUMNS,
            epoch=1, global_step=1, train_study_ids=[], valid_study_ids=[],
            dataset_fingerprint="fp", metrics={}, best_metric=None, is_best=False,
        )
        fresh_model, fresh_optimizer = make_model_and_optimizer()
        with pytest.raises(CheckpointIncompatibleError):
            load_checkpoint(path, model=fresh_model, optimizer=fresh_optimizer,
                             expected_target_columns=["abnormal", "acl", "meniscus"])

    def test_dataset_fingerprint_mismatch_is_rejected(self, tmp_path):
        model, optimizer = make_model_and_optimizer()
        path = tmp_path / "last.pt"
        save_checkpoint(
            path, model=model, optimizer=optimizer, config=CONFIG, target_columns=TARGET_COLUMNS,
            epoch=1, global_step=1, train_study_ids=[], valid_study_ids=[],
            dataset_fingerprint="fp-old", metrics={}, best_metric=None, is_best=False,
        )
        fresh_model, fresh_optimizer = make_model_and_optimizer()
        with pytest.raises(CheckpointIncompatibleError):
            load_checkpoint(path, model=fresh_model, optimizer=fresh_optimizer,
                             expected_target_columns=TARGET_COLUMNS,
                             expected_dataset_fingerprint="fp-new")

    def test_incompatible_model_architecture_is_rejected(self, tmp_path):
        model, optimizer = make_model_and_optimizer()
        path = tmp_path / "last.pt"
        save_checkpoint(
            path, model=model, optimizer=optimizer, config=CONFIG, target_columns=TARGET_COLUMNS,
            epoch=1, global_step=1, train_study_ids=[], valid_study_ids=[],
            dataset_fingerprint="fp", metrics={}, best_metric=None, is_best=False,
        )
        mismatched_model = RSNAKneeCNN(num_planes=1, dropout=0.0)
        mismatched_optimizer = torch.optim.AdamW(mismatched_model.parameters(), lr=1e-3)
        with pytest.raises(CheckpointIncompatibleError):
            load_checkpoint(path, model=mismatched_model, optimizer=mismatched_optimizer,
                             expected_target_columns=TARGET_COLUMNS)

    def test_malformed_checkpoint_missing_required_keys_is_rejected(self, tmp_path):
        path = tmp_path / "bad.pt"
        torch.save({"model": {}, "optimizer": {}}, path)
        model, optimizer = make_model_and_optimizer()
        with pytest.raises(CheckpointMalformedError):
            load_checkpoint(path, model=model, optimizer=optimizer)


class TestFingerprints:
    def test_studies_fingerprint_is_order_independent(self):
        assert studies_fingerprint(["a", "b", "c"]) == studies_fingerprint(["c", "a", "b"])

    def test_studies_fingerprint_changes_with_membership(self):
        assert studies_fingerprint(["a", "b"]) != studies_fingerprint(["a", "c"])

    def test_labeled_frame_fingerprint_order_independent(self):
        import pandas as pd

        rows = [{"StudyInstanceUID": "s1", **{t: float(i % 2) for i, t in enumerate(TARGET_COLUMNS)}},
                {"StudyInstanceUID": "s2", **{t: float((i + 1) % 2) for i, t in enumerate(TARGET_COLUMNS)}}]
        frame_a = pd.DataFrame(rows)
        frame_b = pd.DataFrame(list(reversed(rows)))
        assert labeled_frame_fingerprint(frame_a) == labeled_frame_fingerprint(frame_b)
