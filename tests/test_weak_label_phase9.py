"""Phase 9: cache identity, cross-session restore, epoch-limited sessions, RNG restore on a different GPU count."""

import json
import shutil
import warnings

import numpy as np
import pytest
import torch

from mri_core import rsna_checkpoint
from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from mri_core.weak_label_data import WeakLabelDicomDataset
from mri_core.weak_label_train import train_weak_label
from scripts.kaggle_weak_label_experiment import main as cli

from .test_weak_label_dicom_and_submission import CFG as DICOM_CFG
from .test_weak_label_dicom_and_submission import values_masks, world  # noqa: F401
from .test_weak_label_experiment import CFG, make_dataset
from .test_weak_label_real_smoke import build_manifest_for

pytestmark = pytest.mark.rsna


class TestCacheIdentity:
    def dataset(self, world, cache, **cfg):
        values, masks = values_masks(len(world.train_ids))
        return WeakLabelDicomDataset(world.metadata, world.train_ids, values, masks, {**DICOM_CFG, **cfg}, cache_dir=cache)

    def test_entry_stored_under_another_study_is_rejected_and_rebuilt(self, world, tmp_path, monkeypatch):
        from mri_core.rsna_knee_train import RSNAKneeDicomDataset
        cache = tmp_path / "cache"
        ds = self.dataset(world, cache)
        ds[0], ds[1]
        entries = [torch.load(e, map_location="cpu", weights_only=True) for e in cache.glob("*.pt")]
        assert all(v["meta"]["version"] == 1 and v["meta"]["uid"] in world.train_ids for v in entries)
        file0, file1 = ds._cache_path(world.train_ids[0]), ds._cache_path(world.train_ids[1])
        shutil.copyfile(file1, file0)  # study 1's entry under study 0's name: right shape, wrong identity
        decoded = []
        original = RSNAKneeDicomDataset.__getitem__
        monkeypatch.setattr(RSNAKneeDicomDataset, "__getitem__", lambda self, i: (decoded.append(i), original(self, i))[1])
        self.dataset(world, cache)[0]
        assert decoded == [0]  # it was decoded again instead of trusting the misnamed entry
        assert torch.load(file0, weights_only=True)["meta"]["uid"] == world.train_ids[0]  # and repaired in place
        decoded.clear()
        self.dataset(world, cache)[0]
        assert decoded == []  # now a genuine hit

    def test_other_preprocessing_settings_never_reuse_old_entries(self, world, tmp_path):
        cache = tmp_path / "cache"
        self.dataset(world, cache)[0]
        other = self.dataset(world, cache, image_size=16)
        sample = other[0]
        assert sample["images"].shape[-1] == 16 and len(list(cache.glob("*.pt"))) == 2


class TestEpochLimitedSessionsAndCrossDirRestore:
    def run(self, out, epochs, **kw):
        train, val = make_dataset(24, 0), make_dataset(8, 1, start=100)
        return train_weak_label(train, val, {**CFG, "epochs": epochs}, out, fingerprint="fp", train_ids=train.ids,
                                val_ids=val.ids, **kw)

    def test_session_stops_after_n_epochs_and_a_copied_directory_continues_it(self, tmp_path):
        full = self.run(tmp_path / "full", 3)
        first = self.run(tmp_path / "s1", 3, stop_after_epochs=1)
        assert first["status"] == "paused_epoch_limit" and first["final_epoch"] == 1
        shutil.copytree(tmp_path / "s1", tmp_path / "s2")  # what restoring a previous session's output looks like
        second = self.run(tmp_path / "s2", 3)
        assert second["status"] == "complete" and second["resumed_from_epoch"] == 1 and second["final_epoch"] == 3
        a = torch.load(tmp_path / "full" / "last.pt", map_location="cpu", weights_only=True)
        b = torch.load(tmp_path / "s2" / "last.pt", map_location="cpu", weights_only=True)
        for k in a["model"]:
            assert torch.equal(a["model"][k], b["model"][k]), k
        assert a["global_step"] == b["global_step"] and a["scheduler"]["last_epoch"] == b["scheduler"]["last_epoch"] == 3
        assert [h["train_loss"] for h in full["history"]] == [h["train_loss"] for h in second["history"]]

    def test_session_log_records_class_weights_and_trainable_counts(self, tmp_path):
        self.run(tmp_path / "log", 1)
        start = json.loads((tmp_path / "log" / "runtime_log.jsonl").read_text().splitlines()[0])
        assert len(start["pos_weight"]) == 12 and len(start["trainable_entries"]) == 12
        assert all(w >= 1.0 for w in start["pos_weight"]) and start["trainable_entries"][0] > 0

    def test_verify_checkpoint_cli_confirms_split_manifest_and_state(self, world, tmp_path, capsys):
        manifest = build_manifest_for(world)
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({**DICOM_CFG, "epochs": 3, "val_fraction": 0.25, "cache_dir": str(tmp_path / "cache")}))
        common = ["--data-dir", str(world.meta), "--dicom-root", str(world.raw), "--manifest", str(manifest),
                  "--config", str(cfg), "--smoke-subset", "6", "--min-per-class", "2"]
        assert cli(["train", *common, "--arm", "A", "--stop-after-epochs", "1", "--work-dir", str(tmp_path / "s1")]) == 0
        capsys.readouterr()
        shutil.copytree(tmp_path / "s1", tmp_path / "restored")
        assert cli(["verify-checkpoint", *common, "--checkpoint", str(tmp_path / "restored" / "last.pt"),
                    "--work-dir", str(tmp_path / "v")]) == 0
        out = capsys.readouterr().out
        report = json.loads(out[out.index('{\n  "checkpoint_sha256"'):])
        assert report["compatible"] and report["epoch"] == 1 and report["global_step"] > 0
        assert report["fingerprint_matches"] and report["train_split_matches"] and report["validation_split_matches"]
        assert report["scheduler_last_epoch"] == 1 and all(v for k, v in report["state_present"].items() if k != "scaler")
        # a different selection must be reported incompatible, not silently accepted
        other = [c if c != "6" else "5" for c in common]
        assert cli(["verify-checkpoint", *other, "--checkpoint", str(tmp_path / "restored" / "last.pt"),
                    "--work-dir", str(tmp_path / "v2")]) == 1
        # and the restored directory continues to epoch 3
        assert cli(["train", *common, "--arm", "A", "--work-dir", str(tmp_path / "restored")]) == 0
        assert torch.load(tmp_path / "restored" / "last.pt", map_location="cpu", weights_only=True)["epoch"] == 3


class TestCudaRngRestoreOnDifferentDeviceCount:
    def test_overlapping_devices_are_restored_and_a_warning_is_raised(self, monkeypatch):
        applied = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, index: applied.append(index))
        torch.manual_seed(0)
        checkpoint = {"torch_rng": torch.get_rng_state(), "cuda_rng": [torch.zeros(1), torch.zeros(1)],
                      "python_rng": __import__("random").getstate(), "numpy_rng": [*np.random.get_state()[:1], np.random.get_state()[1].tolist(), *np.random.get_state()[2:]]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rsna_checkpoint._restore_rng_states(checkpoint)
        assert applied == [0] and any("CUDA RNG" in str(w.message) for w in caught)
