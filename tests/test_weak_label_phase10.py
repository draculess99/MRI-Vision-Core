"""Phase 10: early stopping, restored-cache use, exclusion lists, cache warming and split summaries."""

import json
import shutil

import pytest
import torch

from mri_core.weak_label_data import WeakLabelDicomDataset
from mri_core.weak_label_train import early_stop_due, train_weak_label
from scripts.kaggle_weak_label_experiment import main as cli

from .test_weak_label_dicom_and_submission import CFG as DICOM_CFG
from .test_weak_label_dicom_and_submission import values_masks, world  # noqa: F401
from .test_weak_label_experiment import CFG, make_dataset
from .test_weak_label_real_smoke import build_manifest_for

pytestmark = pytest.mark.rsna


def hist(*losses):
    return [{"val_loss": v} for v in losses]


class TestEarlyStoppingRule:
    def test_patience_counts_epochs_without_a_min_delta_improvement(self):
        assert not early_stop_due(hist(1.0, 0.9, 0.8), 3, 0.002)
        assert not early_stop_due(hist(1.0, 0.9, 0.95, 0.96), 3, 0.002)  # only two stale epochs
        assert early_stop_due(hist(1.0, 0.9, 0.95, 0.96, 0.91), 3, 0.002)
        assert early_stop_due(hist(1.0, 0.9, 0.899, 0.898, 0.8985), 3, 0.002)  # improvements below min_delta do not count
        assert not early_stop_due(hist(1.0, 1.0, 1.0, 1.0), 0, 0.002)  # patience 0 disables it

    def run(self, out, epochs, **cfg):
        train, val = make_dataset(24, 0), make_dataset(8, 1, start=100)
        return train_weak_label(train, val, {**CFG, "epochs": epochs, **cfg}, out, fingerprint="fp",
                                train_ids=train.ids, val_ids=val.ids)

    def test_training_stops_early_reports_the_best_epoch_and_keeps_best_checkpoint_separate(self, tmp_path):
        result = self.run(tmp_path / "es", 12, early_stopping_patience=1, early_stopping_min_delta=10.0)
        assert result["status"] == "early_stopped" and result["final_epoch"] == 2  # nothing beats a huge min_delta
        best = torch.load(tmp_path / "es" / "last_best.pt", map_location="cpu", weights_only=True)
        last = torch.load(tmp_path / "es" / "last.pt", map_location="cpu", weights_only=True)
        assert last["epoch"] == 2 and best["epoch"] == result["best_epoch"]
        assert (tmp_path / "es" / "last.pt").read_bytes() != (tmp_path / "es" / "last_best.pt").read_bytes() or best["epoch"] == last["epoch"]

    def test_a_run_that_already_stopped_early_does_not_train_again_on_resume(self, tmp_path):
        first = self.run(tmp_path / "r", 12, early_stopping_patience=1, early_stopping_min_delta=10.0)
        before = (tmp_path / "r" / "last.pt").read_bytes()
        again = self.run(tmp_path / "r", 12, early_stopping_patience=1, early_stopping_min_delta=10.0)
        assert again["status"] == "early_stopped" and again["final_epoch"] == first["final_epoch"]
        assert (tmp_path / "r" / "last.pt").read_bytes() == before


class TestRestoredCacheIsActuallyUsed:
    def test_restored_entries_are_hits_and_nothing_is_decoded_again(self, world, tmp_path, monkeypatch):
        from mri_core.rsna_knee_train import RSNAKneeDicomDataset
        values, masks = values_masks(len(world.train_ids))
        first = WeakLabelDicomDataset(world.metadata, world.train_ids, values, masks, DICOM_CFG, cache_dir=tmp_path / "old")
        originals = [first[i] for i in range(len(world.train_ids))]
        assert first.cache_misses == len(world.train_ids) and first.cache_hits == 0
        shutil.copytree(tmp_path / "old", tmp_path / "restored")  # what the session-2 restore step does
        monkeypatch.setattr(RSNAKneeDicomDataset, "__getitem__", lambda *a: (_ for _ in ()).throw(AssertionError("decoded")))
        restored = WeakLabelDicomDataset(world.metadata, world.train_ids, values, masks, DICOM_CFG, cache_dir=tmp_path / "restored")
        for i, sample in enumerate(originals):
            assert torch.equal(restored[i]["images"], sample["images"])
        assert restored.cache_hits == len(world.train_ids) and restored.cache_misses == 0

    def test_preflight_after_restore_reports_all_hits(self, world, tmp_path, capsys):
        manifest = build_manifest_for(world)
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({**DICOM_CFG, "epochs": 1, "val_fraction": 0.25, "cache_dir": str(tmp_path / "cache")}))
        common = ["--data-dir", str(world.meta), "--dicom-root", str(world.raw), "--manifest", str(manifest), "--config", str(cfg)]
        assert cli(["cache-warm", *common, "--work-dir", str(tmp_path / "w")]) == 0
        warm = json.loads((tmp_path / "w" / "cache_warm.json").read_text())
        assert warm["cache_entries_before"] == 0 and warm["cache_entries_after"] == warm["studies"] == 8
        assert cli(["real-preflight", *common, "--work-dir", str(tmp_path / "p")]) == 0
        report = json.loads((tmp_path / "p" / "real_preflight.json").read_text())
        assert report["cache_hits"] == 8 and report["cache_misses"] == 0


class TestExclusionsWarmAndSplitSummary:
    def test_cache_warm_lists_unreadable_studies_and_the_exclusion_list_removes_them(self, world, tmp_path):
        manifest = build_manifest_for(world)
        victim = world.train_ids[2]
        for f in (world.raw / "train_series" / victim / f"{victim}.s.Axial").iterdir():
            f.write_bytes(b"corrupt")
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({**DICOM_CFG, "epochs": 2, "val_fraction": 0.25, "cache_dir": str(tmp_path / "cache"),
                                   "early_stopping_patience": 3}))
        common = ["--data-dir", str(world.meta), "--dicom-root", str(world.raw), "--manifest", str(manifest), "--config", str(cfg)]
        assert cli(["cache-warm", *common, "--work-dir", str(tmp_path / "w")]) == 0
        assert json.loads((tmp_path / "w" / "excluded_unreadable.json").read_text()) == [victim]
        excl = str(tmp_path / "w" / "excluded_unreadable.json")
        assert cli(["train", *common, "--exclude-studies", excl, "--arm", "A", "--work-dir", str(tmp_path / "run")]) == 0
        state = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=True)
        assert victim not in state["train_study_ids"] + state["valid_study_ids"]
        assert len(state["train_study_ids"]) + len(state["valid_study_ids"]) == 7
        assert state["metrics"]["latest"]["invalid_samples"] == 0
        summary = json.loads((tmp_path / "run" / "split_summary.json").read_text())
        assert summary["included_studies"] == 7 and summary["train_studies"] + summary["validation_studies"] == 7
        assert set(summary["train_targets"]) == {"ACL", "Medial Meniscus", "Lateral Meniscus"}
        assert summary["train_targets"]["ACL"]["positive"] + summary["train_targets"]["ACL"]["negative"] == summary["train_studies"]
