"""Pre-GPU review fixes: AMP/scaler path, cache integrity, failure attribution, submission audit, export integrity."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mri_core import weak_label_data, weak_label_train
from mri_core.rsna_checkpoint import CheckpointIncompatibleError
from mri_core.rsna_knee_dataset import SUBMISSION_COLUMNS, TARGET_COLUMNS
from mri_core.weak_label_data import WeakLabelDicomDataset
from mri_core.weak_label_models import ResNet18Backbone
from mri_core.weak_label_submit import SubmissionError, load_checkpoint_model, run_submission
from mri_core.weak_label_train import DEFAULT_CONFIG, train_weak_label
from scripts.kaggle_weak_label_experiment import main as experiment_cli

from .test_weak_label_dicom_and_submission import CFG as DICOM_CFG
from .test_weak_label_dicom_and_submission import save_baseline, save_new, values_masks, world  # noqa: F401
from .test_weak_label_experiment import CFG, fit, make_dataset

pytestmark = pytest.mark.rsna

REPO = Path(__file__).resolve().parents[1]


class TestAmpAndScaler:
    @pytest.fixture(autouse=True)
    def cpu_amp(self, monkeypatch):
        monkeypatch.setattr(weak_label_train, "AMP_DEVICE_TYPES", ("cuda", "cpu"))

    def test_scaler_state_is_saved_restored_and_training_stays_finite(self, tmp_path):
        result = fit(tmp_path / "a", epochs=3, **{})
        # fit() uses the default (amp False in CFG? DEFAULT_CONFIG amp=False) so run explicitly with amp below
        train, val = make_dataset(24, 0), make_dataset(8, 1, start=100)
        cfg = {**CFG, "amp": True, "epochs": 2}
        first = train_weak_label(train, val, {**cfg, "epochs": 1}, tmp_path / "amp", fingerprint="fp",
                                 train_ids=train.ids, val_ids=val.ids)
        state = torch.load(tmp_path / "amp" / "last.pt", map_location="cpu", weights_only=True)
        assert state["scaler"] and "scale" in state["scaler"]
        second = train_weak_label(train, val, cfg, tmp_path / "amp", fingerprint="fp", train_ids=train.ids, val_ids=val.ids)
        assert second["resumed_from_epoch"] == 1 and second["final_epoch"] == 2
        assert all(np.isfinite([h["train_loss"], h["val_loss"]]).all() for h in first["history"] + second["history"])
        assert result["final_epoch"] == 3

    def test_resuming_with_a_different_amp_setting_is_refused(self, tmp_path):
        train, val = make_dataset(24, 0), make_dataset(8, 1, start=100)
        train_weak_label(train, val, {**CFG, "amp": False, "epochs": 1}, tmp_path / "m", fingerprint="fp",
                         train_ids=train.ids, val_ids=val.ids)
        with pytest.raises(CheckpointIncompatibleError):
            train_weak_label(train, val, {**CFG, "amp": True, "epochs": 2}, tmp_path / "m", fingerprint="fp",
                             train_ids=train.ids, val_ids=val.ids)

    def test_cuda_autocast_would_use_float16_never_bfloat16(self):
        source = (REPO / "mri_core" / "weak_label_train.py").read_text(encoding="utf-8")
        assert "torch.float16 if torch_device.type == \"cuda\"" in source and "bfloat16 if torch_device.type == \"cuda\"" not in source


class TestPretrainedArchitectureMatch:
    def test_backbone_matches_the_standard_resnet18_layout(self):
        net = ResNet18Backbone()
        state = net.state_dict()
        assert len(state) == 122 and sum(p.numel() for p in net.parameters()) == 11_689_512
        for key in ("conv1.weight", "bn1.running_mean", "bn1.num_batches_tracked", "layer1.0.conv1.weight",
                    "layer2.0.downsample.0.weight", "layer2.0.downsample.1.running_var", "layer4.1.bn2.weight", "fc.weight"):
            assert key in state, key
        assert state["fc.weight"].shape == (1000, 512) and state["layer4.1.conv2.weight"].shape == (512, 512, 3, 3)


class TestCacheIntegrity:
    def dataset(self, world, tmp_path):
        values, masks = values_masks(len(world.train_ids))
        return WeakLabelDicomDataset(world.metadata, world.train_ids, values, masks, DICOM_CFG,
                                     cache_dir=tmp_path / "cache")

    def test_corrupt_or_stale_cache_entries_are_rebuilt_not_trusted(self, world, tmp_path):
        ds = self.dataset(world, tmp_path)
        good = ds[0]
        entry = next((tmp_path / "cache").glob("*.pt"))
        entry.write_bytes(b"truncated garbage")  # simulates a damaged cache file
        repaired = ds[0]
        assert torch.equal(good["images"], repaired["images"])
        assert torch.load(entry, map_location="cpu", weights_only=True)["images"].shape == good["images"].shape
        torch.save({"images": torch.zeros(1, 1, 1, 2, 2), "mask": torch.ones(1, 1, dtype=torch.bool)}, entry)  # wrong shape
        assert torch.equal(good["images"], ds[0]["images"])

    def test_no_temp_files_are_left_behind_and_version_changes_the_key(self, world, tmp_path, monkeypatch):
        ds = self.dataset(world, tmp_path)
        for i in range(3):
            ds[i]
        assert not list((tmp_path / "cache").glob("*.tmp")) and len(list((tmp_path / "cache").glob("*.pt"))) == 3
        before = ds._cache_path("x")
        monkeypatch.setattr(weak_label_data, "CACHE_VERSION", 2)
        assert ds._cache_path("x") != before


class TestFailureAttribution:
    def test_skipped_training_studies_are_recorded_by_id(self, world, tmp_path):
        victim = world.train_ids[0]
        for f in (world.raw / "train_series" / victim / f"{victim}.s.Axial").iterdir():
            f.write_bytes(b"corrupt")
        values, masks = values_masks(len(world.train_ids))
        make = lambda ids, v, m: WeakLabelDicomDataset(world.metadata, ids, v, m, DICOM_CFG, on_error="skip")
        ids = world.train_ids
        # victim goes to the training half: put it first and give the other studies to validation
        train = make(ids[:5], values[:5], masks[:5])
        val = make(ids[5:], values[5:], masks[5:])
        result = train_weak_label(train, val, {**DICOM_CFG, "max_invalid_fraction": 0.5}, tmp_path / "run",
                                  fingerprint="f", train_ids=ids[:5], val_ids=ids[5:])
        recorded = json.loads((tmp_path / "run" / "invalid_studies.json").read_text())
        assert recorded == [victim] and result["invalid_study_ids_file"].endswith("invalid_studies.json")
        assert result["history"][0]["invalid_studies"] == 1


class TestSubmissionAudit:
    def test_only_approved_targets_can_be_replaced(self, world):
        save_baseline(world.tmp / "b.pt", 1)
        save_new(world.tmp / "n.pt", 2)
        with pytest.raises(SubmissionError, match="approved"):
            run_submission(world.metadata, [world.tmp / "b.pt"], world.tmp / "n.pt", ["Effusion"], world.tmp / "o.csv")
        out = run_submission(world.metadata, [world.tmp / "b.pt"], world.tmp / "n.pt", ["Effusion"], world.tmp / "o2.csv",
                             approved_targets=["Effusion"])
        assert out["targets_from_new_model"] == ["Effusion"]

    def test_audit_file_attributes_every_fallback_and_target_source(self, world):
        save_baseline(world.tmp / "b.pt", 1)
        save_new(world.tmp / "n.pt", 2)
        victim = world.test_ids[1]
        for f in (world.raw / "test_series" / victim / f"{victim}.s.Coronal").iterdir():
            f.write_bytes(b"corrupt")
        result = run_submission(world.metadata, [world.tmp / "b.pt"], world.tmp / "n.pt",
                                ["ACL", "Medial Meniscus", "Lateral Meniscus"], world.tmp / "o.csv", max_failed_fraction=0.5)
        audit = json.loads(Path(result["audit"]).read_text())
        assert result["fallback_studies"] == [victim] and audit["fallback_count"] == 1
        entry = audit["fallback_studies"][0]
        assert entry["study"] == victim and entry["fill"] == "mean_of_readable_studies"
        assert sorted(entry["unreadable_in"]) == sorted([str(world.tmp / "b.pt"), str(world.tmp / "n.pt")])
        assert audit["target_source"]["ACL"] == "new_model" and audit["target_source"]["MCL"] == "baseline"
        assert len(audit["checkpoints"]) == 2 and all(len(h) == 64 for h in audit["checkpoints"].values())
        assert audit["seconds_per_study"] >= 0

    def test_a_clean_run_reports_no_fallbacks(self, world):
        save_baseline(world.tmp / "b.pt", 1)
        result = run_submission(world.metadata, [world.tmp / "b.pt"], None, [], world.tmp / "o.csv")
        assert result["fallback_studies"] == [] and json.loads(Path(result["audit"]).read_text())["fallback_count"] == 0

    @pytest.mark.parametrize("path", ["outputs/rsna-knee-cv/fold0/checkpoint_best.pt",
                                      "outputs/rsna-knee-ensemble/checkpoint_rsna_stratified_fold1.pt",
                                      "outputs/rsna-fivefold-benchmark/random/fold0/last.pt"])
    def test_existing_baseline_checkpoints_load_and_produce_a_valid_submission(self, world, path):
        checkpoint = REPO / path
        if not checkpoint.is_file():
            pytest.skip("local baseline checkpoint not present")
        model, config = load_checkpoint_model(checkpoint, torch.device("cpu"))
        assert type(model).__name__ == "RSNAKneeCNN"
        run_submission(world.metadata, [checkpoint], None, [], world.tmp / "real.csv")  # synthetic MRI, real weights
        frame = pd.read_csv(world.tmp / "real.csv")
        values = frame[list(TARGET_COLUMNS)].to_numpy()
        assert tuple(frame.columns) == SUBMISSION_COLUMNS and np.isfinite(values).all()
        assert values.min() >= 0 and values.max() <= 1


class TestExportAndSmokeCli:
    def test_export_verifies_copies_and_records_provenance(self, tmp_path, capsys):
        fit(tmp_path / "run", epochs=2)
        code = experiment_cli(["export", "--work-dir", str(tmp_path / "run"), "--export-dir", str(tmp_path / "out")])
        assert code == 0
        manifest = json.loads((tmp_path / "out" / "MANIFEST.json").read_text())
        provenance = json.loads((tmp_path / "out" / "provenance.json").read_text())
        assert provenance["epoch"] == 2 and provenance["dataset_fingerprint"] == "fp" and provenance["arm"] == "A"
        assert provenance["n_train_studies"] == 24 and provenance["target_columns"] == list(TARGET_COLUMNS)
        import hashlib
        for name, info in manifest.items():
            if name == "runtime":
                continue
            assert hashlib.sha256((tmp_path / "out" / name).read_bytes()).hexdigest() == info["sha256"]
        assert (tmp_path / "out" / "weak_label_run_export.zip").is_file()
        with pytest.raises(FileNotFoundError):
            experiment_cli(["export", "--work-dir", str(tmp_path / "empty"), "--export-dir", str(tmp_path / "o2")])

    def test_synthetic_smoke_passes_all_checks_on_cpu(self, tmp_path):
        code = experiment_cli(["smoke", "--work-dir", str(tmp_path / "smoke"), "--device", "cpu", "--studies", "12",
                               "--image-size", "16", "--slices", "3", "--batch-size", "4"])
        report = json.loads((tmp_path / "smoke" / "smoke_report.json").read_text())
        assert code == 0 and report["checks_passed"] and set(report["arms"]) == {"A", "B", "C"}
        assert all(all(a["checks"].values()) for a in report["arms"].values())
        assert report["environment"]["cuda_available"] is False


class TestTrainCliEndToEnd:
    def build_manifest(self, world):
        from mri_core.report_label_rules import RuleBasedExtractor
        from mri_core.weak_label_manifest import build_manifest, write_manifest
        en = "Findings: The study of the knee was performed and the following is noted in this report.\n"
        train = pd.read_csv(world.meta / "train.csv", dtype={"StudyInstanceUID": str})
        reports = ["Complete tear of the anterior cruciate ligament.", "No tear of the anterior cruciate ligament."]
        train["Report"] = [en + reports[i % 2] for i in range(len(train))]
        rows, summary = build_manifest(train, RuleBasedExtractor(), active_targets=["ACL"])
        return write_manifest(rows, summary, world.tmp / "weak.jsonl")[0]

    def test_cli_train_resume_subset_and_missing_series_attribution(self, world, tmp_path):
        manifest = self.build_manifest(world)
        victim = world.train_ids[0]
        import shutil
        shutil.rmtree(world.raw / "train_series" / victim / f"{victim}.s.Sagittal")
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({**DICOM_CFG, "epochs": 2, "val_fraction": 0.3, "cache_dir": str(tmp_path / "cache")}))
        base = ["train", "--data-dir", str(world.meta), "--dicom-root", str(world.raw), "--manifest", str(manifest),
                "--config", str(cfg), "--work-dir", str(tmp_path / "wd"), "--max-studies", "6"]
        assert experiment_cli(base + ["--epochs-override", "1"]) == 0
        assert experiment_cli(base + ["--epochs-override", "1"]) == 0  # already complete -> no-op resume
        state = torch.load(tmp_path / "wd" / "last.pt", map_location="cpu", weights_only=True)
        used = state["train_study_ids"] + state["valid_study_ids"]
        assert len(used) == 6 and victim not in used
        assert json.loads((tmp_path / "wd" / "preflight_excluded.json").read_text())[victim]
        assert len(list((tmp_path / "cache").glob("*.pt"))) == 6  # one cache entry per used study
        # a different subset must not silently resume the same checkpoint
        with pytest.raises(CheckpointIncompatibleError):
            experiment_cli(base[:-1] + ["7", "--epochs-override", "1"])
