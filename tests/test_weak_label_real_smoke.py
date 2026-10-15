"""Real-data smoke-test machinery: stratified deterministic subset, real-data pre-flight, stop conditions, throughput stats."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from mri_core.weak_label_data import select_smoke_subset
from mri_core.weak_label_train import TrainingAborted, train_weak_label
from scripts.kaggle_weak_label_experiment import main as cli

from .test_weak_label_dicom_and_submission import CFG as DICOM_CFG
from .test_weak_label_dicom_and_submission import world  # noqa: F401
from .test_weak_label_experiment import CFG, make_dataset

pytestmark = pytest.mark.rsna

T = len(TARGET_COLUMNS)
TIER1 = ("ACL", "Medial Meniscus", "Lateral Meniscus")


def manifest_arrays(n=300, seed=0, rare=0.1):
    rng = np.random.default_rng(seed)
    values = np.zeros((n, T), dtype=np.float32)
    masks = np.zeros((n, T), dtype=np.float32)
    for j in (0, 2, 3):
        masks[:, j] = rng.random(n) < 0.8
        values[:, j] = (rng.random(n) < (rare if j == 0 else 0.4)).astype(np.float32)
    values[masks == 0] = np.nan
    return [f"1.2.826.{i}" for i in range(n)], values, masks


class TestSmokeSubset:
    def test_is_deterministic_study_level_and_never_exceeds_n(self):
        ids, values, masks = manifest_arrays()
        a = select_smoke_subset(ids, values, masks, 64)
        b = select_smoke_subset(ids[::-1], values[::-1], masks[::-1], 64)
        assert a["indices"] == sorted(set(a["indices"])) and len(a["indices"]) == 64
        assert {ids[i] for i in a["indices"]} == {ids[::-1][i] for i in b["indices"]}
        assert select_smoke_subset(ids, values, masks, 64, seed=1)["indices"] != a["indices"]

    def test_every_study_has_a_decided_tier1_target_and_both_classes_are_present(self):
        ids, values, masks = manifest_arrays()
        masks[5, [0, 2, 3]] = 0  # a study with nothing decided must never be picked
        values[5, [0, 2, 3]] = np.nan
        sel = select_smoke_subset(ids, values, masks, 64, min_per_class=8)
        assert 5 not in sel["indices"]
        assert all(masks[i, [0, 2, 3]].sum() > 0 for i in sel["indices"])
        for target, counts in sel["class_balance"].items():
            assert counts["positive"] >= 8 and counts["negative"] >= 8, (target, counts)
        assert sel["unmet_quota"] == {}

    def test_infeasible_quotas_are_reported_not_hidden(self):
        ids, values, masks = manifest_arrays(rare=0.0)  # ACL never positive
        sel = select_smoke_subset(ids, values, masks, 32, min_per_class=8)
        assert sel["unmet_quota"].get("ACL:positive") == 8 and sel["class_balance"]["ACL"]["positive"] == 0


class TestStopConditions:
    def test_non_finite_loss_stops_before_a_checkpoint_is_written(self, tmp_path):
        train, val = make_dataset(24, 0), make_dataset(8, 1, start=100)
        train.images[0] = float("nan")  # one poisoned study
        with pytest.raises(TrainingAborted, match="non-finite"):
            train_weak_label(train, val, {**CFG, "epochs": 2, "shuffle_seed": 0}, tmp_path / "bad", fingerprint="fp",
                             train_ids=train.ids, val_ids=val.ids)
        assert not (tmp_path / "bad" / "last.pt").exists()

    def test_epoch_record_and_log_carry_image_counts_and_data_wait(self, tmp_path):
        train, val = make_dataset(24, 0), make_dataset(8, 1, start=100)
        result = train_weak_label(train, val, {**CFG, "epochs": 1}, tmp_path / "ok", fingerprint="fp",
                                  train_ids=train.ids, val_ids=val.ids)
        record = result["history"][0]
        assert record["train_images"] == int(train.mask.sum()) and record["data_wait_seconds"] >= 0
        log = [json.loads(l) for l in (tmp_path / "ok" / "runtime_log.jsonl").read_text().splitlines()]
        epoch = next(r for r in log if r["event"] == "epoch")
        assert epoch["train_images"] == record["train_images"] and "data_wait_seconds" in epoch


def build_manifest_for(world):
    from mri_core.report_label_rules import RuleBasedExtractor
    from mri_core.weak_label_manifest import build_manifest, write_manifest
    en = "Findings: The study of the knee was performed and the following is noted in this report.\n"
    pos = "Complete tear of the anterior cruciate ligament. Tear of the medial meniscus. The lateral meniscus is intact."
    neg = "No tear of the anterior cruciate ligament. The medial meniscus is intact. Tear of the lateral meniscus."
    train = pd.read_csv(world.meta / "train.csv", dtype={"StudyInstanceUID": str})
    train["Report"] = [en + (pos if i % 2 == 0 else neg) for i in range(len(train))]
    rows, summary = build_manifest(train, RuleBasedExtractor(), active_targets=list(TIER1))
    return write_manifest(rows, summary, world.tmp / "weak.jsonl")[0]


class TestRealPreflightAndSmokeTrain:
    def args(self, world, manifest, cfg, *extra):
        return ["--data-dir", str(world.meta), "--dicom-root", str(world.raw), "--manifest", str(manifest),
                "--config", str(cfg), "--work-dir", str(world.tmp / "wd"), *extra]

    def config(self, world, **kw):
        path = world.tmp / "cfg.json"
        path.write_text(json.dumps({**DICOM_CFG, "epochs": 1, "val_fraction": 0.25, "cache_dir": str(world.tmp / "cache"), **kw}))
        return path

    def test_preflight_validates_tensors_and_reports_latency_cache_and_balance(self, world, capsys):
        manifest = build_manifest_for(world)
        code = cli(["real-preflight", *self.args(world, manifest, self.config(world), "--smoke-subset", "6")])
        report = json.loads((world.tmp / "wd" / "real_preflight.json").read_text())
        assert code == 0 and report["passed"] and report["studies_selected"] == 6 and report["decoded_ok"] == 6
        assert report["train"] + report["validation"] == 6 and report["gold_or_test_overlap"] == 0
        assert report["tensor_shape"] == [3, 2, 1, 8, 8] and report["decode_seconds_per_study"]["mean"] > 0
        assert report["cache_bytes"] > 0 and report["free_disk_gb"] > 0
        for counts in report["class_balance"].values():
            assert counts["positive"] > 0 and counts["negative"] > 0

    def test_preflight_fails_and_names_the_corrupt_study_so_training_must_not_start(self, world):
        manifest = build_manifest_for(world)
        victim = world.train_ids[0]
        for f in (world.raw / "train_series" / victim / f"{victim}.s.Axial").iterdir():
            f.write_bytes(b"corrupt")
        code = cli(["real-preflight", *self.args(world, manifest, self.config(world), "--smoke-subset", "8")])
        report = json.loads((world.tmp / "wd" / "real_preflight.json").read_text())
        assert code == 1 and not report["passed"] and len(report["problems"]) == 1
        assert report["problems"][0]["study"] == "..." + victim[-6:]

    def test_smoke_train_one_epoch_writes_selection_checkpoint_and_images_processed(self, world):
        manifest = build_manifest_for(world)
        cfg = self.config(world)
        assert cli(["train", *self.args(world, manifest, cfg, "--smoke-subset", "6", "--arm", "A")]) == 0
        wd = world.tmp / "wd"
        selection = json.loads((wd / "smoke_selection.json").read_text())
        state = torch.load(wd / "last.pt", map_location="cpu", weights_only=True)
        assert len(selection["studies"]) == 6 and state["epoch"] == 1
        assert sorted(state["train_study_ids"] + state["valid_study_ids"]) == sorted(selection["studies"])
        assert state["metrics"]["latest"]["train_images"] > 0
        assert cli(["export", "--work-dir", str(wd), "--export-dir", str(world.tmp / "exp")]) == 0
