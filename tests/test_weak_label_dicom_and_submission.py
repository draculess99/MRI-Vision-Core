"""Synthetic-DICOM tests: missing/corrupt series handling and the offline, schema-exact submission workflow."""

import socket
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from mri_core.rsna_knee_dataset import SERIES_COLUMNS, SUBMISSION_COLUMNS, TARGET_COLUMNS, load_rsna_metadata
from mri_core.weak_label_data import WeakLabelDicomDataset, hash_split, preflight_series
from mri_core.weak_label_models import build_model
from mri_core.weak_label_submit import SubmissionError, combine, run_submission, validate_submission
from mri_core.weak_label_train import DEFAULT_CONFIG, TrainingAborted, train_weak_label
from scripts.kaggle_weak_label_experiment import main as experiment_cli

from .dicom_factory import AXIAL, CORONAL, SAGITTAL, write_series

pytestmark = pytest.mark.rsna

T = len(TARGET_COLUMNS)
PLANES = (("Axial", AXIAL), ("Coronal", CORONAL), ("Sagittal", SAGITTAL))
CFG = {**DEFAULT_CONFIG, "epochs": 1, "batch_size": 2, "num_workers": 0, "device": "cpu", "image_size": 8,
       "slices_per_series": 2, "val_fraction": 0.3, "learning_rate": 1e-3}


def write_study(raw, split, uid, rows):
    for plane, orientation in PLANES:
        series_uid = f"{uid}.s.{plane}"
        rows.append((uid, series_uid, 1, 0, plane))
        write_series(raw / f"{split}_series" / uid / series_uid, count=2, orientation=orientation,
                     series_uid=series_uid, study_uid=uid, rows=8, columns=8)


@pytest.fixture
def world(tmp_path):
    meta, raw = tmp_path / "meta", tmp_path / "raw"
    meta.mkdir()
    train_ids = [f"1.2.826.{i}" for i in range(1, 9)]
    test_ids = ["1.2.826.901", "1.2.826.902", "1.2.826.903", "1.2.826.904"]
    train_rows, test_rows = [], []
    for uid in train_ids:
        write_study(raw, "train", uid, train_rows)
    for uid in test_ids:
        write_study(raw, "test", uid, test_rows)
    labels = [{"StudyInstanceUID": uid, "Report": "r", **{t: np.nan for t in TARGET_COLUMNS}} for uid in train_ids]
    pd.DataFrame(labels).to_csv(meta / "train.csv", index=False)
    pd.DataFrame(train_rows, columns=SERIES_COLUMNS).to_csv(meta / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": test_ids}).to_csv(meta / "test.csv", index=False)
    pd.DataFrame(test_rows, columns=SERIES_COLUMNS).to_csv(meta / "test_series.csv", index=False)
    sample = pd.DataFrame({"StudyInstanceUID": test_ids, **{t: 0.5 for t in TARGET_COLUMNS}})
    sample.to_csv(meta / "sample_submission.csv", index=False)
    return SimpleNamespace(meta=meta, raw=raw, train_ids=train_ids, test_ids=test_ids,
                           metadata=load_rsna_metadata(meta, raw), tmp=tmp_path)


def values_masks(n):
    rng = np.random.default_rng(0)
    values = rng.integers(0, 2, (n, T)).astype(np.float32)
    masks = np.zeros((n, T), dtype=np.float32)
    masks[:, :3] = 1
    values[masks == 0] = np.nan
    return values, masks


class TestMissingAndCorruptSeries:
    def test_preflight_reports_missing_plane_directory_without_reading_pixels(self, world):
        import shutil
        victim = world.train_ids[2]
        shutil.rmtree(world.raw / "train_series" / victim / f"{victim}.s.Coronal")
        problems = preflight_series(world.metadata, world.train_ids)
        assert list(problems) == [victim] and "Coronal series directory missing" in problems[victim]

    def test_empty_series_directory_is_a_problem(self, world):
        victim = world.train_ids[0]
        for f in (world.raw / "train_series" / victim / f"{victim}.s.Axial").iterdir():
            f.unlink()
        assert "Axial series directory empty" in preflight_series(world.metadata, [victim])[victim]

    def test_corrupt_dicom_raises_by_default_and_is_masked_when_skipping(self, world):
        victim = world.train_ids[1]
        for f in (world.raw / "train_series" / victim / f"{victim}.s.Sagittal").iterdir():
            f.write_bytes(b"not a dicom file")
        values, masks = values_masks(len(world.train_ids))
        strict = WeakLabelDicomDataset(world.metadata, world.train_ids, values, masks, CFG, on_error="raise")
        with pytest.raises(Exception):
            strict[1]
        lenient = WeakLabelDicomDataset(world.metadata, world.train_ids, values, masks, CFG, on_error="skip")
        bad, good = lenient[1], lenient[0]
        assert not bool(bad["valid"]) and float(bad["target_mask"].sum()) == 0 and bool(bad["mask"].any(dim=1).all())
        assert bool(good["valid"]) and float(good["target_mask"].sum()) == 3
        model = build_model("A", CFG)
        assert model(bad["images"][None], bad["mask"][None]).shape == (1, T)  # a skipped sample still runs

    def test_training_survives_a_few_bad_studies_but_aborts_on_many(self, world):
        values, masks = values_masks(len(world.train_ids))
        bad = world.train_ids[:2]
        for uid in bad:
            for f in (world.raw / "train_series" / uid / f"{uid}.s.Axial").iterdir():
                f.write_bytes(b"corrupt")
        tr, va = hash_split(world.train_ids, 0.3, 0)
        mk = lambda idx: WeakLabelDicomDataset(world.metadata, [world.train_ids[i] for i in idx], values[idx], masks[idx],
                                               CFG, on_error="skip")
        ids = lambda idx: [world.train_ids[i] for i in idx]
        with pytest.raises(TrainingAborted):
            train_weak_label(mk(tr), mk(va), {**CFG, "max_invalid_fraction": 0.05}, world.tmp / "abort",
                             fingerprint="f", train_ids=ids(tr), val_ids=ids(va))
        ok = train_weak_label(mk(tr), mk(va), {**CFG, "max_invalid_fraction": 0.9}, world.tmp / "ok",
                              fingerprint="f", train_ids=ids(tr), val_ids=ids(va))
        assert ok["final_epoch"] == 1 and ok["invalid_samples"] >= 1

    def test_preflight_cli_counts_trainable_targets_and_flags_problems(self, world, capsys):
        from mri_core.weak_label_manifest import build_manifest, write_manifest
        from mri_core.report_label_rules import RuleBasedExtractor
        en = "Findings: The study of the knee was performed and the following is noted in this report.\n"
        train = pd.read_csv(world.meta / "train.csv", dtype={"StudyInstanceUID": str})
        train["Report"] = en + "Complete tear of the anterior cruciate ligament."
        rows, summary = build_manifest(train, RuleBasedExtractor(), active_targets=["ACL"])
        manifest, _ = write_manifest(rows, summary, world.tmp / "weak.jsonl")
        code = experiment_cli(["preflight", "--data-dir", str(world.meta), "--dicom-root", str(world.raw),
                               "--manifest", str(manifest), "--work-dir", str(world.tmp / "wd"), "--arm", "A"])
        out = capsys.readouterr().out
        assert code == 0 and '"ACL": 8' in out and '"studies_with_series_problems": 0' in out


def save_baseline(path, seed):
    torch.manual_seed(seed)
    model = build_model("A", CFG)
    torch.save({"model": model.state_dict(), "config": {**CFG, "arm": "A"}, "target_columns": list(TARGET_COLUMNS)}, path)


def save_new(path, seed):
    torch.manual_seed(seed)
    config = {**CFG, "arm": "B"}
    model = build_model("B", config)
    torch.save({"model": model.state_dict(), "config": config, "target_columns": list(TARGET_COLUMNS)}, path)


class TestSubmission:
    def test_new_model_only_changes_the_chosen_targets_and_schema_is_exact(self, world):
        save_baseline(world.tmp / "b1.pt", 1)
        save_baseline(world.tmp / "b2.pt", 2)
        save_new(world.tmp / "new.pt", 3)
        base = run_submission(world.metadata, [world.tmp / "b1.pt", world.tmp / "b2.pt"], None, [], world.tmp / "base.csv")
        new = run_submission(world.metadata, [world.tmp / "b1.pt", world.tmp / "b2.pt"], world.tmp / "new.pt",
                             ["ACL", "Medial Meniscus", "Lateral Meniscus"], world.tmp / "new.csv")
        a, b = pd.read_csv(world.tmp / "base.csv"), pd.read_csv(world.tmp / "new.csv")
        assert tuple(b.columns) == SUBMISSION_COLUMNS and list(b.StudyInstanceUID) == world.test_ids
        keep = [t for t in TARGET_COLUMNS if t not in ("ACL", "Medial Meniscus", "Lateral Meniscus")]
        assert np.allclose(a[keep].to_numpy(), b[keep].to_numpy())  # baseline retained exactly
        assert not np.allclose(a["ACL"], b["ACL"])  # the chosen targets come from the new model
        values = b[list(TARGET_COLUMNS)].to_numpy()
        assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
        assert new["failed_studies"] == 0 and base["targets_from_new_model"] == []

    def test_inference_needs_no_network(self, world, monkeypatch):
        save_baseline(world.tmp / "b.pt", 1)

        def blocked(*a, **k):
            raise AssertionError("network access attempted")
        monkeypatch.setattr(socket.socket, "connect", blocked)
        monkeypatch.setattr(socket, "getaddrinfo", blocked)
        assert run_submission(world.metadata, [world.tmp / "b.pt"], None, [], world.tmp / "o.csv")["studies"] == 4

    def test_an_unreadable_test_study_gets_a_neutral_value_within_the_limit(self, world):
        save_baseline(world.tmp / "b.pt", 1)
        victim = world.test_ids[0]
        for f in (world.raw / "test_series" / victim / f"{victim}.s.Axial").iterdir():
            f.write_bytes(b"corrupt")
        with pytest.raises(SubmissionError):  # 1 of 4 studies is far above the default 5% limit
            run_submission(world.metadata, [world.tmp / "b.pt"], None, [], world.tmp / "o.csv")
        result = run_submission(world.metadata, [world.tmp / "b.pt"], None, [], world.tmp / "o.csv", max_failed_fraction=0.5)
        frame = pd.read_csv(world.tmp / "o.csv")
        others = frame[frame.StudyInstanceUID != victim][list(TARGET_COLUMNS)].mean().to_numpy()
        assert result["failed_studies"] == 1
        assert np.allclose(frame[frame.StudyInstanceUID == victim][list(TARGET_COLUMNS)].to_numpy()[0], others)

    def test_wrong_target_order_checkpoint_and_existing_output_are_refused(self, world):
        save_baseline(world.tmp / "b.pt", 1)
        state = torch.load(world.tmp / "b.pt", weights_only=True)
        state["target_columns"] = list(reversed(TARGET_COLUMNS))
        torch.save(state, world.tmp / "wrong.pt")
        with pytest.raises(SubmissionError):
            run_submission(world.metadata, [world.tmp / "wrong.pt"], None, [], world.tmp / "o.csv")
        run_submission(world.metadata, [world.tmp / "b.pt"], None, [], world.tmp / "o.csv")
        with pytest.raises(SubmissionError):
            run_submission(world.metadata, [world.tmp / "b.pt"], None, [], world.tmp / "o.csv")

    def test_combine_blend_and_validation_rules(self):
        base, new = np.full((2, T), 0.2), np.full((2, T), 0.8)
        out = combine(base, new, ["ACL"], 0.5)
        assert out[0, 0] == pytest.approx(0.5) and out[0, 1] == 0.2
        assert np.array_equal(combine(base, None, ["ACL"]), base)
        with pytest.raises(SubmissionError):
            combine(base, new, ["Nope"])
        with pytest.raises(SubmissionError):
            combine(base, new, ["ACL"], 1.5)
        template = pd.DataFrame({"StudyInstanceUID": ["a"], **{t: 0.5 for t in TARGET_COLUMNS}})
        good = template.copy()
        validate_submission(good, template)
        for broken in (good[["StudyInstanceUID", *reversed(TARGET_COLUMNS)]], good.assign(ACL=1.5), good.assign(ACL=np.nan),
                       good.assign(StudyInstanceUID="b")):
            with pytest.raises(SubmissionError):
                validate_submission(broken, template)


class TestDecodeCache:
    def test_cached_samples_are_identical_to_first_decode_and_skip_decoding(self, world, monkeypatch):
        values, masks = values_masks(len(world.train_ids))
        ds = WeakLabelDicomDataset(world.metadata, world.train_ids, values, masks, CFG, cache_dir=world.tmp / "cache")
        first = ds[0]
        assert len(list((world.tmp / "cache").glob("*.pt"))) == 1
        from mri_core.rsna_knee_train import RSNAKneeDicomDataset

        def boom(self, index):
            raise AssertionError("decoded again despite the cache")
        monkeypatch.setattr(RSNAKneeDicomDataset, "__getitem__", boom)
        second = ds[0]
        assert torch.equal(first["images"], second["images"]) and torch.equal(first["mask"], second["mask"])
        assert torch.equal(first["target_mask"], second["target_mask"])
