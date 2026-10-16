"""Phase 15 training-health tooling: shuffled-label control, constant baselines, leakage report, gradient diagnostics, CLI flags."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from mri_core.weak_label_diagnostics import constant_baselines, leakage_report, shuffle_labels, shuffled_rows
from mri_core.weak_label_train import InMemoryWeakDataset, masked_bce, train_weak_label
from scripts.kaggle_weak_label_experiment import main as cli

from .test_weak_label_real_smoke import build_manifest_for
from .test_weak_label_resolution import world  # noqa: F401  (fixture)
from .test_weak_label_experiment import CFG

pytestmark = pytest.mark.rsna


def labels(n=40, seed=0):
    rng = np.random.default_rng(seed)
    values = np.full((n, 12), np.nan, dtype=np.float32)
    masks = np.zeros((n, 12), dtype=np.float32)
    for j in range(3):
        masks[:, j] = rng.random(n) < 0.8
        values[:, j] = (rng.random(n) < 0.25).astype(np.float32)
    values[masks == 0] = np.nan
    return values, masks


def test_shuffle_is_deterministic_moves_every_row_and_keeps_prevalence():
    values, masks = labels()
    a = shuffle_labels(values, masks, 7)
    b = shuffle_labels(values, masks, 7)
    assert np.array_equal(a[2], b[2]) and not (a[2] == np.arange(len(values))).any()
    assert sorted(a[2]) == list(range(len(values)))
    assert np.array_equal(np.nansum(a[0], axis=0), np.nansum(values, axis=0)) and np.array_equal(a[1].sum(axis=0), masks.sum(axis=0))
    assert not np.array_equal(shuffled_rows(40, 1), shuffled_rows(40, 2))
    for row in range(len(values)):  # values and masks move together: a masked entry stays NaN
        assert np.array_equal(np.isnan(a[0][row]), a[1][row] == 0)


def test_constant_baselines_match_the_training_loss_and_the_weighted_optimum_is_best_on_train():
    values, masks = labels(200)
    out = constant_baselines(values[:150], masks[:150], values[150:], masks[150:], 10.0)
    pw = torch.tensor([out["pos_weight"][t] for t in TARGET_COLUMNS])
    logits = torch.zeros(50, 12)
    manual = float(masked_bce(logits, torch.from_numpy(np.nan_to_num(values[150:])), torch.from_numpy(masks[150:]), pw, torch.ones(12)))
    assert out["val_loss_p05"] == pytest.approx(manual, rel=1e-5)
    assert out["train_loss_weighted_optimum"] <= min(out["train_loss_p05"], out["train_loss_train_prevalence"]) + 1e-6
    assert out["constant_auc"] == 0.5 and set(out["train_prevalence"]) == set(TARGET_COLUMNS[:3])


def test_leakage_report():
    assert leakage_report(["a", "b"], ["c"], ["g"], ["t"])["clean"]
    assert not leakage_report(["a", "b"], ["b"])["clean"]
    assert leakage_report(["a", "b"], ["a", "b"], allow_val_equals_train=True)["clean"]
    assert not leakage_report(["a", "g"], ["c"], ["g"])["clean"] and not leakage_report(["a", "a"], ["c"])["clean"]
    assert not leakage_report(["a"], ["t"], [], ["t"])["clean"]


def test_epoch_records_carry_gradient_and_prediction_diagnostics(tmp_path):
    n = 12
    values, masks = labels(n)
    images = torch.rand(n, 3, 3, 1, 16, 16)
    ds = InMemoryWeakDataset([f"s{i}" for i in range(n)], images, torch.ones(n, 3, 3, dtype=torch.bool), values, masks)
    cfg = {**CFG, "arm": "A", "epochs": 2, "image_size": 16, "slices_per_series": 3, "num_workers": 0, "batch_size": 4, "device": "cpu"}
    result = train_weak_label(ds, ds, cfg, tmp_path, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids)
    record = result["history"][-1]
    assert record["grad_norm_mean"] > 0 and record["grad_norm_max"] >= record["grad_norm_mean"]
    assert 0.0 <= record["grad_clipped_fraction"] <= 1.0 and record["nonfinite_grad_steps"] == 0
    assert set(record["val_prob_std"]) == set(TARGET_COLUMNS[:3]) and all(0 <= v <= 1 for v in record["val_prob_mean"].values())


def test_cli_shuffled_and_overfit_runs(world):  # noqa: F811
    manifest = build_manifest_for(SimpleNamespace(meta=world.meta, tmp=world.tmp))
    cfg = world.tmp / "cfg.json"
    cfg.write_text(json.dumps({**CFG, "epochs": 1, "val_fraction": 0.34, "num_workers": 0, "arm": "C", "image_size": 16,
                               "slices_per_series": 3, "cache_dir": str(world.tmp / "cache")}))
    common = ["--data-dir", str(world.meta), "--dicom-root", str(world.raw), "--manifest", str(manifest), "--config", str(cfg)]
    fingerprints = {}
    for name, extra in (("plain", []), ("shuffled", ["--shuffle-train-labels", "3"]), ("overfit", ["--val-equals-train"])):
        work = world.tmp / name
        assert cli(["train", *common, "--work-dir", str(work), *extra]) == 0
        state = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
        fingerprints[name] = state["dataset_fingerprint"]
        leak = json.loads((work / "leakage.json").read_text())
        assert leak["clean"] and leak["val_equals_train_by_design"] == (name == "overfit")
        assert "val_loss_p05" in json.loads((work / "baselines.json").read_text())
        assert (name == "overfit") == (sorted(state["train_study_ids"]) == sorted(state["valid_study_ids"]))
    assert len(set(fingerprints.values())) == 3  # a control can never resume or be mistaken for the real run
