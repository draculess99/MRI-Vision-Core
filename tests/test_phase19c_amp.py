"""Phase 19C: AMP overflow handling, skipped updates, gradient accumulation and resumable update accounting in the shared weak-label trainer."""
import json

import numpy as np
import pytest
import torch

from mri_core import weak_label_train
from mri_core.weak_label_models import build_model
from mri_core.weak_label_train import TrainingAborted, train_weak_label

from .test_convnext_roi import run_cfg, toy_data


def poisoned_factory(cfg, poison):
    """ConvNeXt arm D whose first parameter receives an infinite gradient on the given (0-based) backward calls: a deterministic stand-in for an fp16 overflow."""
    calls = []

    def factory():
        model = build_model("D", cfg)
        first = next(model.parameters())

        def hook(grad):
            calls.append(1)
            return grad * float("inf") if poison == "all" or (len(calls) - 1) in poison else grad

        first.register_hook(hook)
        return model
    return factory


def rows(path):
    return [json.loads(line) for line in (path / "update_log.jsonl").read_text().splitlines()]


def adam_steps(path):
    state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)["optimizer"]["state"]
    return sorted({int(v["step"]) for v in state.values()})


def test_a_non_finite_gradient_never_reaches_the_optimizer_without_amp(tmp_path):
    ds = toy_data(8)
    cfg = {**run_cfg("D"), "epochs": 1, "batch_size": 4}  # 2 updates per epoch
    out = train_weak_label(ds, ds, cfg, tmp_path, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids, model_factory=poisoned_factory(cfg, {0}))
    r = rows(tmp_path)
    assert [x["skipped"] for x in r] == [True, False] and r[0]["grad_norm"] is None and r[0]["nonfinite_tensors"] >= 1 and r[1]["finite"]
    assert adam_steps(tmp_path) == [1]  # exactly one real optimizer update; Adam moments never saw the poisoned gradient
    latest = out["history"][-1]
    assert (latest["updates_applied"], latest["updates_skipped"], latest["global_step_updates"]) == (1, 1, 2) and latest["nonfinite_grad_steps"] == 1
    state = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=True)
    assert all(torch.isfinite(v).all() for v in state["model"].values() if v.is_floating_point())
    assert state["global_step"] == 2 and state["metrics"]["counters"] == {"updates_applied": 1, "updates_skipped": 1, "nonfinite_tensors": r[0]["nonfinite_tensors"]}


def test_amp_overflow_lowers_the_scale_skips_the_step_and_logs_scale_before_and_after(tmp_path, monkeypatch):
    monkeypatch.setattr(weak_label_train, "AMP_DEVICE_TYPES", ("cuda", "cpu"))
    ds = toy_data(8)
    cfg = {**run_cfg("D"), "epochs": 1, "batch_size": 2, "amp": True, "amp_init_scale": 1024.0}  # 4 updates
    train_weak_label(ds, ds, cfg, tmp_path, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids, model_factory=poisoned_factory(cfg, {0, 1}))
    r = rows(tmp_path)
    assert r[0]["scale_before"] == 1024.0 and r[0]["scale_after"] == 512.0 and r[0]["skipped"]  # first overflow halves the safer initial scale
    assert r[1]["scale_before"] == 512.0 and r[1]["scale_after"] == 256.0 and r[1]["skipped"]
    assert all(x["finite"] and x["scale_after"] == x["scale_before"] for x in r[2:])
    assert adam_steps(tmp_path) == [2]  # four attempts, two skipped, two applied
    state = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=True)
    assert state["scaler"]["scale"] == 256.0 and state["global_step"] == 4


def test_persistent_non_finite_gradients_abort_before_any_checkpoint_is_written(tmp_path):
    ds = toy_data(8)
    cfg = {**run_cfg("D"), "epochs": 2, "batch_size": 2, "max_consecutive_skipped_updates": 3}
    with pytest.raises(TrainingAborted, match="consecutive optimizer updates were skipped"):
        train_weak_label(ds, ds, cfg, tmp_path, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids, model_factory=poisoned_factory(cfg, "all"))
    assert not (tmp_path / "last.pt").exists()


def test_scale_collapse_is_treated_as_instability_not_start_up_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(weak_label_train, "AMP_DEVICE_TYPES", ("cuda", "cpu"))
    ds = toy_data(8)
    cfg = {**run_cfg("D"), "epochs": 1, "batch_size": 2, "amp": True, "amp_init_scale": 2.0, "amp_min_scale": 1.0, "max_consecutive_skipped_updates": 99}
    with pytest.raises(TrainingAborted, match="GradScaler scale fell"):
        train_weak_label(ds, ds, cfg, tmp_path, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids, model_factory=poisoned_factory(cfg, "all"))


def test_gradient_accumulation_counts_real_updates_and_handles_a_short_final_group(tmp_path):
    ds = toy_data(6)
    cfg = {**run_cfg("D"), "epochs": 2, "batch_size": 2, "grad_accum_steps": 2}  # 3 batches per epoch -> groups of 2 and 1
    out = train_weak_label(ds, ds, cfg, tmp_path, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids)
    r = rows(tmp_path)
    assert [x["micro_batches"] for x in r] == [2, 1, 2, 1] and [x["update"] for x in r] == [1, 2, 3, 4]
    assert all(x["finite"] for x in r) and adam_steps(tmp_path) == [4]
    state = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=True)
    assert state["global_step"] == 4 and state["scheduler"]["last_epoch"] == 2  # global step = optimizer updates; the epoch-based schedule advanced once per epoch
    assert out["history"][-1]["grad_accum_steps"] == 2 and out["history"][-1]["optimizer_updates_total"] == 4
    assert all(abs(x["lr"]) > 0 for x in r)


def test_accumulated_gradient_equals_the_mean_of_the_micro_batch_gradients():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    x, y = torch.randn(4, 4), torch.randn(4, 1)
    reference = torch.autograd.grad(sum(torch.nn.functional.mse_loss(model(x[i:i + 2]), y[i:i + 2]) for i in (0, 2)) / 2, list(model.parameters()))
    for p in model.parameters():
        p.grad = None
    for i in (0, 2):  # exactly the scaling train_weak_label applies: loss / group
        (torch.nn.functional.mse_loss(model(x[i:i + 2]), y[i:i + 2]) / 2).backward()
    assert all(torch.allclose(p.grad, g, atol=1e-6) for p, g in zip(model.parameters(), reference))


def test_resume_with_amp_and_accumulation_matches_an_uninterrupted_run_and_keeps_counters_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(weak_label_train, "AMP_DEVICE_TYPES", ("cuda", "cpu"))
    ds = toy_data(8)
    cfg = {**run_cfg("D"), "epochs": 3, "batch_size": 2, "grad_accum_steps": 2, "amp": True, "amp_init_scale": 1024.0, "num_workers": 0}
    full = tmp_path / "full"
    train_weak_label(ds, ds, cfg, full, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids, model_factory=poisoned_factory(cfg, {0}))
    part = tmp_path / "part"
    first = train_weak_label(ds, ds, cfg, part, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids, model_factory=poisoned_factory(cfg, {0}), stop_after_epochs=1)
    assert first["status"] == "paused_epoch_limit"
    second = train_weak_label(ds, ds, cfg, part, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids, model_factory=poisoned_factory(cfg, set()))
    assert second["resumed_from_epoch"] == 1 and second["final_epoch"] == 3
    a, b = (torch.load(d / "last.pt", map_location="cpu", weights_only=True) for d in (full, part))
    assert a["global_step"] == b["global_step"] == 6 and a["metrics"]["counters"] == b["metrics"]["counters"] and a["scaler"]["scale"] == b["scaler"]["scale"]
    assert a["scheduler"]["last_epoch"] == b["scheduler"]["last_epoch"] == 3 and a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert [(x["update"], x["skipped"]) for x in rows(full)] == [(x["update"], x["skipped"]) for x in rows(part)] and len(rows(part)) == 6  # no duplicated or lost rows
    assert all(k in b for k in ("torch_rng", "python_rng", "numpy_rng", "scheduler", "scaler", "optimizer", "loader_rng"))


def test_resuming_with_a_different_accumulation_is_refused_and_a_truncated_update_log_is_trimmed(tmp_path):
    from mri_core.rsna_checkpoint import CheckpointIncompatibleError
    ds = toy_data(8)
    cfg = {**run_cfg("D"), "epochs": 2, "batch_size": 2, "grad_accum_steps": 2}
    train_weak_label(ds, ds, cfg, tmp_path, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids, stop_after_epochs=1)
    with (tmp_path / "update_log.jsonl").open("a") as handle:  # rows of an epoch whose checkpoint was lost
        handle.write(json.dumps({"epoch": 2, "update": 3}) + "\n")
    with pytest.raises(CheckpointIncompatibleError):
        train_weak_label(ds, ds, {**cfg, "grad_accum_steps": 1}, tmp_path, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids)
    train_weak_label(ds, ds, cfg, tmp_path, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids)
    assert [x["update"] for x in rows(tmp_path)] == [1, 2, 3, 4]
    assert np.isfinite([x["lr"] for x in rows(tmp_path)]).all()


def test_update_log_analysis_flags_late_skips_and_inconsistent_accounting(tmp_path):
    from mri_core.weak_label_train import analyze_update_log

    def row(u, skipped, before, after, applied):
        return {"epoch": 1, "update": u, "finite": not skipped, "skipped": skipped, "scale_before": before, "scale_after": after, "grad_norm": None if skipped else 3.0, "applied_total": applied}

    good = [row(1, True, 1024.0, 512.0, 0), row(2, False, 512.0, 512.0, 1), row(3, False, 512.0, 512.0, 2)]
    (tmp_path / "update_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in good))
    a = analyze_update_log(tmp_path)
    assert a["stable"] and a["skipped"] == 1 and a["applied"] == 2 and a["scale_min"] == 512.0 and a["consistent"] and a["scale_changes"] == [(1, 1024.0, 512.0)]
    late = good + [row(u, False, 512.0, 512.0, u - 1) for u in range(4, 25)] + [row(25, True, 512.0, 256.0, 23)]
    (tmp_path / "update_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in late))
    assert not analyze_update_log(tmp_path)["stable"] and analyze_update_log(tmp_path)["skips_after_calibration"] == 1
    (tmp_path / "update_log.jsonl").write_text(json.dumps(good[0]) + "\n" + json.dumps(good[2]) + "\n")  # a lost row
    assert not analyze_update_log(tmp_path)["consistent"]
