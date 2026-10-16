"""Pre-registered Phase 16 decision rules and registry integrity."""

import pytest

from mri_core.weak_label_prereg import evaluate, load_registry, run_name

REG = load_registry()


def fill(real_val=0.64, shuf_val=0.52, real_hold=0.60, shuf_hold=0.51, real_loss=1.0, floor=1.05, clips=(1.0, 5.0), seeds=None):
    out = {}
    for s in seeds or REG["seeds"]:
        for c in clips:
            out[run_name(s, c, "real")] = {"val_final_macro_auc": real_val, "val_final_loss": real_loss, "val_floor_loss": floor,
                                           "holdout_macro_auc": real_hold}
            out[run_name(s, c, "shuf")] = {"val_final_macro_auc": shuf_val, "val_final_loss": floor, "val_floor_loss": floor,
                                           "holdout_macro_auc": shuf_hold}
    return out


def test_registry_is_self_consistent():
    assert REG["seeds"] == [101, 102, 103, 104, 105] and REG["grad_clip_arms"] == [1.0, 5.0] and REG["learning_rate"] == 1e-4
    assert not set(REG["seeds"]) & {0, 1}  # earlier phases used seeds 0 and 1
    assert set(REG["partitions"]) == {str(s) for s in REG["seeds"]}
    assert all(len(p["validation_sha256"]) == 64 for p in REG["partitions"].values())
    assert REG["holdout_pool"]["studies_before_series_and_readability_filters"] == 501 and REG["epochs"] == 12


def test_supported_requires_all_seeds_and_all_criteria():
    v = evaluate(REG, fill())
    assert v["arms"]["clip1"]["verdict"] == "supported" and v["arms"]["clip5"]["verdict"] == "supported"


def test_promising_with_one_failing_seed_and_not_supported_with_two():
    results = fill()
    results[run_name(101, 1.0, "real")]["val_final_macro_auc"] = 0.55  # C1 fails in one seed
    assert evaluate(REG, results)["arms"]["clip1"]["verdict"] == "promising"
    assert evaluate(REG, {**results, run_name(102, 1.0, "real"): {**results[run_name(102, 1.0, "real")], "holdout_macro_auc": 0.50}}
                    )["arms"]["clip1"]["verdict"] == "promising"  # each criterion is counted separately (registered rule)
    results[run_name(102, 1.0, "real")]["val_final_macro_auc"] = 0.55  # C1 fails in a second seed
    verdict = evaluate(REG, results)
    assert verdict["arms"]["clip1"]["verdict"] == "not_supported"
    assert verdict["arms"]["clip5"]["verdict"] == "supported"  # arms are judged separately


def test_loss_criterion_blocks_support():
    assert evaluate(REG, fill(real_loss=1.06))["arms"]["clip1"]["verdict"] == "not_supported"


def test_incomplete_and_invalid_control():
    assert evaluate(REG, fill(seeds=REG["seeds"][:3]))["arms"]["clip1"]["verdict"] == "incomplete"
    bad = evaluate(REG, fill(shuf_val=0.62, shuf_hold=0.62))
    assert bad["arms"]["clip1"]["verdict"] == "invalid_control" and bad["control_alarm"] is True
    assert evaluate(REG, fill())["control_alarm"] is False


def test_clip_comparison_needs_same_sign_everywhere_and_a_margin():
    results = fill()
    for s in REG["seeds"]:
        results[run_name(s, 5.0, "real")]["val_final_macro_auc"] = 0.68
    assert evaluate(REG, results)["clip_comparison"]["different"] is True
    results[run_name(101, 5.0, "real")]["val_final_macro_auc"] = 0.60
    assert evaluate(REG, results)["clip_comparison"]["different"] is False


# ---------------------------------------------------------------- undefined AUC is reported as unavailable, never substituted
import numpy as np  # noqa: E402

from mri_core.weak_label_prereg import final_epoch_record  # noqa: E402
from mri_core.weak_label_train import macro_auc, masked_auc  # noqa: E402


def state_with(aucs_per_epoch, losses=None):
    losses = losses or [1.0] * len(aucs_per_epoch)
    hist = [{"epoch": i + 1, "train_loss": 1.0, "val_loss": l, "val_auc_weak": a} for i, (a, l) in enumerate(zip(aucs_per_epoch, losses))]
    return {"epoch": len(hist), "metrics": {"history": hist}}


NONE3 = {"ACL": None, "Medial Meniscus": None, "Lateral Meniscus": None}


def test_final_epoch_record_with_no_defined_auc_does_not_divide_by_zero():  # the Phase 16 dry-run ZeroDivisionError
    rec = final_epoch_record(state_with([NONE3, NONE3]))
    assert rec["val_final_macro_auc"] is None and rec["val_best_macro_auc"] is None and rec["val_final_auc_defined_targets"] == []


def test_final_epoch_record_uses_only_defined_targets():
    rec = final_epoch_record(state_with([NONE3, {"ACL": 0.7, "Medial Meniscus": None, "Lateral Meniscus": 0.5}], losses=[1.1, 1.0]))
    assert rec["val_final_macro_auc"] == pytest.approx(0.6) and rec["val_final_auc_defined_targets"] == ["ACL", "Lateral Meniscus"]
    assert rec["val_best_epoch"] == 2


def test_macro_auc_empty_and_all_none_are_unavailable_not_a_number():
    assert macro_auc({}) is None and macro_auc(NONE3) is None
    assert macro_auc({"a": 0.0, "b": None}) == 0.0  # a genuine AUC of 0.0 is kept


def test_masked_auc_single_class_fully_masked_and_missing_label_masks():
    labels = np.zeros((6, 12), dtype=np.float32)
    probs = np.random.default_rng(0).random((6, 12)).astype(np.float32)
    masks = np.zeros((6, 12), dtype=np.float32)
    masks[:, 0] = 1  # ACL: every labelled study negative -> single class -> undefined
    labels[:3, 1] = 1
    masks[:, 1] = 1  # MCL: both classes -> defined
    labels[:, 2] = np.array([1, 0, 1, 0, 1, 0])
    masks[[0, 1], 2] = 1  # Medial Meniscus: only studies 0 (pos) and 1 (neg) carry a label; the rest are masked
    labels[:, 3] = 1  # Lateral Meniscus: labels present but fully masked -> undefined
    auc = masked_auc(labels, probs, masks)
    assert auc["ACL"] is None and auc["MCL"] is not None and auc["Medial Meniscus"] in (0.0, 1.0) and auc["Lateral Meniscus"] is None
    changed = probs.copy()
    changed[2:, 2] = 1 - changed[2:, 2]  # predictions on masked (unlabelled) studies cannot change the AUC
    assert masked_auc(labels, changed, masks)["Medial Meniscus"] == auc["Medial Meniscus"]
    masks[:, 2] = 0
    masks[0, 2] = 1  # a single labelled study: no positive-negative pair
    assert masked_auc(labels, probs, masks)["Medial Meniscus"] is None


def test_evaluate_excludes_undefined_pairs_and_never_calls_them_supported():
    results = fill()
    results[run_name(103, 1.0, "real")]["holdout_macro_auc"] = None
    arm = evaluate(REG, results)["arms"]["clip1"]
    assert arm["verdict"] == "incomplete" and arm["seeds_with_undefined_auc"] == [103] and arm["complete_seeds"] == 4
    results[run_name(104, 5.0, "shuf")]["val_final_macro_auc"] = None
    assert evaluate(REG, results)["arms"]["clip5"]["verdict"] == "incomplete"
    results[run_name(101, 5.0, "real")]["val_final_macro_auc"] = None
    assert evaluate(REG, results)["clip_comparison"] == {"incomplete": True}
