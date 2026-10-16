"""Pre-registered Phase 16 decision rules and registry integrity."""

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
