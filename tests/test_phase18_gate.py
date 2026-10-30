"""Phase 18 submission gate: metric, paired bootstrap, gate logic and schema validation."""

import numpy as np
import pandas as pd
import pytest

from mri_core.phase18_gate import evaluate_gate, macro_auc, paired_bootstrap, per_target_auc, validate_submission_frame
from mri_core.rsna_knee_dataset import SUBMISSION_COLUMNS, TARGET_COLUMNS


def data(n=58, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 12)) < 0.4).astype(int)
    y[:2] = 1
    y[2:4] = 0
    return y, rng


def test_macro_auc_matches_the_mean_of_per_target_aucs_and_perfect_is_one():
    y, rng = data()
    assert macro_auc(y, y.astype(float)) == 1.0
    p = rng.random(y.shape)
    pt = per_target_auc(y, p)
    assert macro_auc(y, p) == pytest.approx(np.mean(list(pt.values())))


def test_paired_bootstrap_separates_a_real_improvement_from_noise():
    y, rng = data(seed=1)
    good = y + rng.normal(scale=0.8, size=y.shape)
    noise_a, noise_b = rng.random(y.shape), rng.random(y.shape)
    real = paired_bootstrap(y, good, noise_a, rounds=300)
    null = paired_bootstrap(y, noise_a, noise_b, rounds=300)
    assert real["ci95"][0] > 0 and real["point"] > 0.2
    assert null["ci95"][0] < 0 < null["ci95"][1]


def gate(y, cand, base, seeds, integrity=None, remaining=5):
    return evaluate_gate(y, cand, base, seeds, integrity or {"no_gold_in_training": True, "resume_ok": True}, submissions_remaining=remaining, rounds=200)


def test_gate_passes_only_when_every_condition_holds():
    y, rng = data(seed=2)
    good = y + rng.normal(scale=0.7, size=y.shape)
    good2 = y + rng.normal(scale=0.7, size=y.shape)
    base = rng.random(y.shape)
    ok = gate(y, (good + good2) / 2, base, {"s1": good, "s2": good2})
    assert ok["gate_passed"] and all(ok["checks"].values())
    assert not gate(y, (good + good2) / 2, base, {"s1": good, "s2": rng.random(y.shape)})["gate_passed"]  # a seed that fails to reproduce
    assert not gate(y, (good + good2) / 2, base, {"s1": good, "s2": good2}, integrity={"no_gold_in_training": False})["gate_passed"]
    assert not gate(y, (good + good2) / 2, base, {"s1": good, "s2": good2}, remaining=0)["checks"]["G5_submissions_remaining"]
    weak = base + 0.15 * (y - 0.5)  # slightly better than the baseline but nowhere near 0.58
    result = gate(y, weak, base, {"s1": weak})
    assert result["candidate_macro_auc"] < 0.80 and (not result["checks"]["G1_paired_ci_lower_bound_above_zero"] or not result["gate_passed"] or result["candidate_macro_auc"] >= 0.58)


def test_a_candidate_equal_to_the_baseline_fails_g1_and_g2():
    y, rng = data(seed=3)
    base = rng.random(y.shape)
    r = gate(y, base, base, {"s1": base})
    assert not r["checks"]["G1_paired_ci_lower_bound_above_zero"] and not r["checks"]["G2_every_seed_beats_baseline"] and not r["gate_passed"]


def frame(uids):
    return pd.DataFrame({"StudyInstanceUID": uids, **{t: np.full(len(uids), 0.5) for t in TARGET_COLUMNS}})


def test_submission_schema_validation():
    template = frame(["a", "b", "c"])
    assert validate_submission_frame(frame(["a", "b", "c"]), template) == []
    assert validate_submission_frame(frame(["a", "b"]), template)
    assert validate_submission_frame(frame(["a", "c", "b"]), template)
    bad = frame(["a", "b", "c"])
    bad.loc[0, "ACL"] = 1.2
    assert validate_submission_frame(bad, template)
    bad.loc[0, "ACL"] = np.nan
    assert "missing values" in validate_submission_frame(bad, template)
    swapped = frame(["a", "b", "c"])[[SUBMISSION_COLUMNS[0], *reversed(SUBMISSION_COLUMNS[1:])]]
    assert validate_submission_frame(swapped, template)
