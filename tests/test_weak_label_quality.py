"""Label-quality audit and candidate filtering: rules, scope guards, no silent negatives, statistics helpers."""

import numpy as np
import pytest

from mri_core import weak_label_quality as wq
from mri_core.rsna_knee_dataset import TARGET_COLUMNS


def entry(state, reason=None, sha="e0", tier_masked=False, section="findings"):
    return {"state": state, "reason": reason, "evidence_sha256": sha if state != "not_mentioned" else None, "section": section,
            "tier_masked": tier_masked, "mask": int(state in wq.DECIDED and not tier_masked), "value": {"positive": 1.0, "negative": 0.0}.get(state)}


def row(uid, overrides=None, default=("not_mentioned", None)):
    targets = {}
    for i, t in enumerate(TARGET_COLUMNS):
        st, reason = (overrides or {}).get(t, default)
        targets[t] = entry(st, reason, sha=f"sha-{uid}-{i}", tier_masked=t not in ("ACL", "Medial Meniscus", "Lateral Meniscus"))
    return {"study_uid": uid, "language": "en", "supported": True, "targets": targets}


def test_confidence_tiers_follow_the_documented_rules():
    assert wq.label_confidence(entry("positive"), 1) == "high"
    assert wq.label_confidence(entry("positive", "inherited_side"), 1) == "low"
    assert wq.label_confidence(entry("negative", "negated_lesion"), 1) == "high"
    assert wq.label_confidence(entry("negative", "list_inherited"), 1) == "low"
    assert wq.label_confidence(entry("negative", None), 1) == "medium"
    assert wq.label_confidence(entry("negative", "normal_structure"), wq.SHARED_EVIDENCE_LIMIT) == "low"  # global statement
    assert wq.label_confidence({**entry("negative", "normal_structure"), "evidence_sha256": None}, 1) == "low"  # unsupported
    assert wq.label_confidence(entry("uncertain", "hedged"), 1) == "undecided" and wq.label_confidence(entry("not_mentioned"), 0) == "undecided"


def test_shared_evidence_counts_group_targets_by_sentence():
    r = row("a", {"ACL": ("negative", "list_inherited"), "MCL": ("negative", "list_inherited")})
    r["targets"]["MCL"]["evidence_sha256"] = r["targets"]["ACL"]["evidence_sha256"]
    counts = wq.shared_evidence_counts(r)
    assert counts["ACL"] == 2 and counts["MCL"] == 2 and counts["Fracture"] == 0


def test_stability_labels():
    assert wq.stability(["positive", "positive", "positive"]) == "stable"
    assert wq.stability(["positive", "negative", "positive"]) == "flipped"
    assert wq.stability(["positive", "uncertain", "positive"]) == "unstable"
    assert wq.stability(["positive", None, "positive"]) == "unstable"  # dropped by a later profile
    assert wq.stability([None, None]) == "missing"


def make_profiles():
    ov = {"ACL": ("positive", None), "Medial Meniscus": ("negative", "negated_lesion"), "Lateral Meniscus": ("negative", "list_inherited")}
    p1 = {"s1": row("s1", ov), "s2": row("s2", {"ACL": ("uncertain", "hedged"), "Medial Meniscus": ("not_mentioned", None)}),
          "s3": row("s3", {"ACL": ("negative", "normal_structure")})}
    p2 = {"s1": row("s1", ov), "s2": row("s2", {"ACL": ("positive", None)}), "s3": row("s3", {"ACL": ("positive", None)})}  # s3 flips
    return {"0.1.0": p1, "0.2.0": p2}


def test_candidate_mask_is_a_subset_of_the_original_labels_and_never_rewrites_values():
    profiles = make_profiles()
    ids = ["s1", "s2", "s3"]
    res = wq.candidate_labels(profiles, ids, tier_targets=("ACL", "Medial Meniscus", "Lateral Meniscus"))
    j = {t: TARGET_COLUMNS.index(t) for t in TARGET_COLUMNS}
    assert res["candidate"][0, j["ACL"]] and res["candidate"][0, j["Medial Meniscus"]]
    assert not res["candidate"][0, j["Lateral Meniscus"]]  # low confidence (list_inherited)
    assert not res["candidate"][2, j["ACL"]] and res["stability"][2, j["ACL"]] == "flipped"
    assert not res["candidate"][1].any()  # uncertain / not mentioned are never candidates
    assert not (res["candidate"] & (res["mask"] == 0)).any()  # candidate implies originally decided
    assert res["values"][0, j["ACL"]] == 1.0 and res["values"][0, j["Medial Meniscus"]] == 0.0
    assert wq.never_negative_from_missing(profiles["0.1.0"], res, ids)


def test_uncertain_and_missing_never_become_negatives():
    profiles = make_profiles()
    res = wq.candidate_labels(profiles, ["s2"])
    for t in ("ACL", "Medial Meniscus"):
        j = TARGET_COLUMNS.index(t)
        assert res["mask"][0, j] == 0 and np.isnan(res["values"][0, j]) and not res["candidate"][0, j]
    broken = {"mask": res["mask"].copy(), "candidate": res["candidate"].copy()}
    broken["mask"][0, TARGET_COLUMNS.index("ACL")] = 1  # an uncertain label silently made trainable
    assert not wq.never_negative_from_missing(profiles["0.1.0"], broken, ["s2"])


def test_tier_masked_targets_are_not_candidates_unless_ignored_for_audit():
    profiles = {"0.1.0": {"s1": row("s1", {"Effusion": ("positive", None)})}}
    j = TARGET_COLUMNS.index("Effusion")
    assert not wq.candidate_labels(profiles, ["s1"])["candidate"][0, j]
    assert wq.candidate_labels(profiles, ["s1"], ignore_tier_mask=True)["candidate"][0, j]


def test_audit_scope_refuses_holdout_gold_and_test_studies():
    wq.assert_audit_scope(["a", "b"], ["a", "b", "c"], {"gold": ["g"], "test": ["t"]})
    with pytest.raises(wq.AuditScopeError):
        wq.assert_audit_scope(["a", "z"], ["a", "b"], {})
    with pytest.raises(wq.AuditScopeError):
        wq.assert_audit_scope(["a", "g"], ["a", "g"], {"gold": ["g"]})
    with pytest.raises(wq.AuditScopeError):
        wq.assert_audit_scope(["a", "h"], ["a", "h"], {"phase16 hold-out pool": ["h"]})


def test_fingerprint_is_deterministic_and_sensitive_to_the_subset():
    profiles = make_profiles()
    ids = ["s1", "s2", "s3"]
    a = wq.candidate_labels(profiles, ids)
    assert wq.candidate_fingerprint(ids, a) == wq.candidate_fingerprint(ids, wq.candidate_labels(profiles, ids))
    b = wq.candidate_labels(profiles, ids, require_stable=False)
    assert wq.candidate_fingerprint(ids, a) != wq.candidate_fingerprint(ids, b)


def test_statistics_helpers():
    assert wq.hanley_mcneil_se(0.7, 100, 300) == pytest.approx(0.0333, abs=0.002)
    n_pos, total = wq.positives_needed(0.70, 0.2, 0.05)
    assert 100 < total < 800 and n_pos == pytest.approx(total * 0.2, abs=2)
    assert wq.positives_needed(0.7, 0.2, 0.03)[1] > total  # tighter interval needs more studies
    assert wq.two_by_two_p(10, 100, 10, 100) == pytest.approx(1.0) and wq.two_by_two_p(40, 100, 5, 100) < 0.001
    assert wq.two_by_two_p(0, 0, 3, 10) is None and wq.two_by_two_p(0, 10, 0, 10) is None
    y = np.array([0, 0, 1, 1])
    assert wq.auc_of_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0 and wq.auc_of_score(np.zeros(4, int), np.arange(4.0)) is None


def test_cv_metadata_auc_detects_an_acquisition_dependent_label_and_a_clean_one():
    import pandas as pd
    from scripts.audit_weak_label_quality import cv_metadata_auc
    rng = np.random.default_rng(0)
    n = 300
    ids = [f"s{i}" for i in range(n)]
    table = pd.DataFrame({"n_series": rng.integers(3, 12, n).astype(float), "manufacturer": "ge"}, index=ids)
    confounded = (rng.random(n) < 1 / (1 + np.exp(-(table["n_series"].values - 7)))).astype(int)
    clean = (rng.random(n) < 0.3).astype(int)
    ones = np.ones(n, int)
    assert cv_metadata_auc(table, ids, confounded, ones)["auc"] > 0.75
    assert abs(cv_metadata_auc(table, ids, clean, ones)["auc"] - 0.5) < 0.1
    assert cv_metadata_auc(table, ids, np.zeros(n, int), ones) is None  # single class: unavailable, not 0.5
