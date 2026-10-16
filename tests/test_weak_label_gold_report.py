"""Frozen-checkpoint gold evaluation: leakage refusal, one look, identical studies, undefined metrics, paired comparison."""

import numpy as np
import pandas as pd
import pytest
import torch

from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from mri_core.weak_label_gold_report import auc_or_none, confusion_at, score_models_on_gold, wilson
from mri_core.weak_label_manifest import LeakageError
from mri_core.weak_label_train import GoldAlreadyScoredError

from .test_weak_label_experiment import fit, make_dataset

TARGETS = ("ACL", "Medial Meniscus", "Lateral Meniscus")


class FakeGold:
    """Stands in for RSNAKneeDicomDataset: yields tensors for gold studies, can fail on chosen ones."""

    def __init__(self, ids, fail=()):
        data = make_dataset(len(ids), seed=9)
        self.studies = pd.DataFrame({"StudyInstanceUID": ids})
        self.images, self.mask, self.fail = data.images, data.mask, set(fail)

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        if i in self.fail:
            raise ValueError("corrupt series")
        return {"images": self.images[i], "mask": self.mask[i]}


def gold_frame(n=40, seed=0, ids=None):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"StudyInstanceUID": ids or [f"gold.{i}" for i in range(n)]})
    for t in TARGET_COLUMNS:
        frame[t] = (rng.random(len(frame)) < 0.4).astype(float)
    return frame


@pytest.fixture
def checkpoints(tmp_path):
    fit(tmp_path / "a", epochs=1, arm="A")
    fit(tmp_path / "b", epochs=1, arm="B")
    return {"A": tmp_path / "a" / "last.pt", "B": tmp_path / "b" / "last.pt"}


def score(checkpoints, tmp_path, gold=None, fail=(), **kw):
    gold = gold if gold is not None else gold_frame()
    return score_models_on_gold(checkpoints, gold, None, ledger_path=tmp_path / "ledger.jsonl", rounds=100, targets=TARGETS,
                                dataset_factory=lambda: FakeGold(list(gold["StudyInstanceUID"]), fail), **kw)


class TestMetricHelpers:
    def test_undefined_is_none_not_zero(self):
        assert auc_or_none(np.array([0, 0, 0, 0, 1]), np.arange(5.0)) is None
        assert auc_or_none(np.array([0, 1] * 5), np.array([0.1, 0.9] * 5)) == 1.0
        assert confusion_at(np.array([0, 0]), np.array([0.1, 0.2]), 0.5)["sensitivity"] is None
        assert wilson(0, 0) is None and wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1]

    def test_confusion_counts_add_up(self):
        y, p = np.array([1, 1, 0, 0, 1]), np.array([0.9, 0.2, 0.6, 0.1, 0.5])
        c = confusion_at(y, p, 0.5)
        assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (2, 1, 1, 1) and c["sensitivity"] == pytest.approx(2 / 3)
        assert c["specificity"] == 0.5 and c["ppv"] == pytest.approx(2 / 3)


class TestGoldScoring:
    def test_report_structure_counts_and_paired_comparison(self, checkpoints, tmp_path):
        report = score(checkpoints, tmp_path, thresholds=(0.3, 0.5))
        assert report["n_evaluated"] == report["n_gold_total"] == 40 and report["invalid_or_missing"] == []
        for t in TARGETS:
            entry = report["per_target"][t]
            assert entry["positives"] + entry["negatives"] == 40
            for name in "AB":
                m = entry["models"][name]
                assert 0 <= m["auc"] <= 1 and len(m["confusion"]) == 2
                c = m["confusion"][1]
                assert c["tp"] + c["fp"] + c["tn"] + c["fn"] == 40
            assert "A-B" in entry["paired_auc_difference"]
        assert set(report["macro"]) == {"A", "B"} and "A-B" in report["macro_paired_difference"]
        assert report["models"]["A"]["gold_overlap_with_training"] == 0
        assert report["_arrays"]["probabilities"]["A"].shape == (40, 3)

    def test_a_study_unreadable_for_any_model_is_dropped_for_all_and_reported(self, checkpoints, tmp_path):
        report = score(checkpoints, tmp_path, fail={3})
        assert report["n_evaluated"] == 39 and report["invalid_or_missing"][0]["study"] == "...gold.3"[-9:]
        assert len(report["invalid_or_missing"]) == 1 and "gold.3" not in report["_arrays"]["uids"]
        assert report["_arrays"]["probabilities"]["A"].shape[0] == report["_arrays"]["probabilities"]["B"].shape[0] == 39

    def test_each_checkpoint_gets_one_look(self, checkpoints, tmp_path):
        score(checkpoints, tmp_path)
        with pytest.raises(GoldAlreadyScoredError):
            score(checkpoints, tmp_path)
        again = score(checkpoints, tmp_path, force=True)
        lines = (tmp_path / "ledger.jsonl").read_text().strip().splitlines()
        assert len(lines) == 4 and '"forced": true' in lines[-1]
        assert again["n_evaluated"] == 40

    def test_checkpoint_trained_on_a_gold_study_is_refused_before_any_scoring(self, checkpoints, tmp_path):
        train_ids = torch.load(checkpoints["A"], map_location="cpu", weights_only=True)["train_study_ids"]
        with pytest.raises(LeakageError):
            score(checkpoints, tmp_path, gold=gold_frame(ids=train_ids[:10] + [f"gold.{i}" for i in range(30)]))
        assert not (tmp_path / "ledger.jsonl").exists()  # a refused look is not recorded as a look

    def test_rare_class_gives_undefined_auc_and_is_deterministic(self, checkpoints, tmp_path):
        gold = gold_frame()
        gold["ACL"] = 0.0
        gold.loc[0, "ACL"] = 1.0
        a = score(checkpoints, tmp_path, gold=gold)
        assert a["per_target"]["ACL"]["models"]["A"]["auc"] is None and a["macro"]["A"]["auc"] is None
        b = score(checkpoints, tmp_path, gold=gold, force=True)
        assert np.array_equal(a["_arrays"]["probabilities"]["A"], b["_arrays"]["probabilities"]["A"])
        assert a["per_target"]["Medial Meniscus"]["models"]["B"]["auc_ci95"] == b["per_target"]["Medial Meniscus"]["models"]["B"]["auc_ci95"]
