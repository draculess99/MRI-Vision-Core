"""Gold-label evaluation harness: split discipline, metric correctness, and the one-look EVAL ledger."""

import numpy as np
import pandas as pd
import pytest

from mri_core.report_label_eval import (EvalAlreadySpentError, assign_splits, evaluate, gold_studies, run_gold_evaluation,
                                        split_of)
from mri_core.report_labels import LabelState, ReportLabels, TargetLabel, unsupported_labels
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

P, N, U, M = (LabelState.POSITIVE, LabelState.NEGATIVE, LabelState.UNCERTAIN, LabelState.NOT_MENTIONED)


def make_train(n_gold=40, n_unlabeled=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_gold + n_unlabeled):
        labelled = i < n_gold
        row = {"StudyInstanceUID": f"1.2.826.{i}", "Report": f"report {i}"}
        for j, target in enumerate(TARGET_COLUMNS):
            row[target] = float(rng.integers(0, 2)) if labelled else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


class TruthExtractor:
    """Stub: returns the gold label as positive/negative (perfect), or a fixed state for everything."""

    def __init__(self, train, mode="perfect", version="stub-1", language="en"):
        self.version, self.mode, self.language = version, mode, language
        self.truth = train.set_index("StudyInstanceUID")

    def extract(self, report_text, study_uid=None):
        if self.mode == "unsupported":
            return unsupported_labels("tr", "stub", self.version, study_uid)
        labels = []
        for target in TARGET_COLUMNS:
            if self.mode == "perfect":
                state = P if self.truth.loc[study_uid, target] == 1 else N
            else:
                state = M
            labels.append(TargetLabel(target, state, self.language, "stub", self.version))
        return ReportLabels(study_uid, self.language, True, "stub", self.version, tuple(labels))


class TestSplit:
    def test_split_is_deterministic_independent_of_row_order_and_uses_both_halves(self):
        train = make_train()
        gold = gold_studies(train)
        a = assign_splits(gold)
        b = assign_splits(gold.sample(frac=1, random_state=3).reset_index(drop=True))
        assert dict(zip(gold.StudyInstanceUID, a)) == dict(zip(
            gold.sample(frac=1, random_state=3).reset_index(drop=True).StudyInstanceUID, b))
        assert set(a) == {"dev", "eval"}
        assert split_of("1.2.826.5") == split_of("1.2.826.5")

    def test_only_complete_label_studies_are_gold(self):
        train = make_train(n_gold=12, n_unlabeled=5)
        assert len(gold_studies(train)) == 12


class TestMetrics:
    def test_perfect_extractor_scores_perfectly(self):
        train = make_train(60, 0)
        gold = gold_studies(train)
        ex = TruthExtractor(train)
        report = evaluate([ex.extract("", u) for u in gold.StudyInstanceUID], gold, rounds=50)
        for target in TARGET_COLUMNS:
            o = report["targets"][target]["overall"]
            assert o["positive_precision"] == 1 and o["positive_recall"] == 1
            assert o["negative_reliability"] == 1 and o["coverage"] == 1 and o["auroc"] == 1
            assert o["confusion"]["positive"]["gold_negative"] == 0
            assert o["ci95"]["auroc"] is not None
        assert report["macro_auroc_defined_targets"] == 1

    def test_not_mentioned_everywhere_has_zero_coverage_and_chance_auroc(self):
        train = make_train(60, 0)
        gold = gold_studies(train)
        ex = TruthExtractor(train, mode="blank")
        report = evaluate([ex.extract("", u) for u in gold.StudyInstanceUID], gold, rounds=20)
        o = report["targets"]["ACL"]["overall"]
        assert o["coverage"] == 0 and o["positive_recall"] == 0 and o["positive_precision"] is None
        assert o["auroc"] == 0.5
        assert o["confusion"]["not_mentioned"]["gold_positive"] + o["confusion"]["not_mentioned"]["gold_negative"] == 60

    def test_auroc_is_none_when_a_class_is_too_rare(self):
        train = make_train(30, 0)
        train["ACL"] = 0.0
        train.loc[0, "ACL"] = 1.0
        gold = gold_studies(train)
        ex = TruthExtractor(train)
        assert evaluate([ex.extract("", u) for u in gold.StudyInstanceUID], gold, rounds=20)["targets"]["ACL"]["overall"]["auroc"] is None

    def test_unsupported_language_is_its_own_group_with_no_coverage(self):
        train = make_train(30, 0)
        gold = gold_studies(train)
        ex = TruthExtractor(train, mode="unsupported")
        report = evaluate([ex.extract("", u) for u in gold.StudyInstanceUID], gold, rounds=20)
        assert report["language_groups"] == {"unsupported": 30}
        assert report["targets"]["MCL"]["unsupported"]["coverage"] == 0

    def test_misaligned_inputs_are_rejected(self):
        train = make_train(10, 0)
        with pytest.raises(ValueError):
            evaluate([], gold_studies(train))


class TestEvalLedger:
    def test_dev_is_unlimited_and_eval_is_one_look_per_version(self, tmp_path):
        train = make_train(40, 0)
        ex = TruthExtractor(train)
        ledger = tmp_path / "ledger.jsonl"
        for _ in range(3):
            run_gold_evaluation(ex, train, split="dev", ledger_path=ledger, rounds=10)
        assert not ledger.exists()
        first = run_gold_evaluation(ex, train, split="eval", ledger_path=ledger, rounds=10)
        with pytest.raises(EvalAlreadySpentError):
            run_gold_evaluation(ex, train, split="eval", ledger_path=ledger, rounds=10)
        with pytest.raises(EvalAlreadySpentError):  # "all" also contains the EVAL half
            run_gold_evaluation(ex, train, split="all", ledger_path=ledger, rounds=10)
        run_gold_evaluation(TruthExtractor(train, version="stub-2"), train, split="eval", ledger_path=ledger, rounds=10)
        run_gold_evaluation(ex, train, split="eval", ledger_path=ledger, force=True, rounds=10)
        lines = [l for l in ledger.read_text().splitlines() if l]
        assert len(lines) == 3 and '"forced": true' in lines[-1]
        assert first["split_sizes"]["dev"] + first["split_sizes"]["eval"] == 40

    def test_dev_and_eval_halves_are_disjoint_and_cover_all_gold(self):
        train = make_train(58, 0)
        ex = TruthExtractor(train)
        dev = run_gold_evaluation(ex, train, split="dev", rounds=5)
        ev = run_gold_evaluation(ex, train, split="eval", rounds=5)
        assert dev["n_studies"] + ev["n_studies"] == 58
        gold = gold_studies(train)
        parts = assign_splits(gold)
        assert (parts == "dev").sum() == dev["n_studies"] and (parts == "eval").sum() == ev["n_studies"]
