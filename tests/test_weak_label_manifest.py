"""Weak-label manifest builder: eligibility, leakage guards, masking, determinism and safe output handling."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from mri_core.report_label_rules import RuleBasedExtractor
from mri_core.weak_label_manifest import (LeakageError, ManifestError, assert_no_leakage, build_manifest,
                                          load_training_arrays, select_eligible, write_manifest)
from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from scripts.build_weak_label_manifest import main as cli_main

EN = "Findings: The study of the knee was performed and the following is noted in this report.\n"
ES = "Hallazgos: Se realiza el estudio de la rodilla y se observa lo siguiente en el informe.\n"
TR = "Diz MR incelemesinde ve bir ile yok izlenmektedir dir normal gorunumde uzerinde ön çapraz bağ yırtığı mevcuttur."
SECRET_PHRASE = "Complete tear of the anterior cruciate ligament."


def make_train():
    reports = {
        "A": EN + SECRET_PHRASE + " No joint effusion.",
        "B": ES + "Rotura del cuerno posterior del menisco medial. Sin derrame articular.",
        "C": TR,
        "D": EN + "Possible tear of the medial meniscus.",
        "E": EN + "The study is otherwise unremarkable.",
        "GOLD": EN + "Complete tear of the anterior cruciate ligament.",
        "TEST": EN + "No tear of the anterior cruciate ligament.",
        "HOLD": EN + "Large joint effusion.",
        "BLANK": "",
    }
    rows = []
    for uid, text in reports.items():
        row = {"StudyInstanceUID": f"1.2.826.{uid}", "Report": text or np.nan}
        for target in TARGET_COLUMNS:
            row[target] = 1.0 if uid == "GOLD" else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


TEST_IDS = ["1.2.826.TEST"]
HOLDOUT = ["1.2.826.HOLD"]


class TestEligibility:
    def test_gold_test_holdout_and_empty_reports_are_excluded(self):
        eligible, counts, gold = select_eligible(make_train(), test_ids=TEST_IDS, holdout_ids=HOLDOUT)
        assert set(eligible.StudyInstanceUID) == {f"1.2.826.{x}" for x in "ABCDE"}
        assert gold == {"1.2.826.GOLD"}
        assert counts["excluded_has_structured_labels"] == 1 and counts["excluded_in_test"] == 1
        assert counts["excluded_in_holdout"] == 1 and counts["excluded_no_report"] == 1 and counts["eligible"] == 5

    def test_a_partially_labelled_study_is_treated_as_gold_and_excluded(self):
        train = make_train()
        train.loc[train.StudyInstanceUID == "1.2.826.A", "ACL"] = 0.0
        eligible, _, gold = select_eligible(train, test_ids=TEST_IDS)
        assert "1.2.826.A" in gold and "1.2.826.A" not in set(eligible.StudyInstanceUID)

    def test_leakage_guard_raises(self):
        with pytest.raises(LeakageError):
            assert_no_leakage(["1.2.826.GOLD"], gold_ids={"1.2.826.GOLD"}, test_ids=[])
        with pytest.raises(LeakageError):
            assert_no_leakage(["x"], gold_ids=[], test_ids=["x"])
        assert_no_leakage(["x"], gold_ids=["g"], test_ids=["t"])


class TestBuild:
    def build(self, **kw):
        return build_manifest(make_train(), RuleBasedExtractor(), test_ids=TEST_IDS, holdout_ids=HOLDOUT, **kw)

    def test_gold_test_and_holdout_never_appear(self):
        rows, summary = self.build()
        ids = {r["study_uid"] for r in rows}
        assert not ids & {"1.2.826.GOLD", "1.2.826.TEST", "1.2.826.HOLD", "1.2.826.BLANK"}
        # C is an unsupported language; D (only a hedged finding) and E (nothing found) decide no target
        assert ids == {"1.2.826.A", "1.2.826.B"}
        assert summary["counts"]["unsupported_language"] == 1 and summary["counts"]["below_min_decided"] == 2

    def test_masks_values_and_unknowns(self):
        rows, _ = self.build(min_decided=0)
        by = {r["study_uid"]: r["targets"] for r in rows}
        a = by["1.2.826.A"]
        assert a["ACL"]["state"] == "positive" and a["ACL"]["mask"] == 1 and a["ACL"]["value"] == 1.0
        assert a["Effusion"]["state"] == "negative" and a["Effusion"]["value"] == 0.0
        d = by["1.2.826.D"]
        assert d["Medial Meniscus"]["state"] == "uncertain" and d["Medial Meniscus"]["mask"] == 0
        for row in by.values():
            for entry in row.values():
                if entry["state"] in ("uncertain", "not_mentioned"):
                    assert entry["mask"] == 0 and entry["value"] is None  # never a silent zero

    def test_unsupported_rows_are_fully_masked_when_included(self):
        rows, _ = self.build(include_unsupported=True, min_decided=0)
        turkish = next(r for r in rows if r["study_uid"] == "1.2.826.C")
        assert not turkish["supported"] and all(t["mask"] == 0 for t in turkish["targets"].values())

    def test_output_is_stable_across_runs_and_input_order(self):
        _, s1 = self.build()
        shuffled = make_train().sample(frac=1, random_state=7).reset_index(drop=True)
        _, s2 = build_manifest(shuffled, RuleBasedExtractor(), test_ids=TEST_IDS, holdout_ids=HOLDOUT)
        assert s1["manifest_sha256"] == s2["manifest_sha256"] and s1 == s2

    def test_report_text_is_not_stored_unless_requested(self):
        rows, _ = self.build()
        assert SECRET_PHRASE not in json.dumps(rows)
        rows_with, _ = self.build(include_evidence=True)
        assert SECRET_PHRASE in json.dumps(rows_with)


class TestFiles:
    def test_write_load_roundtrip_and_tamper_detection(self, tmp_path):
        rows, summary = build_manifest(make_train(), RuleBasedExtractor(), test_ids=TEST_IDS, holdout_ids=HOLDOUT)
        out, summary_path = write_manifest(rows, summary, tmp_path / "weak" / "m.jsonl")
        ids, values, masks = load_training_arrays(out)
        assert len(ids) == 2 and values.shape == (2, 12) == masks.shape
        assert np.all(np.isnan(values[masks == 0])) and set(np.unique(values[masks == 1])) <= {0.0, 1.0}
        out.write_bytes(out.read_bytes().replace(b'"mask": 0', b'"mask": 1', 1))
        with pytest.raises(ManifestError):
            load_training_arrays(out)

    def test_no_overwrite_without_flag_and_protected_paths(self, tmp_path):
        rows, summary = build_manifest(make_train(), RuleBasedExtractor(), test_ids=TEST_IDS)
        out = tmp_path / "m.jsonl"
        write_manifest(rows, summary, out)
        with pytest.raises(ManifestError):
            write_manifest(rows, summary, out)
        write_manifest(rows, summary, out, overwrite=True)
        with pytest.raises(ManifestError):
            write_manifest(rows, summary, tmp_path / "train.csv")
        with pytest.raises(ManifestError):
            write_manifest(rows, summary, out, overwrite=True, protected=[out])

    def test_tampered_training_mask_for_uncertain_is_rejected(self, tmp_path):
        rows, summary = build_manifest(make_train(), RuleBasedExtractor(), test_ids=TEST_IDS)
        for target in rows[0]["targets"].values():
            if target["state"] == "not_mentioned":
                target["mask"], target["value"] = 1, 0.0
                break
        summary["manifest_sha256"] = hashlib.sha256(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()).hexdigest()
        out, _ = write_manifest(rows, summary, tmp_path / "bad.jsonl")
        with pytest.raises(ManifestError):
            load_training_arrays(out)


class TestCli:
    def csvs(self, tmp_path):
        train = make_train()
        train.to_csv(tmp_path / "train.csv", index=False)
        pd.DataFrame({"StudyInstanceUID": TEST_IDS}).to_csv(tmp_path / "test.csv", index=False)
        return tmp_path / "train.csv", tmp_path / "test.csv"

    def test_dry_run_writes_nothing_and_logs_no_report_text_or_full_uids(self, tmp_path, capsys):
        train_csv, test_csv = self.csvs(tmp_path)
        before = hashlib.sha256(train_csv.read_bytes()).hexdigest()
        before_files = sorted(p.name for p in tmp_path.iterdir())
        assert cli_main(["--train-csv", str(train_csv), "--test-csv", str(test_csv), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "DRY RUN" in out and "1.2.826" not in out and SECRET_PHRASE not in out
        assert sorted(p.name for p in tmp_path.iterdir()) == before_files
        assert hashlib.sha256(train_csv.read_bytes()).hexdigest() == before

    def test_real_run_requires_explicit_output_and_leaves_inputs_untouched(self, tmp_path):
        train_csv, test_csv = self.csvs(tmp_path)
        with pytest.raises(SystemExit):
            cli_main(["--train-csv", str(train_csv)])
        before = hashlib.sha256(train_csv.read_bytes()).hexdigest()
        assert cli_main(["--train-csv", str(train_csv), "--test-csv", str(test_csv),
                         "--output", str(tmp_path / "out" / "weak.jsonl")]) == 0
        assert (tmp_path / "out" / "weak.jsonl").is_file()
        assert hashlib.sha256(train_csv.read_bytes()).hexdigest() == before
        with pytest.raises(ManifestError):
            cli_main(["--train-csv", str(train_csv), "--output", str(train_csv)])
