"""Audit sample: deterministic, gold-excluded, stratified, blinded sheet vs key."""

import csv

import numpy as np
import pandas as pd

from mri_core.report_label_audit import STRATA, build_audit_sample, excerpt_for, write_audit_files
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

EN = "Findings: The study of the knee was performed and the following is noted in this report.\n"
ES = "Hallazgos: Se realiza el estudio de la rodilla y se observa lo siguiente en el informe.\n"


def make_train(n=60):
    texts = [EN + "Complete tear of the anterior cruciate ligament. The lateral meniscus is intact.",
             EN + "The ACL is intact. Tear of the medial meniscus with mild extrusion.",
             ES + "Rotura del menisco medial. Ligamentos cruzados y colaterales sin alteraciones.",
             "Técnica: RMN de la rodilla. Resultados: Rotura de menisco interno. Menisco externo normal.",
             EN + "R/O tear of the lateral meniscus. Possible ACL sprain."]
    rows = [{"StudyInstanceUID": f"1.2.826.{i}", "Report": texts[i % len(texts)] + f" Extra note {i}.",
             **{t: np.nan for t in TARGET_COLUMNS}} for i in range(n)]
    rows.append({"StudyInstanceUID": "1.2.826.gold", "Report": texts[0], **{t: 1.0 for t in TARGET_COLUMNS}})
    return pd.DataFrame(rows)


class TestAuditSample:
    def test_deterministic_gold_excluded_and_ids_unique(self):
        train = make_train()
        a = build_audit_sample(train, {"1.2.826.gold"}, seed=1)
        b = build_audit_sample(train.sample(frac=1, random_state=3).reset_index(drop=True), {"1.2.826.gold"}, seed=1)
        assert [r["audit_id"] + r["study_uid"] + r["target"] for r in a] == [r["audit_id"] + r["study_uid"] + r["target"] for r in b]
        assert all(r["study_uid"] != "1.2.826.gold" for r in a)
        assert len({r["audit_id"] for r in a}) == len(a) and len(a) <= sum(q for *_, q in STRATA)

    def test_changed_labels_are_oversampled_and_quotas_respected(self):
        sample = build_audit_sample(make_train(200), set(), seed=0)
        counts = {name: sum(r["stratum"] == name for r in sample) for name, *_ in STRATA}
        for name, _, _, quota in STRATA:
            assert counts[name] <= quota
        changed = sum(c for n, c in counts.items() if "->" in n)
        assert changed >= sum(c for n, c in counts.items() if "->" not in n)
        # the one-paragraph Spanish report only yields labels under the corrected profile
        assert any(r["stratum"].startswith("not_mentioned->") and r["language"] == "es" for r in sample)

    def test_excerpt_names_the_target_structure_only(self):
        text = EN + "Complete tear of the anterior cruciate ligament. The lateral meniscus is intact."
        assert "cruciate" in excerpt_for(text, "ACL") and "lateral meniscus" not in excerpt_for(text, "ACL")
        assert "lateral meniscus" in excerpt_for(text, "Lateral Meniscus")

    def test_blind_sheet_hides_automated_labels_and_key_keeps_them(self, tmp_path):
        sample = build_audit_sample(make_train(80), set(), seed=0)
        key_ref = {(r["study_uid"], r["target"]): "positive" for r in sample}
        sheet, key = write_audit_files(sample, tmp_path, reference=key_ref)
        sheet_rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
        header = " ".join(sheet_rows[0].keys())
        assert "state_old" not in header and "stratum" not in header and "study_uid" not in header
        assert all(r["adjudicated_state (positive|negative|uncertain|not_mentioned)"] == "" for r in sheet_rows)
        key_rows = list(csv.DictReader(open(key, encoding="utf-8")))
        assert len(key_rows) == len(sheet_rows) == len(sample)
        assert {"state_old", "state_new", "stratum", "reference_third_party_state"} <= set(key_rows[0])
        assert all(r["reference_third_party_state"] == "positive" for r in key_rows)
        assert [r["audit_id"] for r in key_rows] == [r["audit_id"] for r in sheet_rows]
