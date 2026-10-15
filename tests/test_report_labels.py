"""Contracts, language gating and the deterministic English/Spanish extractor (fixture text only)."""

import numpy as np
import pytest

from mri_core.report_label_language import detect_language, is_supported
from mri_core.report_label_rules import EXTRACTOR_VERSION, RuleBasedExtractor
from mri_core.report_labels import (LabelState, ReportLabelError, ReportLabels, TargetLabel, to_training_arrays,
                                    unsupported_labels)
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

EN = "Findings: The study of the knee was performed and the following is noted in this report.\n"
ES = "Hallazgos: Se realiza el estudio de la rodilla y se observa lo siguiente en el informe.\n"
P, N, U, M = (LabelState.POSITIVE, LabelState.NEGATIVE, LabelState.UNCERTAIN, LabelState.NOT_MENTIONED)
extractor = RuleBasedExtractor()


def states(text, prefix=EN):
    labels = extractor.extract(prefix + text)
    assert labels.supported, labels.language
    return {label.target: label.state for label in labels.labels}


def only(text, target, prefix=EN):
    return states(text, prefix)[target]


class TestContracts:
    def test_four_states_and_target_validation(self):
        assert {s.value for s in LabelState} == {"positive", "negative", "uncertain", "not_mentioned"}
        with pytest.raises(ReportLabelError):
            TargetLabel("Nonsense", P, "en", "rules", "v")

    def test_labels_must_cover_every_target_in_order(self):
        with pytest.raises(ReportLabelError):
            ReportLabels(None, "en", True, "rules", "v", ())

    def test_only_positive_and_negative_are_trainable_and_masked_values_are_nan(self):
        base = unsupported_labels("en", "rules", "v")
        mixed = tuple(TargetLabel(t, s, "en", "rules", "v") for t, s in zip(TARGET_COLUMNS, [P, N, U, M] * 3))
        labels = ReportLabels("s", "en", True, "rules", "v", mixed)
        values, mask = to_training_arrays(labels)
        assert list(mask[:4]) == [1, 1, 0, 0]
        assert values[0] == 1 and values[1] == 0
        assert np.isnan(values[2]) and np.isnan(values[3])  # uncertain / not_mentioned never become 0
        assert not np.any(values[mask == 0] == 0)
        soft_values, soft_mask = to_training_arrays(labels, uncertain="soft")
        assert soft_mask[2] == 1 and soft_values[2] == 0.5 and soft_mask[3] == 0
        assert to_training_arrays(base)[1].sum() == 0

    def test_evidence_is_truncated_and_hashed(self):
        label = TargetLabel("ACL", P, "en", "rules", "v", evidence="x" * 500)
        assert len(label.evidence) == 160
        assert "evidence" not in label.to_dict() and len(label.to_dict()["evidence_sha256"]) == 64


class TestLanguageGate:
    @pytest.mark.parametrize("text,expected", [
        (EN + "No tear.", "en"), (ES + "Sin rotura.", "es"),
        ("Rotura de menisco interno. Artrosis femorotibial medial. Derrame.", "es"),
        ("Diz MR incelemesinde ve bir ile yok izlenmektedir dir normal gorunumde uzerinde", "tr"),
        ("Менискът е нормален. Няма данни за руптура на лигаментите.", "cyrillic"),
        ("Δεν παρατηρείται ρήξη του πρόσθιου χιαστού συνδέσμου.", "greek"),
        ("", "unknown"), ("ACL", "unknown")])
    def test_detect(self, text, expected):
        assert detect_language(text) == expected

    def test_only_english_and_spanish_are_supported(self):
        assert is_supported("en") and is_supported("es")
        assert not any(is_supported(c) for c in ("tr", "de", "cyrillic", "greek", "unknown"))

    def test_unsupported_language_is_left_unclassified_not_guessed(self):
        labels = extractor.extract("Diz MR incelemesinde ve bir ile yok izlenmektedir dir normal gorunumde uzerinde "
                                   "ön çapraz bağ yırtığı mevcuttur.")
        assert not labels.supported and labels.language == "tr"
        assert all(l.state is M and l.reason == "unsupported_language" for l in labels.labels)
        assert to_training_arrays(labels)[1].sum() == 0


class TestNegationAndPositives:
    def test_english_negation(self):
        assert only("There is no tear of the anterior cruciate ligament.", "ACL") is N
        assert only("The ACL is intact.", "ACL") is N
        assert only("No joint effusion.", "Effusion") is N

    def test_negation_scope_covers_a_list_but_not_other_clauses(self):
        s = states("No tear of the ACL or medial collateral ligament. Medial meniscus: horizontal tear.")
        assert s["ACL"] is N and s["MCL"] is N and s["Medial Meniscus"] is P

    def test_spanish_negation(self):
        assert only("No se observa rotura del ligamento cruzado anterior.", "ACL", ES) is N
        assert only("Menisco medial de morfologia y señal conservada, sin signos de rotura.", "Medial Meniscus", ES) is N
        assert only("No hay quistes poplíteos patológicos.", "Baker's", ES) is N

    def test_positive_findings_in_both_languages(self):
        assert only("Complete tear of the anterior cruciate ligament.", "ACL") is P
        assert only("Rotura completa del ligamento cruzado anterior.", "ACL", ES) is P
        assert only("Horizontal tear of the posterior horn of the lateral meniscus.", "Lateral Meniscus") is P
        assert only("Rotura del cuerno posterior del menisco medial.", "Medial Meniscus", ES) is P
        assert only("Large joint effusion.", "Effusion") is P
        assert only("Synovitis is present.", "Synovitis") is P
        assert only("Baker cyst measuring 3 cm.", "Baker's") is P
        assert only("Bone contusion of the lateral femoral condyle.", "Contusion") is P
        assert only("Segond fracture of the lateral tibial plateau.", "Fracture") is P

    def test_unmentioned_targets_stay_not_mentioned_never_negative(self):
        s = states("Complete tear of the anterior cruciate ligament.")
        assert s["ACL"] is P
        assert all(s[t] is M for t in TARGET_COLUMNS if t != "ACL")


class TestUncertaintyAndHistory:
    def test_hedged_findings_are_uncertain(self):
        assert only("Possible tear of the medial meniscus.", "Medial Meniscus") is U
        assert only("Cannot exclude a tear of the lateral meniscus.", "Lateral Meniscus") is U
        assert only("Posible rotura del menisco lateral.", "Lateral Meniscus", ES) is U
        assert only("No se puede descartar rotura del ligamento cruzado anterior.", "ACL", ES) is U

    def test_trace_effusion_and_mild_oa_are_uncertain(self):
        assert only("Trace joint effusion.", "Effusion") is U
        assert only("Medial compartment: mild chondromalacia.", "Medial OA") is U

    def test_historical_and_postsurgical_context_is_never_negative(self):
        assert only("Status post anterior cruciate ligament reconstruction. The graft is intact.", "ACL") is U
        assert only("Prior partial medial meniscectomy.", "Medial Meniscus") is U
        assert only("Old healed fracture of the proximal tibia.", "Fracture") is U

    def test_indication_text_is_not_a_finding(self):
        text = "Indication: rule out ACL tear.\nFindings: The study of the knee was performed and the ACL is intact in this report."
        assert extractor.extract(text).state("ACL") is N

    def test_structure_not_visualised_is_uncertain(self):
        assert only("The anterior cruciate ligament is not well visualized.", "ACL") is U

    def test_conflicting_mentions_become_uncertain(self):
        text = EN + "The anterior cruciate ligament is intact.\nImpression:\nTear of the anterior cruciate ligament."
        labels = extractor.extract(text)
        assert labels.state("ACL") is U
        assert next(l for l in labels.labels if l.target == "ACL").reason == "conflicting_mentions"

    def test_degenerative_marrow_edema_is_not_called_a_contusion(self):
        assert only("Subchondral marrow edema in the medial tibial plateau with osteoarthritis.", "Contusion") is U


class TestMultipleFindingsInOneSentence:
    def test_no_cross_target_contamination(self):
        s = states("The ACL is intact and there is a tear of the medial meniscus with trace effusion.")
        assert s["ACL"] is N and s["Medial Meniscus"] is P and s["Effusion"] is U
        assert s["Lateral Meniscus"] is M

    def test_lesion_word_does_not_leak_across_a_semicolon_or_but(self):
        s = states("Complete ACL tear; the menisci are normal. Medial meniscus tear but the lateral meniscus is intact.")
        assert s["ACL"] is P and s["Medial Meniscus"] is P and s["Lateral Meniscus"] is N

    def test_negated_list_of_different_findings(self):
        s = states("No bone contusion or fracture.")
        assert s["Contusion"] is N and s["Fracture"] is N

    def test_a_positive_item_after_a_negated_one_keeps_its_own_polarity(self):
        s = states("No tear of the ACL and moderate effusion.")
        assert s["ACL"] is N and s["Effusion"] is P

    def test_anterior_and_posterior_horns_do_not_split_a_single_finding(self):
        assert only("Tear of the posterior and anterior horns of the medial meniscus.", "Medial Meniscus") is P

    def test_both_menisci_statement_applies_to_both(self):
        s = states("The medial and lateral menisci are intact.")
        assert s["Medial Meniscus"] is N and s["Lateral Meniscus"] is N


class TestOsteoarthritisCompartments:
    def test_compartments_are_separated(self):
        s = states("Medial compartment: moderate chondromalacia with osteophytes.\n"
                   "Lateral compartment: cartilage is normal.\nThe patellofemoral cartilage appears intact.")
        assert (s["Medial OA"], s["Lateral OA"], s["PF OA"]) == (P, N, N)

    def test_only_the_named_compartment_is_positive(self):
        s = states("Lateral compartment: severe osteoarthritis with full thickness cartilage loss.")
        assert s["Lateral OA"] is P and s["Medial OA"] is M and s["PF OA"] is M

    def test_side_is_carried_from_the_previous_sentence(self):
        s = states("In the medial compartment, there is a horizontal tear of the medial meniscus. "
                   "There is no focal chondrosis or chondral injury.")
        assert s["Medial OA"] is N and s["Lateral OA"] is M and s["Medial Meniscus"] is P

    def test_location_free_degenerative_mention_is_uncertain_for_all_three(self):
        s = states("Mild degenerative changes of the knee.")
        assert (s["Medial OA"], s["Lateral OA"], s["PF OA"]) == (U, U, U)

    def test_spanish_cartilage_normal_in_both_femorotibial_compartments(self):
        s = states("Cartílagos de ambos compartimentos femorotibiales sin alteraciones significativas.", ES)
        assert s["Medial OA"] is N and s["Lateral OA"] is N and s["PF OA"] is M

    def test_meniscal_degeneration_is_not_osteoarthritis(self):
        s = states("Degenerative signal in the posterior horn of the medial meniscus.")
        assert s["Medial OA"] is M and s["Lateral OA"] is M and s["PF OA"] is M
        assert s["Medial Meniscus"] is U


class TestDeterminismAndPrivacy:
    def test_output_is_deterministic_and_versioned(self):
        text = EN + "Complete ACL tear. No effusion."
        a, b = extractor.extract(text, "u1"), extractor.extract(text, "u1")
        assert a == b and a.extractor_version == EXTRACTOR_VERSION

    def test_serialised_labels_exclude_report_text_by_default(self):
        labels = extractor.extract(EN + "Complete ACL tear.", "u1")
        assert "Complete ACL tear" not in str(labels.to_dict())
        assert "Complete ACL tear" in str(labels.to_dict(include_evidence=True))
