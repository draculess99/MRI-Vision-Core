"""Rule profile 0.2.0 (audited fixes) is opt-in; the default profile must keep reproducing the original manifest behaviour."""

import pytest

from mri_core.report_label_rules import EXTRACTOR_VERSION, EXTRACTOR_VERSION_V2, RuleBasedExtractor, split_inline_headings, split_sections
from mri_core.report_labels import LabelState
from mri_core.weak_label_manifest import build_manifest

from .test_weak_label_manifest import ES, EN, make_train

P, N, U, M = LabelState.POSITIVE, LabelState.NEGATIVE, LabelState.UNCERTAIN, LabelState.NOT_MENTIONED
v1, v2 = RuleBasedExtractor("0.1.0"), RuleBasedExtractor("0.2.0")
FILL = "The study of the knee was performed and the following is noted in this report.\n"


def state(extractor, text, target):
    labels = extractor.extract("Findings: " + FILL + text)
    assert labels.supported
    return labels.state(target)


class TestProfiles:
    def test_default_profile_is_the_original_and_versions_differ(self):
        assert RuleBasedExtractor().profile == "0.1.0" and RuleBasedExtractor().version == EXTRACTOR_VERSION
        assert v2.version == EXTRACTOR_VERSION_V2 != EXTRACTOR_VERSION
        with pytest.raises(ValueError):
            RuleBasedExtractor("9.9")

    def test_original_behaviour_on_the_audited_cases_is_unchanged_in_the_default_profile(self):
        assert state(v1, "R/O tear of the lateral meniscus.", "Lateral Meniscus") is P  # the documented v0.1.0 weakness
        assert state(v1, "Oblique tear of the lateral meniscus posterior horn without extrusion.", "Lateral Meniscus") is N

    def test_manifest_records_the_profile_version_and_changes_the_hash(self):
        rows1, s1 = build_manifest(make_train(), v1)
        rows2, s2 = build_manifest(make_train(), v2)
        assert s1["extractor_version"] == EXTRACTOR_VERSION and s2["extractor_version"] == EXTRACTOR_VERSION_V2
        assert s1["manifest_sha256"] != s2["manifest_sha256"]


class TestInlineHeadings:
    SPANISH = "Técnica: RMN de la rodilla. Resultados: Rotura de menisco interno. Se observa derrame articular en la rodilla. Impresión: Rotura de menisco medial."

    def test_a_leading_ignored_heading_no_longer_swallows_a_one_paragraph_report(self):
        assert v1.extract(self.SPANISH).state("Medial Meniscus") is M  # the audited bug: whole report ignored
        labels = v2.extract(self.SPANISH)
        assert labels.state("Medial Meniscus") is P and labels.state("Effusion") is P

    def test_splitter_is_accent_tolerant_case_insensitive_and_leaves_ordinary_text_alone(self):
        text = "TECNICA: sag PD. Comparison: none. Findings: ACL intact. The technique: was good."
        out = [l for l in split_inline_headings(text).splitlines() if l.strip()]
        assert [line.split(":")[0].strip().lower() for line in out[:3]] == ["tecnica", "comparison", "findings"]
        assert split_inline_headings("No headings, just a sentence about the meniscus.") == "No headings, just a sentence about the meniscus."
        names = [name for name, _ in split_sections("Technique: sag PD. Findings: tear of the medial meniscus.", inline_headings=True)]
        assert names == ["findings"]

    def test_a_report_without_headings_is_unchanged(self):
        text = FILL + "Complete tear of the anterior cruciate ligament."
        assert v1.extract(text).state("ACL") is v2.extract(text).state("ACL") is P


class TestAuditedRuleFixes:
    def test_rule_out_is_a_hedge(self):
        assert state(v2, "R/O tear of medial meniscal root.", "Medial Meniscus") is U
        assert state(v2, "Rule out tear of the lateral meniscus.", "Lateral Meniscus") is U

    def test_a_weak_finding_does_not_veto_an_explicit_tear(self):
        text = "Radial tear of the posterior horn of the medial meniscus.\nMild extrusion of the medial meniscal body.\nMyxoid degeneration of the medial meniscus."
        assert state(v2, text, "Medial Meniscus") is P

    def test_hedges_and_history_still_veto(self):
        assert state(v2, "Tear of the medial meniscus.\nPossible tear of the medial meniscus.", "Medial Meniscus") is U
        assert state(v2, "Tear of the medial meniscus.\nStatus post medial meniscectomy.", "Medial Meniscus") is U

    def test_explicit_negation_beats_a_weak_finding(self):
        text = "Degenerative change of the medial meniscus without tear.\nMyxoid degeneration of the medial meniscus."
        assert state(v2, text, "Medial Meniscus") is N

    def test_a_negation_of_a_later_weak_term_does_not_negate_an_earlier_tear(self):
        sentence = "Oblique tear of the lateral meniscus posterior horn without extrusion."
        assert state(v1, sentence, "Lateral Meniscus") is N and state(v2, sentence, "Lateral Meniscus") is P

    def test_a_real_negation_before_the_tear_still_negates(self):
        assert state(v2, "No tear of the lateral meniscus.", "Lateral Meniscus") is N
        assert state(v2, "There is no evidence of a tear of the medial meniscus.", "Medial Meniscus") is N
        assert state(v2, "Tear of the medial meniscus is not seen.", "Medial Meniscus") is N

    def test_continuous_ligaments_read_as_intact_in_both_languages(self):
        assert state(v2, "The anterior cruciate ligament is continuous.", "ACL") is N
        assert v2.extract("Hallazgos: " + "Se realiza el estudio de la rodilla y se observa lo siguiente en el informe.\n"
                          "Ligamentos cruzados y colaterales continuos.").state("ACL") is N

    def test_a_lone_weak_negation_with_a_hedge_is_not_a_negative(self):
        assert state(v2, "Increased signal of the ACL, not all in a straight line, R/O grade 1 injury.", "ACL") is U

    def test_unsupported_languages_and_four_states_are_preserved(self):
        tr = v2.extract("Diz MR incelemesinde ve bir ile yok izlenmektedir dir normal gorunumde uzerinde ön çapraz bağ yırtığı mevcuttur.")
        assert not tr.supported and all(l.state is M for l in tr.labels)
        assert {s.value for s in LabelState} == {"positive", "negative", "uncertain", "not_mentioned"}
