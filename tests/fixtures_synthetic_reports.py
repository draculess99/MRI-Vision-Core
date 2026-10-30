"""Synthetic RSNA-like MRI reports for Phase 20A testing.

These are realistic but wholly synthetic reports, suitable for testing
agent extraction logic without exposing real patient data.
"""

SYNTHETIC_REPORTS = {
    "simple_acl_tear": {
        "study_uid": "1.2.826.synth.001.acl_tear",
        "text": """
CLINICAL HISTORY: ACL injury, MRI for evaluation.

TECHNIQUE: Multiplanar, multisequence MRI of the left knee without contrast.

FINDINGS:
There is complete ACL tear involving the proximal and mid-portions of the ligament.
The remaining ACL fibers are disrupted. Medial meniscus appears intact.
Lateral meniscus intact.

IMPRESSION:
Complete ACL tear.
""",
        "targets": {
            "ACL": "positive",
            "Medial Meniscus": "negative",
            "Lateral Meniscus": "negative",
        },
    },

    "uncertain_medial_meniscus": {
        "study_uid": "1.2.826.synth.002.medial_uncertain",
        "text": """
CLINICAL HISTORY: Knee pain, evaluate for meniscal tear.

TECHNIQUE: Standard multiplanar MRI of the right knee.

FINDINGS:
ACL is intact and appears normal. There is a possible small tear of the
medial meniscus at the posterior horn, though this may represent
degenerative changes. Lateral meniscus appears normal.

IMPRESSION:
Possible medial meniscal tear, though degenerative change cannot be excluded.
""",
        "targets": {
            "ACL": "negative",
            "Medial Meniscus": "uncertain",
            "Lateral Meniscus": "negative",
        },
    },

    "negation_with_intact": {
        "study_uid": "1.2.826.synth.003.negation",
        "text": """
CLINICAL HISTORY: Follow-up MRI evaluation.

TECHNIQUE: Multiplanar MRI without contrast.

FINDINGS:
The ACL is intact. No evidence of ACL tear or disruption.
No evidence of meniscal tears. The medial meniscus is intact.
The lateral meniscus is intact. No effusion.

IMPRESSION:
Intact ACL. No meniscal tears. No abnormalities noted.
""",
        "targets": {
            "ACL": "negative",
            "Medial Meniscus": "negative",
            "Lateral Meniscus": "negative",
        },
    },

    "medial_lateral_confusion_risk": {
        "study_uid": "1.2.826.synth.004.confusion",
        "text": """
CLINICAL HISTORY: Acute knee injury.

TECHNIQUE: MRI of the left knee.

FINDINGS:
There is a tear of the lateral meniscus, involving the posterior horn.
The ACL is intact. The medial meniscus is intact and normal.

IMPRESSION:
Lateral meniscal tear, posterior horn. Medial meniscus intact.
""",
        "targets": {
            "ACL": "negative",
            "Medial Meniscus": "negative",
            "Lateral Meniscus": "positive",
        },
    },

    "missing_finding": {
        "study_uid": "1.2.826.synth.005.missing",
        "text": """
CLINICAL HISTORY: Knee pain.

TECHNIQUE: Standard knee MRI.

FINDINGS:
ACL is intact. There is a medial meniscal tear at the body.
Lateral structures not specifically commented upon.

IMPRESSION:
Medial meniscal tear, intact ACL.
""",
        "targets": {
            "ACL": "negative",
            "Medial Meniscus": "positive",
            "Lateral Meniscus": "not_mentioned",
        },
    },

    "hedging_and_history": {
        "study_uid": "1.2.826.synth.006.hedging",
        "text": """
CLINICAL HISTORY: Rule out meniscal tear.

TECHNIQUE: MRI.

FINDINGS:
There is an area of increased signal within the medial meniscus that could
represent a tear, though this may be related to prior degenerative changes.
The ACL appears intact. Lateral meniscus cannot be excluded from having a
small tear, though no definite finding is identified.

IMPRESSION:
Possible medial meniscal tear versus degenerative signal. Rule out lateral
meniscal tear.
""",
        "targets": {
            "ACL": "negative",
            "Medial Meniscus": "uncertain",
            "Lateral Meniscus": "uncertain",
        },
    },

    "spanish_section": {
        "study_uid": "1.2.826.synth.007.spanish",
        "text": """
CLINICAL HISTORY: Lesion de ligamento.

FINDINGS:
Hay rotura del ACL que involucra las porciones proximal y media.
El menisco medial está intacto.
El menisco lateral está intacto.

IMPRESSION:
Rotura completa del ACL.
""",
        "targets": {
            "ACL": "positive",
            "Medial Meniscus": "negative",
            "Lateral Meniscus": "negative",
        },
    },

    "double_negation": {
        "study_uid": "1.2.826.synth.008.double_neg",
        "text": """
FINDINGS:
There is no evidence of ACL tear. The medial meniscus is not torn.
The lateral meniscus is not torn and appears normal.

IMPRESSION:
No acute findings.
""",
        "targets": {
            "ACL": "negative",
            "Medial Meniscus": "negative",
            "Lateral Meniscus": "negative",
        },
    },
}


def get_synthetic_report(key: str) -> dict:
    """Retrieve a synthetic report by key."""
    if key not in SYNTHETIC_REPORTS:
        raise ValueError(f"Unknown synthetic report key: {key}. Available: {list(SYNTHETIC_REPORTS.keys())}")
    return SYNTHETIC_REPORTS[key]


def get_all_synthetic_reports() -> dict:
    """Return all synthetic reports."""
    return SYNTHETIC_REPORTS.copy()
