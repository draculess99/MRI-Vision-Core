"""Synthetic workflow demonstration -- NOT real neural-network inference.

Exists so the deterministic evidence-review / human-review / LLM-interpretation workflow can
be demonstrated (e.g. on Railway, or anywhere a real RSNA dataset and trained checkpoint
cannot or should not be bundled/redistributed) without a real study, a real checkpoint, or any
real forward pass. Every number here is a fixed, deterministic placeholder built the same way
this project's own offline tests already build synthetic fixtures -- nothing here is invented
clinical or predictive content, and nothing here is ever labeled as a real result.

Builds a `mri_core.rsna_inference_engine.StudyInferenceResult` directly (the same shared,
unmodified type the real bridged path produces -- via `agent_bridge_adapter`, not reused here
deliberately) so the downstream evidence-review/review-gate/LLM code runs on this synthetic
input through the identical, unmodified code path a real result would use. `agent_bridge_adapter
.to_study_inference_result()` is NOT reused here: it unconditionally attaches a warning stating
an externally trained Kaggle checkpoint was bridged, which would be false for this synthetic
path -- building the result directly avoids that specific mislabeling risk.
"""

from __future__ import annotations

from mri_core.rsna_inference_engine import STATUS_OK, ModelProvenance, StudyInferenceResult
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

SYNTHETIC_STUDY_UID = "SYNTHETIC-DEMO-NOT-A-REAL-STUDY-NO-REAL-INFERENCE-RAN"
SYNTHETIC_CHECKPOINT_PATH = "SYNTHETIC-DEMO-NO-REAL-CHECKPOINT-USED.ckpt"

SYNTHETIC_DISCLAIMER = (
    "SYNTHETIC WORKFLOW DEMONSTRATION. These 12 scores are fixed, fabricated placeholder "
    "numbers -- not the output of any neural-network forward pass, real or otherwise. No real "
    "MRI data, no real checkpoint, and no real inference were used to produce this screen. "
    "This exists only to demonstrate the evidence-review, human-review, and LLM-interpretation "
    "workflow in an environment without a real RSNA dataset/checkpoint available (for example, "
    "a Railway deployment before any real data has been authorized and bundled)."
)

SYNTHETIC_WARNING = (
    "This is a fully synthetic demonstration result. No real study was loaded, no real "
    "checkpoint was used, and no real neural-network inference occurred."
)


def build_synthetic_demo_result() -> StudyInferenceResult:
    """A fixed, deterministic placeholder StudyInferenceResult -- fabricated numbers, no real
    forward pass. Calling this twice always returns identical probabilities (deterministic,
    not randomized). Built directly from the shared StudyInferenceResult/ModelProvenance types
    (not via agent_bridge_adapter -- see module docstring for why)."""
    probabilities = {col: round(0.1 * ((i * 3 + 1) % 7), 4) for i, col in enumerate(TARGET_COLUMNS)}
    provenance = ModelProvenance(
        checkpoint_path=SYNTHETIC_CHECKPOINT_PATH,
        checkpoint_sha256="0" * 64,
        epoch=None,
        fold=None,
        target_columns=tuple(TARGET_COLUMNS),
    )
    return StudyInferenceResult(
        study_uid=SYNTHETIC_STUDY_UID,
        status=STATUS_OK,
        probabilities=dict(probabilities),
        is_ensemble=False,
        model_provenance=[provenance],
        latency_seconds=0.0,
        warnings=[SYNTHETIC_WARNING],
        disclaimer=SYNTHETIC_DISCLAIMER,
    )
