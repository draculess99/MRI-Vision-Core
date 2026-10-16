"""Adapter: bridged Kaggle attention-MIL inference -> the shared agents' input contract.

`mri_core.agents.orchestrator.process_inference_result()` takes exactly one type:
`mri_core.rsna_inference_engine.StudyInferenceResult`. The bridge in `kaggle_mil_bridge.py`
produces a plain dict (see `run_real_checkpoint_smoke.py`), not that dataclass, because its
checkpoint is architecturally foreign to `mri_core`'s own loaders -- so the two are NOT
directly compatible, confirmed by inspecting `contracts.build_evidence_bundle()`, which reads
`result.study_uid`, `.status`, `.probabilities`, `.is_ensemble`, `.model_provenance` (each with
`.to_dict()`), `.latency_seconds`, `.warnings`, `.disclaimer`.

This module is that adapter. It constructs a real `StudyInferenceResult` with a real
`ModelProvenance` -- both imported unchanged from `mri_core.rsna_inference_engine`, never
reimplemented or subclassed -- so every downstream agents-layer module (specialist, critic,
consensus, review_gate) runs completely unmodified on a bridged result.

What is preserved exactly: study_uid, the 12 probabilities (already in mri_core's
TARGET_COLUMNS order -- confirmed identical to kaggle_mil_bridge.TARGET_COLUMNS), latency,
checkpoint path/SHA-256, the bridge's own non-diagnostic disclaimer text.

What is explicitly marked unavailable rather than guessed: `ModelProvenance.fold` is populated
only when the checkpoint filename matches the `foldN-...` convention used by the five real
Kaggle checkpoints; any other filename yields `fold=None`, never a fabricated guess.
`ModelProvenance.epoch` is the Lightning checkpoint's own `epoch` field, not an mri_core
training epoch.

A warning is always attached identifying this checkpoint as bridged/external, so the critic
layer's ENGINE_WARNINGS_PRESENT check fires and consensus never reports evidence as
"complete" for a bridged run without that caveat surviving to the workflow status.
"""

from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence

from mri_core.rsna_inference_engine import STATUS_OK, ModelProvenance, StudyInferenceResult

BRIDGE_WARNING = (
    "This result used an externally trained Kaggle checkpoint (efficientnet_b0 + attention-MIL) "
    "bridged into this adapter, not mri_core's native RSNAKneeCNN. It has not been evaluated, "
    "calibrated, or validated by this project."
)

_FOLD_PATTERN = re.compile(r"fold(\d+)", re.IGNORECASE)


def _parse_fold(checkpoint_path: str) -> Optional[int]:
    """Best-effort fold index from the `foldN-knee-mil-best.ckpt` filename convention.

    Returns None (never a guess) when the filename does not match -- callers must treat
    None as "unavailable", not as fold 0.
    """
    match = _FOLD_PATTERN.search(checkpoint_path)
    return int(match.group(1)) if match else None


def to_study_inference_result(
    *,
    study_uid: str,
    probabilities: Mapping[str, float],
    latency_seconds: float,
    checkpoint_path: str,
    checkpoint_sha256: str,
    disclaimer: str,
    extra_warnings: Sequence[str] = (),
    epoch: Optional[int] = None,
) -> StudyInferenceResult:
    """Build a real StudyInferenceResult from the bridge's run_study_inference() output.

    `probabilities` must already be in mri_core.rsna_knee_dataset.TARGET_COLUMNS order (both
    kaggle_mil_bridge.run_study_inference and run_real_checkpoint_smoke.main produce it that
    way); this function does not reorder or validate that -- build_evidence_bundle() does, and
    raises EvidenceBundleError if it is ever wrong.
    """
    provenance = ModelProvenance(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        epoch=epoch,
        fold=_parse_fold(checkpoint_path),
        target_columns=tuple(probabilities.keys()),
    )
    return StudyInferenceResult(
        study_uid=str(study_uid),
        status=STATUS_OK,
        probabilities=dict(probabilities),
        is_ensemble=False,
        model_provenance=[provenance],
        latency_seconds=float(latency_seconds),
        warnings=[BRIDGE_WARNING, *extra_warnings],
        review_required=True,
        disclaimer=disclaimer,
    )
