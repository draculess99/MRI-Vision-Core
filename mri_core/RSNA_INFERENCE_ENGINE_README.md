# RSNA Inference Engine — Shared, Non-Diagnostic Research Module

**Module:** `mri_core/rsna_inference_engine.py`

This is a thin, reusable wrapper around the existing, already-tested inference primitives
(`mri_core.rsna_knee_inference.run_inference`, `mri_core.rsna_knee_ensemble.run_ensemble_inference`).
It exists so that every hackathon branch (OpenCV, Nebius, Amazon, AMD, …) and every Streamlit
app imports **one** module for model loading/inference, instead of each one re-implementing
checkpoint loading, error handling, and provenance tracking on its own.

**No training happens here. No new model architecture is introduced. No DICOM parsing is
duplicated.** This module only orchestrates what already exists and already passed its own
test suites (`test_rsna_knee_inference.py`, `test_rsna_knee_ensemble.py`).

---

## Non-Diagnostic / Medical Safety Notice

> **This is a research prototype. It is not a diagnostic medical device, has no clinical
> validation, and must not be used for patient care decisions.**

- Every probability returned is a **raw sigmoid output** of a model trained on 46–48 studies
  per fold (58 labeled studies total, 5-fold cross-validated). It is **not a calibrated
  confidence score** — sigmoid outputs from a model this small, trained this briefly, do not
  carry the statistical guarantees that term implies.
- The engine returns **probabilities only**; it never converts them into a binary
  abnormal/normal decision, a severity grade, or any clinical language. That choice (if one
  is ever needed) belongs to the caller, with its own explicit, documented threshold policy —
  this module deliberately does not make it.
- When a study cannot be evaluated (missing plane, incomplete local download, unknown study
  UID), the engine returns `status="unable_to_evaluate"` with `review_required=True` rather
  than fabricating a result or silently substituting synthetic data.
- The **deterministic image-quality report** (`mri_core.decision.generate_decision_report`,
  `QualityStatus` OK/REVIEW/INVALID) is unchanged and is the correct place to look for the
  existing human-review trigger. This module does not replace, duplicate, or weaken it — the
  two are deliberately kept independent, exactly as they were before this module existed.

## Dataset Limitations

- Trained on **58 fully-labeled RSNA knee studies only** — the entire RSNA competition
  `train.csv` has 4,407 studies, but only 58 have all 12 target labels (verified directly from
  the CSV in an earlier phase; this is a hard ceiling, not a download gap).
- Each of the 5 fold checkpoints saw only 45–48 of those 58 studies during training.
- See `outputs/rsna-fivefold-benchmark/comparison_report.md` for the full benchmark: mean fold
  macro-AUROC 0.57 (random init) / 0.54 (MRNet-transfer), both below the previously-inspected
  0.6078 Kaggle-notebook reference, with explicit caveats about why that comparison isn't exact.

## How to Load a Model

```python
from pathlib import Path
from mri_core.rsna_knee_dataset import load_rsna_metadata
from mri_core.rsna_inference_engine import verify_ensemble_compatibility

metadata = load_rsna_metadata("data/rsna-knee", dicom_root="data/rsna-knee/raw")

# Single trusted checkpoint
checkpoint = Path("outputs/rsna-fivefold-benchmark/random/fold0/last.pt")

# Or the 5-fold ensemble -- verify compatibility first (no inference run, just a structural check)
fold_checkpoints = [
    Path(f"outputs/rsna-fivefold-benchmark/random/fold{i}/last.pt") for i in range(5)
]
report = verify_ensemble_compatibility(fold_checkpoints)
assert report.compatible, report.mismatches
```

## How to Run Inference on a Local Study

```python
from mri_core.rsna_inference_engine import run_single_model_inference, run_ensemble_model_inference

result = run_single_model_inference(metadata, checkpoint, study_uid)
# or
result = run_ensemble_model_inference(metadata, fold_checkpoints, study_uid)

if result.status == "ok":
    print(result.probabilities)       # {"ACL": 0.51, "MCL": 0.51, ...} -- 12 ordered floats in [0,1]
    print(result.latency_seconds)     # measured wall time for this call
    print(result.model_provenance)    # which checkpoint(s), by SHA-256 + epoch + fold
elif result.status == "unable_to_evaluate":
    print(result.warnings)            # e.g. "Study unavailable: ..." -- show this to the user, don't guess
```

`result.to_dict()` gives a plain-dict/JSON-serializable version of the same information,
including `result.disclaimer` (the exact non-diagnostic notice text above).

## How the Ensemble Operates

- Exactly 5 fold checkpoints, each trained on a disjoint cross-validation fold of the same
  58 studies (`outputs/rsna-fivefold-benchmark/manifest.json` records the exact split).
- Each model's sigmoid output is computed independently on the same input tensor, then the
  5 probability vectors are **arithmetic-mean averaged** — no voting, no thresholding.
- If any one of the 5 checkpoints is missing, malformed, or architecturally incompatible,
  the whole call raises `InferenceEngineError` — **no partial ensemble is ever returned.**
- **Important distinction, carried over from the Phase 5 benchmark design:** these 5 models
  each trained on 4/5 of the same 58 studies the ensemble would be "evaluated" on if you ran
  it on all 58 again. Scoring the ensemble on those same 58 studies and calling it an unbiased
  benchmark would be circular. The valid accuracy evidence is the Phase 5
  **out-of-fold** predictions (`outputs/rsna-fivefold-benchmark/{random,transfer}/fold*/oof_predictions.json`),
  where each study's prediction came only from the fold that held it out. The ensemble, as
  exposed by this module, is for **functional inference demonstrations** — confirming the
  plumbing works end-to-end on a real study — not a new accuracy claim.

## Checkpoint Provenance

Every `StudyInferenceResult.model_provenance` entry carries:
- `checkpoint_path` — exact file used
- `checkpoint_sha256` — full file hash, so a caller (or a future audit) can prove exactly
  which weights produced a given result
- `epoch`, `fold` — read directly from the checkpoint's own stored metadata
- `target_columns` — the 12 target names, in the exact order the probabilities dict uses

## Which Benchmark Results Are Applicable

Use `outputs/rsna-fivefold-benchmark/comparison_report.md` and `comparison_summary.json` as
the only validated accuracy reference for these checkpoints. Do not re-derive a new accuracy
number by running the ensemble (or any single fold) on all 58 studies and presenting that as
unbiased — every one of those studies was training data for 4 of the 5 folds.

## How Another Hackathon Branch Can Import This

```python
# From any hackathon branch (OpenCV, Nebius, Amazon, AMD), with mri_core on the path:
from mri_core.rsna_inference_engine import run_single_model_inference, run_ensemble_model_inference

# Keep model-loading logic out of your Streamlit app file entirely -- call these functions,
# render result.probabilities / result.warnings / result.disclaimer, and stop there.
```

No hackathon branch has been modified by this phase — this module exists only on
`hackathon/opencv-aws` so far, ready to be imported once each branch's own integration is
explicitly requested.
