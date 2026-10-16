# Inference integration manifest (for the AMD hackathon application)

Research prototype; not a diagnostic device. No model here is clinically validated.

## Source
- Repository: https://github.com/draculess99/MRI-Vision-Core, branch `hackathon/opencv-aws`
- Commit: `b4d06cfc3c70ff3fa5b84fd33ca11360306205a7` (Phase 14). Do not merge into `main` or `hackathon/amd`; copy files or cherry-pick.

## Files needed
| Purpose | Path |
|---|---|
| Shared engine (provenance, hashing, single/ensemble inference, compatibility check) | `mri_core/rsna_inference_engine.py`, `mri_core/RSNA_INFERENCE_ENGINE_README.md` |
| Inference primitives | `mri_core/rsna_knee_inference.py`, `mri_core/rsna_knee_ensemble.py` |
| Model architecture (RSNAKneeCNN, 12 outputs) | `mri_core/rsna_knee_model.py` |
| Preprocessing / dataset (plane selection, slice sampling, per-slice 1-99 percentile normalisation, bilinear resize) | `mri_core/rsna_knee_dataset.py`, `mri_core/rsna_knee_train.py` (`RSNAKneeDicomDataset`, `_normalize`), `mri_core/dicom_series.py` |
| Deterministic quality report (OK / REVIEW / INVALID) | `mri_core/decision.py` |
| Review gate, evidence contracts, offline stub reasoning adapter | `mri_core/agents/` (`review_gate.py`, `llm_adapter.py`, `contracts.py`, ...) |

The AMD branch (`5e3b793`) has none of `rsna_inference_engine.py` or `mri_core/agents/`.

## Model
- Production architecture RSNAKneeCNN; input per study: 3 planes (Axial, Coronal, Sagittal) x up to 9 slices x 1 channel x 128x128.
- Output: 12 logits, sigmoid per target, order `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture` (taken from the checkpoint's `target_columns`).
- Checkpoint keys required: `model`, `config`, `target_columns`.
- Verified in the checkpoint files: `target_columns` match the order above; `config` has image_size 128, dropout 0.2. The single checkpoint also stores slices_per_series 9 and the three planes; the ensemble folds omit those two keys, so the code defaults (9 slices, all three planes) apply. Both were saved at epoch 13.

## Dependencies
Python 3.12; torch 2.x (tested 2.14.0 CPU); numpy, pandas, pydicom, opencv-python-headless, Pillow, scikit-learn, streamlit (UI).

## Checkpoints (binaries are NOT in Git)
Approved existing baseline (SHA-256):
- `outputs/rsna-knee/checkpoint_best.pt` `479d28861619dd2f53768d7fc1925d70ca127d0b19c14a00fe08c211765e0635` (414,553 bytes)
- Five-fold ensemble `outputs/rsna-knee-ensemble/checkpoint_rsna_stratified_fold{1..5}.pt`:
  1 `bd4f2afabc509df0373dd78ee2ecec173173b791ff52f1cdefee7716dc957e1b`, 2 `92c477b9e0cb75fc7a9bae32dd9f0b7ee4ddb8478598c84cbef5094443858751`,
  3 `b9a106a6d24e260f65b6090841ad24f0ee00ef99471d79e4ad51a1e57c0a3207`, 4 `2b4847ddebaf2e884f41f75fa78d20bcb1fe0f55c978ebe0f5dad046cebd00f1`,
  5 `4dfcc5af6d66e98dfcbdca7183584f4dde271da710c37f14e23654925edef12d`

Optional Phase 14 experimental checkpoints (model C0, a different architecture; use `mri_core/weak_label_models.py`, not the production engine):
`last_best.pt` 128x9 `e3fcb278d2865701`..., 192x9 `1b679d7f93848734`..., 128x15 `dda95cc33e8eaaa9`..., 192x15 `268727b366a7e2a7`... (137.3 MB each; full hashes in `outputs/kaggle_smoke/out_p14/phase14/final_summary.json`, local only, git-ignored).

## Limitations
- The five-fold ensemble is documented (`mri_core/RSNA_INFERENCE_ENGINE_README.md`) as trained on the 58 gold-labelled studies only (45-48 per fold); mean fold macro-AUC about 0.57; public leaderboard 0.536. The training set of the single `checkpoint_best.pt` is not recorded in the file and was not re-verified here. Probabilities are uncalibrated.
- Phase 14 checkpoints were trained on automated weak labels (3 knee targets, English/Spanish reports); weak-validation AUC 0.597-0.606 is agreement with automated labels, within seed noise, not clinical accuracy, and not validated on the gold studies. They are not recommended for integration.
- Do not describe any model as clinically validated.
