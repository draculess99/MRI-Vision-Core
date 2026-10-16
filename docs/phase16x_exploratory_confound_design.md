# Phase 16X: post-hoc exploratory acquisition-confounding diagnostic (prepared, NOT launched)

**Status: post-hoc and exploratory. It is not a confirmatory test and not part of the Phase 16 pre-registration.** It was designed after the Phase 16
control alarm (shuffled twin 0.611 / 0.617 AUC) and after seeing that a metadata-only logistic regression reaches 0.633 (validation) and 0.652
(hold-out) macro AUC. The registered Phase 16 criteria (`configs/phase16_registry.json`, `mri_core/weak_label_prereg.py`) are unchanged and
Phase 16 stays stopped. Validation and hold-out studies are only **scored** here; no model, feature, hyper-parameter, stratum or checkpoint is
selected with them. Results of this diagnostic must not be used to tune a model; they can only motivate a documented amendment.

## Question
Does the trained C0 rank studies by weak label better than (a) DICOM-header acquisition metadata and (b) untrained networks? A positive answer
needs an advantage over both that survives uncertainty and holds within acquisition strata.

## Design (code: `mri_core/weak_label_confound.py`, driver `scripts/kaggle_phase16x_driver.py`, tests `tests/test_weak_label_confound.py`)
1. **Header-only features** for every manifest study (1,701): pydicom `stop_before_pixels` with a fixed tag list (rows, columns, pixel spacing,
   slice thickness, slice spacing measured from positions, span, instance-number gaps, spacing irregularity, TE, TR, field strength, manufacturer)
   for the series the image model actually reads (same `select_series`), plus series counts per study and plane. No pixel data, no patient, date
   or institution fields. "Series completeness" = slice count, instance-number gap fraction, spacing irregularity.
2. **Metadata-only baselines**, fixed and untuned: L2 logistic regression and a small boosted tree (depth 3, 100 iterations). Imputation, scaling and
   manufacturer categories are learned from each **registered training partition only** (seeds 101-105); each is scored on its own registered validation set
   and on the 501-study hold-out. Single-class training targets give no prediction (reported as unavailable).
3. **Untrained C0, no optimisation:** forward passes on the seed-101 validation and hold-out studies for ten fixed seeds (9001-9010, fixed now) plus the
   exact initialisation of the Phase 16 seed-101 run. Eval mode (BatchNorm at initial running statistics). AUC reported per seed with mean, sd, min, max and
   the count of seeds with macro AUC >= 0.60 - the null band of this metric for an image network that has seen no labels.
4. **Trained model:** the recovered Phase 16 seed-101 real-label checkpoint (SHA-256 `last.pt` recorded in its `result.json`; re-hashed before and after use;
   refused if any hold-out study was in its training or validation set). Taken from the attached Phase 16 kernel output, so nothing is uploaded.
   The shuffled twin's weights were deleted by design, so it cannot be re-scored; untrained networks play that role.
5. **Comparison and strata:** macro AUC with study-level bootstrap CI (1,000 resamples); paired bootstrap differences trained - metadata (logreg,
   boosted); **within-stratum AUC** (comparisons only between studies of the same stratum, weighted by positive-negative pairs) for strata of
   (a) series count (tertiles from training studies) and (b) quintiles of the metadata-logreg score per target (cut points from training studies), i.e. does the
   model add ranking information beyond what acquisition metadata already provides? The share of positive-negative pairs that survive stratification is reported.
   Spearman correlation between each model's scores and the metadata score.
6. **Reading guide (descriptive, not decision rules):** no evidence of learning beyond acquisition if the paired difference to the metadata baselines has a
   CI containing 0 and the within-stratum AUC CIs contain 0.5 (about 0.5 is expected from a pure acquisition proxy). Evidence of something beyond acquisition
   would be a within-stratum AUC with a CI above 0.5 and a positive paired difference. A single trained model and one partition cannot establish generality.

## Uncertainty and limits
Bootstrap over studies (about +/-0.03 validation, +/-0.02 hold-out for macro AUC); ten untrained seeds give the initialisation spread; one trained checkpoint
only; header features cover acquisition, not site or patient factors; strata are coarse; hold-out and validation come from the same source and labeler.

## Integrity, leakage and preservation checks (all enforced in the driver)
Code bundle hash-verified; manifest hash, selection hash, hold-out pool hash and all five validation-partition hashes must equal the registry (strict outside the
local rehearsal); gold, test and train/validation/hold-out disjointness asserted for every partition; checkpoint hash before and after use; features restricted
to manifest studies; unreadable-header studies reported (stop if > 2%); a planned stop exits cleanly (code 0) with the reason in `final_summary.json`. New private kernel,
new output folder; Phase 15, Phase 16 and smoke outputs, all checkpoints and the leaderboard submission are untouched; nothing is deleted; no training occurs.

## Cost
| Step | Estimate |
|---|---|
| Header features (about 250k headers, 8 threads, cold input) | 3-10 min |
| Metadata baselines (5 partitions, two models) | < 1 min |
| Decode 128x9 cache for about 760 studies (validation + hold-out) | 7-9 min |
| 11 untrained + 1 trained forward passes (T4) | about 5 min |
| Bootstrap (two sets, about 70 metrics, 1,000 rounds) | 3-5 min |
| **Total wall time** | **about 20-30 min (kernel limit 2,700 s), about 0.5 GPU-hours of quota** |
Quota now 25.41 h remaining after the Phase 16 stop. Disk about 1 GB.

## Requirement before launch
The Phase 16 kernel must be attached as a kernel source (`draculess/rsna-weak-label-phase16`; its output includes the selection lists and the checkpoint). If Kaggle will not
attach an errored kernel's output, the fallback is to upload the recovered 137 MB checkpoint (hash-verified) as a private dataset - that needs explicit approval and is not done.
