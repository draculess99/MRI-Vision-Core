# Phase 16 stop: why does the shuffled-label control score about 0.61 AUC?

Research prototype; AUCs are agreement with automated weak labels, not clinical accuracy. No gold studies, no GPU, no images were used here;
analysis is `scripts/phase16_null_diagnosis.py` (labels from the v0.1.0 manifest, `train_series.csv`, report language). The registered
Phase 16 rules are unchanged; Phase 16 stays stopped.

## Observed (kernel `rsna-weak-label-phase16`, seed 101, clip 1.0; kernel stopped by the registered control alarm)
| | validation macro AUC | hold-out macro AUC |
|---|---|---|
| real labels | 0.639 | 0.667 |
| shuffled twin | 0.611 | 0.617 |
Shuffled-twin predictions are almost constant (std about 0.001) and its loss stays at the constant floor.

## Demonstrated (checked)
- **Shuffle implementation is correct.** Re-deriving the Kaggle shuffle (seed 1101, same 942 training studies): every row moved (942/942); the
  multiset of (label, mask) rows is preserved; agreement between original and shuffled labels equals chance (ACL 0.775 vs 0.768 expected,
  Medial 0.513 vs 0.530, Lateral 0.760 vs 0.749), correlation within +/-0.05. Validation and hold-out labels come straight from the manifest and
  never pass through the shuffle; train/validation/hold-out sets are pairwise disjoint (0 overlaps). Code path: `cmd_train` permutes only the
  training rows, then builds the training dataset from the permuted arrays; the validation dataset is built from the unpermuted ones.
- **The AUC computation is not the cause.** `roc_auc_score` is rank-based: a near-constant predictor (std 1e-3 around 0.5, float32) gives the
  same AUC as the original scores (0.5865 vs 0.5865 in a simulation; 5 ties among 500 values). Near-constant output therefore does not make the AUC
  numerically misleading; it only means the ranking is carried by tiny input-dependent differences.
- **The weak labels are strongly predictable from study metadata alone, without images or any network.** A logistic regression on
  per-study series counts (total, per plane, fluid-sensitive, fat-suppressed), trained on the same 942 training studies and scored on the same
  validation and hold-out studies:

  | Predictor | validation macro AUC | hold-out macro AUC |
  |---|---|---|
  | acquisition metadata only | 0.633 (ACL 0.73, Med 0.57, Lat 0.60) | 0.652 (0.73, 0.58, 0.64) |
  | acquisition + report language | 0.648 | 0.656 |
  | report language + report length | 0.602 | 0.616 |
  | C0 real labels (Kaggle) | 0.639 | 0.667 |
  | C0 shuffled twin (Kaggle) | 0.611 | 0.617 |

  The number of series in a study alone gives ACL AUC 0.71-0.75 (validation and hold-out). Sampling error of these macro AUCs is roughly +/-0.02-0.03,
  not computed here.

## Hypotheses (not yet tested)
1. The image network, even when it cannot learn labels, can order studies by a protocol-correlated image property (slice counts, field of view,
   contrast, series completeness) that is associated with the weak labels; a near-constant output can still carry that ordering.
2. The same association explains part or all of the real model's AUC in Phases 14-16 (real C0 is not better than the metadata-only baseline above).
3. The association arises because studies with positive findings, or with detailed reports that the extractor can decide, tend to have more series.
4. An untrained, randomly initialised C0 may reproduce the 0.6 AUC (not tested; needs images).
Not demonstrated: any implementation bug, label leakage, or a defect in the metric.

## Consequences
- Weak-label AUC (here and in Phases 14-15) cannot be read as anatomical learning until it is shown to exceed a metadata-only baseline. The
  Phase 15 statement "real beats control by about 0.10" is not robust: the shuffled control ranged 0.52-0.61 across seeds.
- Phase 16 remains stopped. Resuming unchanged would likely repeat the alarm; any change to criteria must be a documented amendment made knowing the
  seed-101 result.

## Proposed minimal next diagnostic (Kaggle, not launched; about 0.3 GPU-hours, mostly CPU decoding)
1. DICOM header-only features for the 1,701 manifest studies (slice counts per plane, rows/columns, spacing, thickness, echo/repetition times if
   present): metadata baseline with true image-acquisition features, same splits as the registry.
2. Untrained C0, ten seeds, forward pass only on the seed-101 validation and the 501 hold-out studies: AUC of random networks.
3. Save per-study probabilities from `holdout-eval` and score model AUC within strata of series count, to see whether the trained models carry
   information beyond the acquisition confound.
Decision after that: either add "must exceed the metadata-only baseline" as a documented amendment before resuming, or conclude that the weak
labels in this form cannot support an image-learning claim.
