# Phase 16X diagnostic report (post-hoc, exploratory; Kaggle kernel `draculess/rsna-weak-label-phase16x` version 2)

**Evidence class: exploratory and post-hoc.** The diagnostic was designed after the Phase 16 control alarm and after seeing that metadata predicts the weak
labels. It is not a confirmatory test, the pre-registered Phase 16 rules were not applied or changed, and nothing here may be used to select a model.
AUCs measure agreement with automated weak labels, not clinical accuracy. Gold studies, test images and the leaderboard were not used. No training occurred
(the driver has no optimiser; the output holds no checkpoint).

Reproduce: kernel inputs `draculess/rsna-weak-label-code` (bundle `824db5d`) and `draculess/rsna-weak-label-phase16x-inputs`; local post-analysis
`scripts/phase16x_postanalysis.py`; local null checks `scripts/phase16_null_diagnosis.py`; outputs in `outputs/kaggle_smoke/out_p16x_v2/` (git-ignored).

## Integrity (all verified from `final_summary.json`)
`stage = done`, `all_passed = true`, `stopped = null`, 1,712 s wall (28.5 min; GPU quota 4.72 h to 5.19 h = 0.47 h; 24.81 h remain). Code bundle 72 files hash-verified;
manifest hash `b44de700…` equals the registry; 1,200-study selection hash `83a416a0…` equals the registry; the 501-study hold-out is a subset of the registered
pool (`257eb145…`, equals the registry) and disjoint from all five registered partitions; gold overlap 0, test overlap 0 for all five partitions; every input file
matches `PROVENANCE.json`; checkpoint SHA-256 `c17d4664e95a…` equal to its originally recorded hash before and after use. Header features: 1,701 studies, 0 failed.
Warnings: 71 DICOM "metadata says X but orientation indicates Oblique" warnings during decoding (the known Phase 8 condition; not an error); notebook-conversion
SyntaxWarnings. No incomplete outputs. All five registered validation-partition hashes were enforced by the driver (strict mode).

## Results (macro AUC over ACL, Medial Meniscus, Lateral Meniscus; 95% study-level bootstrap CI, 1,000 resamples)
Seed-101 registered partition: 942 training studies, 258 validation (positives ACL 25, Medial 71, Lateral 38), 501 hold-out (positives 59, 136, 69).

| Predictor | validation | hold-out |
|---|---|---|
| Trained C0 (Phase 16 seed 101, real labels) | 0.639 [0.569, 0.702] | 0.667 [0.622, 0.711] |
| Metadata logistic regression | 0.644 [0.579, 0.709] | 0.668 [0.625, 0.712] |
| Metadata boosted tree | 0.686 [0.634, 0.742] | 0.674 [0.635, 0.717] |
| Untrained C0, ten seeds (9001-9010) | mean 0.504, sd 0.061, range 0.414-0.598, 0 of 10 >= 0.60 | mean 0.524, sd 0.046, range 0.461-0.607, 1 of 10 >= 0.60 |
| Initialisation of the Phase 16 seed-101 run (untrained) | 0.595 | 0.579 |

Metadata baselines fitted on each registered training partition (validation / hold-out): seed 101 0.644/0.668 (logreg), 0.686/0.674 (boosted); 102 0.662/0.689, 0.648/0.679;
103 0.595/0.663, 0.650/0.671; 104 0.656/0.682, 0.668/0.683; 105 0.642/0.672, 0.652/0.675.

Paired differences, trained C0 minus metadata (CI includes zero in every aggregate comparison):
| | validation | hold-out |
|---|---|---|
| minus logistic regression | -0.005 [-0.077, 0.060] | -0.001 [-0.045, 0.043] |
| minus boosted tree | -0.047 [-0.123, 0.024] | -0.007 [-0.050, 0.035] |

Per target (hold-out, trained vs logreg / boosted; paired CI): ACL 0.695 vs 0.783 / 0.794 (-0.088 [-0.158, -0.019] / -0.099 [-0.163, -0.034]);
Medial Meniscus 0.650 vs 0.556 / 0.542 (+0.094 [0.026, 0.169] / +0.109 [0.036, 0.185]); Lateral Meniscus 0.656 vs 0.665 / 0.687 (-0.009 [-0.079, 0.065] / -0.032 [-0.106, 0.037]).
Validation per target: ACL 0.735 vs 0.730/0.794, Medial 0.572 vs 0.556/0.600, Lateral 0.611 vs 0.647/0.665; all paired CIs include zero.
(Six per-target comparisons per set, no multiplicity correction; per-target analysis is a second-order post-hoc look.)

Within-stratum AUC (comparisons only between studies of the same stratum; stratum cut points from training studies):
- Strata by series count: the training tertile cut points coincided (5, 5), so there are effectively two strata (up to 5 series vs more); 60-68% of positive-negative pairs remain.
  Trained 0.639 [0.566, 0.707] (val), 0.675 [0.628, 0.720] (hold-out) - essentially unchanged from the pooled AUC, and equal to metadata (difference -0.008 / -0.004, CI includes 0).
- Strata by quintile of the metadata-logreg score (only 17-21% of pairs remain): trained 0.582 [0.506, 0.657] (val) and 0.603 [0.553, 0.651] (hold-out);
  the logistic metadata score itself 0.493 / 0.531 (about 0.5 as expected); the boosted tree 0.578 / 0.557.
  This is the only place where the trained model keeps above-chance ranking after conditioning on the linear metadata score, but the boosted metadata model also keeps 0.56-0.58
  and the interval for the validation estimate touches 0.5.

Acquisition-feature correlations (Spearman of the trained model's scores with header features): n_series 0.36-0.44 (validation) and 0.38-0.40 (hold-out) across targets, series per plane
(Sagittal 0.44-0.48), axial echo time 0.40-0.53, fluid-sensitive and fat-suppressed series counts 0.34-0.36; Spearman with the metadata-logreg score 0.21-0.29 (validation) and 0.21-0.38 (hold-out).
The trained network's ranking is therefore strongly aligned with acquisition properties, as are the untrained networks' random rankings in sign and size (|rho| up to about 0.24 with the metadata score).

## Interpretation
1. **Why Phase 16's shuffled twin scored 0.61/0.62:** the untrained initialisation of that very run already scores 0.595 (validation) and 0.579 (hold-out), and ten other untrained
   networks range 0.41-0.61. The shuffled twin barely moved from its initialisation (prediction spread 0.001), so its AUC is a property of the random initialisation, not of label leakage.
   The alarm fired because the chosen 0.60 threshold sits inside the spread of untrained networks (null sd about 0.05-0.06). It reflects a real feature of this metric: for a network that
   has seen no labels, a macro AUC of 0.60 is within the observed range.
2. **Trained C0 vs untrained:** the trained model (0.639 / 0.667) is above all ten untrained networks (maximum 0.598 / 0.607) and above its own initialisation (+0.044 / +0.088). With ten
   untrained draws this is an empirical one-sided p of about 0.09; modest, not decisive.
3. **Trained C0 vs metadata:** no advantage in aggregate. Paired differences are -0.005 and -0.001 against logistic regression and -0.047 and -0.007 against the boosted tree, with CI half-widths
   of 0.04-0.07: an advantage larger than about 0.04 (hold-out) to 0.06 (validation) is excluded, a smaller one is not. The trained model's scores are correlated with the same acquisition features.
4. **Leads (exploratory):** (a) Medial Meniscus on hold-out, trained exceeds both metadata models by about 0.1 with CIs above zero, but not on the validation set (which has fewer positives, CIs include 0);
   (b) above-chance within-metadata-score-stratum AUC of 0.58-0.60. Both are second-order post-hoc observations from one checkpoint and one partition; they motivate a pre-registered test, they do not establish anything.
5. **ACL:** the trained model is clearly worse than acquisition metadata on hold-out (-0.09 to -0.10); ACL weak labels are strongly predictable from series counts (AUC 0.78-0.79 with no images).
6. **Not demonstrated:** that C0 learns anatomy; that the weak labels measure disease rather than reporting and protocol patterns; that anything transfers to clinical accuracy or to the competition test set.

## Limits
One trained checkpoint and one partition; the shuffled twin's weights were deleted so it could not be re-scored; metadata features were limited to what headers expose in this pipeline (no contrast, field-of-view or scanner model features beyond those listed);
series-count strata collapsed to two; strata by metadata score keep few pairs; hold-out and validation share source and labeler with the training data; no multiplicity correction.

## Recommendation: **B - improve weak-label quality (and make a metadata baseline a standing control) before further C0 training**
Reasons: (i) the trained model is not distinguishable from acquisition-only prediction, so more training of the same recipe on the same labels is unlikely to produce a defensible claim; (ii) the labels are themselves highly predictable from series count
(ACL AUC about 0.79 without images), which indicates label construction and selection are tied to acquisition/reporting patterns - exactly what a better label audit (blinded human adjudication of the 98-excerpt sample, balanced selection, non-English coverage) should address;
(iii) the one image-specific lead (Medial Meniscus) is worth testing, but only against better labels and with a pre-registered "must beat the metadata baseline within strata" criterion. A (acquisition controls) is premature on labels of this quality;
C is the right fallback for the hackathon deliverable: keep the existing deployed inference engine and benchmark unchanged and make no claim for C0 beyond "agreement with weak labels, not above an acquisition-only baseline".
No further GPU experiment is launched by this report.
