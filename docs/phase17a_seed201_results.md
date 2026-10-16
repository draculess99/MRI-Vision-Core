# Phase 17A seed 201 (Kaggle version 2): results report - exploratory, descriptive only

Research prototype; not a diagnostic device. One seed only. Every number below is agreement with the manifest's automated weak labels on one registered validation partition (238 studies); weak labels are not ground truth,
no independent adjudicated reference exists, and the harness records `claim_allowed = false`. No clinical-accuracy statement is made. Outputs: `outputs/kaggle_smoke/out_p17a_seed201_v2/` (git-ignored); the failed first attempt is preserved in `out_p17a_seed201/`.

## Run integrity (from `final_summary.json`)
`stage = done`, `all_passed = true`, no stop; bundle `b6a3792` (78 files); startup `bundle_search`, `discover_data` and `inputs_package` each 0.0 s (the 9-minute scan is gone; no `inputs_diagnosis`);
override fingerprint `79b8655e...` and selection hash `83a416a0...` equal the registry; seed-201 validation partition hash `b97c2728...` equals the registered value (962 train / 238 validation); leakage clean; gold, test and Phase 16 hold-out overlap 0; 0 unreadable studies.
B0, C1 and S each trained 12 epochs (1,452 steps); B0 and C1 `last.pt` downloaded and their SHA-256 match `result.json`; each holds model, optimizer, scheduler, AMP scaler, torch/python/numpy/CUDA/loader RNG, config (C1 records `label_mask`), epoch/step and metrics.
Dataset fingerprints differ (B0 `955416d9...`, C1 `a521c7ae...`, S `b2308f33...`). Wall time 1,557 s (26 min): decode 635 s, each CNN run about 217-228 s plus 8 s scoring; GPU quota 5.35 h to 5.79 h (0.44 h), 24.21 h remain. No warnings of consequence.

## Results (validation macro and per-target AUC, 95% study-level bootstrap CI, 1,000 resamples; reference = original-mask weak labels)
Validation positives / negatives: ACL 29/181, Medial Meniscus 62/117, Lateral Meniscus 23/180.

| Arm | Medial Meniscus (primary) | ACL | Lateral Meniscus | Macro |
|---|---|---|---|---|
| B0 (original mask) | 0.616 [0.525, 0.698] | 0.703 [0.605, 0.790] | 0.685 [0.565, 0.801] | 0.668 [0.593, 0.733] |
| C1 (candidate mask) | 0.604 [0.510, 0.687] | 0.674 [0.563, 0.772] | 0.683 [0.561, 0.802] | 0.654 [0.573, 0.724] |
| Metadata boosted tree (B0 labels) | 0.544 [0.443, 0.636] | 0.753 [0.635, 0.863] | 0.682 [0.566, 0.789] | 0.659 [0.583, 0.729] |
| Metadata boosted tree (C1 labels) | 0.564 [0.473, 0.651] | 0.728 [0.620, 0.839] | 0.674 [0.551, 0.788] | 0.656 [0.584, 0.725] |
| Metadata logistic regression (B0 / C1 labels) | 0.555 / 0.575 | 0.735 / 0.698 | 0.675 / 0.672 | 0.655 / 0.648 |
| S: shuffled-label control (C1 mask) | 0.557 [0.469, 0.642] | 0.428 [0.323, 0.538] | 0.434 [0.312, 0.566] | 0.473 [0.409, 0.541] |
| N: untrained C0, ten seeds | mean macro 0.512 (range 0.471-0.562) | | | |

Paired differences (point [CI]): C1 minus B0: Medial -0.012 [-0.031, 0.009], ACL -0.030 [-0.065, 0.001], Lateral -0.001 [-0.026, 0.024], macro -0.014 [-0.032, 0.005].
C1 minus boosted metadata (C1 labels): Medial +0.040 [-0.063, 0.148], ACL -0.055 [-0.204, 0.084], Lateral +0.009 [-0.122, 0.153], macro -0.002 [-0.090, 0.091].
B0 minus boosted metadata (B0 labels): Medial +0.072 [-0.038, 0.189], ACL -0.050, Lateral +0.003, macro +0.009 [-0.075, 0.095].
C1 minus S: macro +0.181 [0.053, 0.300]; ACL +0.246 [0.027, 0.437]; Lateral +0.249 [0.007, 0.480]; Medial +0.047 [-0.051, 0.151].
Scoring against the candidate-mask labels instead gives the same picture (Medial: B0 0.616, C1 0.603, boosted metadata 0.540-0.561, S 0.559; macro C1 minus S +0.154 [0.024, 0.280]).
Final validation loss is below the constant-prediction floor for both CNNs (B0 0.991 vs 1.047; C1 0.968 vs 1.009; each measured on its own mask, so B0 and C1 losses are not comparable with each other).

## What this does and does not show
- **B0 and C1 are indistinguishable** (C1 is nominally 0.01 lower; CI includes zero). The high-confidence filter did not help on this seed, as expected from its small effect on the meniscus labels.
- **Neither CNN is demonstrably better than acquisition metadata.** Medial Meniscus is the only target where both CNNs sit above both metadata models (+0.04 to +0.07) but every interval includes zero; for ACL the metadata is nominally better; macro AUC is equal (0.65-0.67 for all).
- **The controls behave:** shuffled-label macro AUC 0.47 and untrained networks 0.47-0.56, so the CNNs' 0.65-0.67 is above these nulls (macro C1-S +0.18 [0.05, 0.30]) - they learn something that agrees with the weak labels - but metadata alone reaches the same level.
- Limits: one seed and one 238-study partition (CIs about +/-0.09 per target); within-stratum analyses were not part of this run; the earlier Phase 16X lead on Medial Meniscus is neither confirmed nor refuted.

## Recommended next step
Do not launch the remaining four seeds to compare B0 and C1: the filter is not distinguishable from the original mask, and weak-label validation cannot yield a claim whatever the seed count (it would only narrow descriptive intervals, about 1.3 GPU-hours).
The value now lies in an independent adjudicated reference (see `docs/phase17a_experiment_protocol.md`, section 5) and in improving label coverage and confounding; the Medial Meniscus signal should be tested against that reference with the C1/B0 checkpoints preserved here (hashes in `result.json`).
Optional and cheap if the user wants tighter descriptive intervals: seeds 202-205 with the same pinned recipe (needs `P17_ALLOW_FULL=1` and explicit approval).
