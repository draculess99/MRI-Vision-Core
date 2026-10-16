# Phase 17A implementation report (pipeline prepared; nothing launched)

Research prototype; not a diagnostic device. CPU and local data only; no Kaggle job, GPU run, training beyond tiny CPU tests, gold evaluation, leaderboard submission, AMD change, deployed-engine change or push.
Protocol and evaluation policy: `docs/phase17a_experiment_protocol.md`. Audit: `docs/phase17_label_quality_audit.md`. Plan: `docs/phase17_plan.md`.

## What was built
| Piece | File | Purpose |
|---|---|---|
| Frozen candidate definition | `configs/phase17_candidate_v1.json`, built by `scripts/build_phase17_freeze.py` | rules, counts, manifest/profile/selection hashes, candidate fingerprint `e3cc0bd9f0806fc0...` (recomputed from the manifests and required to equal the audited value), override-file hash; no study IDs |
| Frozen registry | `configs/phase17_registry.json` | seeds 201-205, validation-partition hashes, recipe, arms, protected sets, evaluation policy (claim not allowed without an adjudicated reference) |
| Label-mask override | `mri_core/weak_label_mask_override.py`; `train --label-mask {B0,C1} --label-mask-file ... --expected-mask-fingerprint ...` | B0 = original mask; C1 = original mask x candidate mask. Only removes entries; values untouched; bound to the manifest SHA-256, exact study IDs and a fingerprint; mismatch, tampering, a masked or differently valued candidate all raise; the dataset fingerprint includes the mask so B0/C1 checkpoints cannot be mixed on resume |
| Validation scoring | `score-validation` command | per-study probabilities for a checkpoint's own validation studies only (refuses overlap with training, gold or test) |
| Comparison harness | `mri_core/weak_label_compare.py` | typed reference labels (`weak` vs `adjudicated`), per-target and macro AUC with bootstrap CIs, paired differences, across-seed descriptive summary, `claim_allowed` guard (false for weak references and for any overlap with a protected set) |
| Metadata-only baseline | existing `mri_core/weak_label_confound.py` (header features), used by the driver for M0 | fitted per seed on the same training studies, once with each mask |
| Experiment driver | `scripts/kaggle_phase17a_driver.py` | B0, C1, S per seed; M0; N (10 untrained seeds); descriptive tables; resume; strict registry/leakage checks; clean exit with reason on any stop |
| Package (not uploaded) | `outputs/kaggle_smoke/p17a` (kernel), `outputs/kaggle_smoke/p17a_inputs` (candidate mask, selection, header features, provenance) | private; to be pushed only after approval |

## Candidate dataset (frozen)
Pool 1,200 studies; manifest v0.1.0. Tier-1 candidate labels: **ACL 806 (132 positive / 674 negative), Medial Meniscus 874 (329 / 545), Lateral Meniscus 1,001 (151 / 850)**; 2,681 of 2,935 original Tier-1 labels (91%); 1,181 studies retain at least one candidate.
Registered validation partitions (seeds 201-205): 238 / 252 / 236 / 233 / 242 studies, all hash-pinned (at most 63 studies overlap any earlier validation set); the training pool equals the Phase 16 pool, so the Phase 16 hold-out (501 studies) remains disjoint from every training and validation set.

## Tests
`tests/test_phase17a.py` (18 tests): mask correctness (B0 equals original, C1 subset, values and masks unchanged, undecided never trainable), exact-ID alignment and mismatched IDs, manifest/fingerprint binding and tamper evidence, registry/candidate consistency and never-used seeds,
partition determinism and disjointness, C1 training through the real CLI (candidate counts smaller than B0, manifest and existing B0 checkpoint hash unchanged, B0/C1 fingerprints differ, wrong or missing override refused), interrupted C1 run resuming to identical losses and weights and a B0 resume of a C1 directory refused,
validation scoring only for the checkpoint's own validation studies without rewriting the checkpoint, AUC and paired-difference correctness, claim guard for weak references and protected-set overlap, misaligned scores rejected, across-seed summary descriptive.
Full regression suite: **882 passed, 8 skipped**. CPU rehearsal of the full driver: see the final report message.

## Cost estimate for the (unlaunched) Kaggle run
About 95 min, about 1.6 GPU-hours (cap 2 h): 15 CNN runs about 70 min, decode 11 min, untrained forward passes 6 min, baselines and bootstraps 7 min, set-up 2 min. Disk about 5 GB.

## Remaining blockers and limits
1. **No independent labelled evaluation exists.** Without it the experiment can only produce descriptive agreement with weak labels; the harness enforces that and no claim is possible.
2. Requires approval, a private inputs dataset upload (UID-level report-derived labels), and a code-dataset version for the current bundle.
3. Header-feature baselines are limited to what the headers expose; the candidate filter changes meniscus labels by 1-2%, so C1 may equal B0.
4. Kaggle behaviour of the new commands (`--label-mask`, `score-validation`) is verified on CPU and in unit tests only; a short GPU smoke test is recommended before the full run.
