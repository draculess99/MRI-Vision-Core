# Phase 17 plan: controlled comparison on improved weak labels (PROPOSED - not started; needs approval)

Research prototype; not a diagnostic device. Builds on `docs/phase17_label_quality_audit.md` and `docs/phase16x_diagnostic_report.md`. No GPU, training or Kaggle job has been run for Phase 17.

## What the evidence says before designing anything
Trained C0 equals acquisition-metadata baselines in aggregate (hold-out 0.667 vs 0.668 / 0.674); an untrained network can reach 0.60 by chance (null sd about 0.05); labels are highly precise where decided
(99.9% agreement with an independent annotator) but have non-random coverage, strong language/site and acquisition confounding (ACL metadata-only AUC 0.74), and rule-confidence filtering barely changes them.
Medial Meniscus is the cleanest target (99% high confidence, metadata-only AUC 0.555). The one image-specific lead is Medial Meniscus on hold-out (+0.09 to +0.11 over metadata), exploratory.

## Hypotheses (pre-registered before any run)
- **H1 (primary, Medial Meniscus):** C0 trained on high-confidence Medial Meniscus labels ranks adjudicated reference labels better than the metadata-only boosted tree, with the lower bound of the paired 95% CI of the AUC difference above 0.
- **H2 (control sanity):** shuffled-label twins and untrained networks are indistinguishable from chance on the adjudicated set (mean within [0.45, 0.55]); otherwise the experiment is invalid (as in Phase 16).
- **H3 (secondary, ACL and Lateral Meniscus):** descriptive only; reported with acquisition-stratified AUCs; no claim without H1 and a within-stratum AUC with a CI above 0.5.
- Not tested: any clinical-accuracy claim, any claim from the leaderboard.

## Arms (all 128x9, lr 1e-4, 12 epochs, clip 1.0, FP16 AMP; training data = the 1,200-study pool, unchanged partitions logic)
| Arm | Description | Role |
|---|---|---|
| M0 | metadata-only boosted tree and logistic regression (header features, fitted on the same training studies) | acquisition control |
| N | untrained C0, 10 fixed seeds | initialisation null |
| S | C0 trained on label-shuffled candidate labels, paired with each real run (same seed, split, initialisation) | learning null |
| B0 | C0 on the original labels (Phase 16 recipe); only the existing seed-101 checkpoint is available, so B0 is re-run for the new seeds | existing CNN baseline |
| C1 | C0 on the high-confidence candidate subset (masked loss elsewhere; uncertain and not-mentioned stay masked, never negative) | treatment |
Optional C2 (only if H1 passes): C1 with acquisition-balanced sampling (weights by series-count stratum). New registered seeds 201-205 (never used), partitions fixed and hashed in a registry before any run.
Cost estimate (from Phase 15/16 timings): 5 seeds x {B0, C1, S-of-C1} x 12 epochs ~ 15 runs x 4.5 min + setup ~ 1.3 GPU-hours; plus a short CPU/GPU forward-pass job for N and M0. Budget cap 2 h.

## Evaluation (the part that makes a claim possible)
1. **Weak-label validation** (the registered partitions): descriptive only, never a basis for claims (labels are acquisition-linked and share errors with the training labels).
2. **New independent evidence required before claiming improvement:**
   a. A **human-adjudicated reference standard**: qualified readers, blinded to model scores and weak labels, label Medial Meniscus (and optionally ACL/Lateral) from the reports **and, ideally, the images**, on a random sample of studies
      that are in neither the 1,200 training pool nor the Phase 16 hold-out pool, nor the gold set. For a 95% CI half-width of 0.05 around an AUC of 0.65-0.75 at 38% prevalence: 160-194 positives (about 420-510 studies);
      for half-width 0.03: 445-540 positives (about 1,170-1,420 studies). A paired comparison against the metadata model needs more; a power analysis is part of the registration.
      The existing 98-excerpt blind audit sheet is a start for report-label precision only.
   b. The **58 gold studies** may be used once, for a final look, only after a candidate is frozen (an underpowered check, never selection).
   c. Optionally a **different annotator or site** (e.g. train English, evaluate Spanish with adjudicated labels) to test transfer across the language/site confound.
   Not accepted as evidence: the Kaggle public leaderboard, validation or hold-out AUC against weak labels, agreement with other automated annotators.
3. **Primary metric:** paired AUC difference C1 minus M0 (boosted tree) on the adjudicated Medial Meniscus labels, study-level bootstrap CI (1,000 resamples), plus within-stratum AUC (series-count strata and metadata-score strata) and the S and N null bands.

## Decision rules (to be frozen in `configs/phase17_registry.json` before any run)
- **supported:** H1 holds in the pooled adjudicated set and in at least 4 of 5 seeds; H2 holds; C1 beats B0 by a margin stated in advance; within-stratum AUC CI above 0.5.
- **not supported / invalid:** otherwise. Any change to criteria after seeing results is a labelled amendment.
- Stopping rules: non-finite loss, partition or hash mismatch, any leakage check failure, control alarm defined from the measured untrained null (not a fixed 0.60), time budget.

## Before any GPU time (CPU, local) - proposed steps once approved
1. Freeze the candidate-label file (`candidate_labels_tier1.npz`, fingerprint `e3cc0bd9...`) and the registry (seeds 201-205, partition hashes, thresholds).
2. Build the adjudication sample frame (random, excluding the three protected/used sets) and the blinded sheet; decide who adjudicates (human readers; no external API with report text).
3. Add training support for a label-mask override (candidate mask) with tests (original manifest never modified), a leakage test, and the registered analysis code, then a CPU dry run.
4. Only then request approval for the Kaggle runs.

## Risks
Adjudication is the bottleneck and cannot be automated; the Medial Meniscus lead may not replicate; acquisition confounding may be only partly captured by header features; English/Spanish-only labels limit generalisation to the hidden competition test set;
filtering barely changes labels, so C1 may equal B0; the metadata baseline can be strengthened (more header and scanner features), making H1 harder.

Note on the adjudication pool: studies outside the 1,200 and the 501 hold-out are mostly the 2,648 eligible studies the extractor cannot read (other languages, or no decided label). Image-based adjudication works for them,
but their report language/site mix differs from the training pool, so the evaluation would partly test cross-site transfer; the sampling frame and its composition must be registered.
