# Phase 17A experiment protocol (FROZEN BEFORE ANY RUN; prepared, NOT launched)

Research prototype; not a diagnostic device. Machine-readable copies: `configs/phase17_registry.json`, `configs/phase17_candidate_v1.json`. Driver: `scripts/kaggle_phase17a_driver.py`.
Nothing in this protocol has been run on a GPU. A launch needs explicit approval.

## 1. Question and targets
Does C0 trained on the higher-confidence candidate labels (C1) rank studies better than C0 trained on the original weak-label mask (B0), and does either exceed what acquisition metadata alone (M0) or
untrained / label-shuffled networks (N, S) achieve? **Primary target: Medial Meniscus** (99% of its decided labels are high confidence; weakest acquisition association). **Secondary: ACL, Lateral Meniscus**
(reported separately, never pooled into a claim; ACL is strongly acquisition- and language-linked).

## 2. Frozen inputs
- Manifest v0.1.0 `b44de700...`; 1,200-study training pool (`83a416a0...`); candidate mask C1: 806 ACL / 874 Medial / 1,001 Lateral labels (positives 132 / 329 / 151), fingerprint `e3cc0bd9...`
  (override-file fingerprint in the registry); the mask only removes entries, keeps the original label values and never converts missing, uncertain or masked labels into negatives.
- Seeds 201-205 (never used earlier); each seed sets initialisation, data order and the 80/20 hash split of the pool; validation-set hashes are in the registry and enforced by the driver.
- Recipe (the Phase 16 recipe, unchanged, no tuning): C0 128x9, FP16 AMP, AdamW lr 1e-4, weight decay 1e-4, batch 8, 12 epochs, warm-up 1, clip 1.0, no early stopping, fixed final epoch.
- Excluded everywhere: gold studies (58), Kaggle test, the 501-study Phase 16 hold-out pool (never trained or validated on; not independent confirmation, see section 5).

## 3. Arms (per seed, same split and initialisation inside a seed)
| Arm | What | Notes |
|---|---|---|
| B0 | C0, original mask | `--label-mask B0` |
| C1 | C0, candidate mask | `--label-mask C1 --label-mask-file ... --expected-mask-fingerprint ...` |
| S | C1 mask, labels shuffled across training studies (seed 2000+seed), validation labels true | learning null; weights not kept |
| M0 | metadata-only logistic regression and boosted tree on header features, fitted on the same training studies, once with the B0 mask and once with the C1 mask | acquisition control |
| N | untrained C0, seeds 9001-9010, forward pass only | initialisation null (Phase 16X: macro AUC 0.41-0.61) |

## 4. Analysis (descriptive; no hypothesis is declared confirmed by weak-label validation)
For each seed, every arm is scored on the seed's validation partition against (a) the manifest's original weak labels and (b) the candidate-mask labels. Reported: per-target and macro AUC with study-level
bootstrap CIs (1,000 resamples), paired differences C1 - B0, C1 - M0 (boosted), B0 - M0, C1 - S, the spread of N, and descriptive aggregation across the five seeds (mean, sd, in how many seeds a difference is positive).
Medial Meniscus is the headline; ACL and Lateral Meniscus are tables. Every output carries `weak_label_descriptive_only = true` and `claim_allowed = false`. Pre-stated interpretation guide, not decision rules:
an arm is only interesting if it clearly exceeds M0 (boosted) and the N / S bands in most seeds; even then the statement is "agreement with automated labels", not accuracy.

## 5. Evaluation strategy: what counts as independent evidence
- **Not independent:** the Phase 16 hold-out (inspected in Phases 16 and 16X), the registered validation partitions (same pool, same labeler, same acquisition mix), any other automated annotator's labels, the Kaggle leaderboard.
- **Required for any claim:** an **adjudicated reference** - labels from qualified readers, blinded to model scores and weak labels, ideally from the images, on studies disjoint from the training pool, every validation
  partition, the Phase 16 hold-out, the gold set and the test set. The harness encodes this: `claim_allowed` is false for weak references and for any overlap with a protected set.
- **How it could be obtained (not done, none available now):** (1) reader adjudication of images for a random sample of the roughly 2,648 eligible studies outside the pool and hold-out (mostly unsupported report languages;
  this also tests cross-site transfer): for an AUC CI half-width of 0.05 at Medial Meniscus prevalence about 38% roughly 420-510 studies (160-194 positives); (2) the 58 gold studies, once, as an underpowered final look after a candidate is frozen;
  (3) an external expert-labelled knee-MRI dataset (e.g. Stanford MRNet, which the repository already has an adapter for) after confirming access and licence; its labels are meniscus tear versus not, not medial versus lateral, and its images differ in format, so it can test transfer only.
- **If none is available, the experiment reports exactly that limitation and no claim.** Weak labels are never substituted as ground truth.

## 6. Integrity, stopping and preservation rules
Code bundle and inputs-package hashes verified in the kernel; manifest, selection, override fingerprint and every seed's validation partition checked against the registry (strict); gold/test/Phase 16 hold-out overlap asserted for every partition;
a C1 run refuses a wrong or missing override and cannot resume as B0; per-run leakage report; stop (clean exit, reason in `final_summary.json`) on a registry or leakage failure, non-finite loss, two consecutive run failures, or if a seed block cannot finish within 6,900 s
(unfinished blocks are skipped, never truncated). Checkpoints: full resumable state each epoch, `last.pt` kept for B0 and C1, hashes recorded for all; shuffled-control weights are not kept; each run writes `result.json`; resume by attaching a previous output (`P17_PREV`).
No existing manifest, label, checkpoint or result is modified; outputs go to a new kernel.

## 7. Cost (estimates from Phase 15/16/16X timings)
15 CNN runs (5 seeds x B0, C1, S) at about 4.2 min plus scoring about 0.4 min each = about 70 min; cache decode of 1,200 studies about 11 min; untrained forward passes (10 seeds x 1,200 studies) about 6 min;
metadata baselines and bootstraps about 7 min; set-up about 2 min. **Total about 95 min, about 1.6 GPU-hours (cap 2 h); disk about 5 GB (cache 1.1 GB, 10 kept checkpoints 1.4 GB).**
Quota before launch: 24.81 h remaining.

## 8. Package and launch (prepared; do not run without approval)
Private dataset `rsna-weak-label-phase17a-inputs` (folder `outputs/kaggle_smoke/p17a_inputs`: candidate mask, selection list, header features, `PROVENANCE.json`; contains UID-level labels derived from reports - keep private) and an updated
`rsna-weak-label-code` version (current HEAD bundle). Kernel `outputs/kaggle_smoke/p17a` (T4, internet off, `kaggle kernels push -p . --accelerator NvidiaTeslaT4 -t 7200`).
