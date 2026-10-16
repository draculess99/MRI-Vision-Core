# Phase 13: label audit, public label datasets, extraction profiles and the bounded C0/C2 comparison

Research prototype; not a diagnostic device. The 58 gold studies were not evaluated, consulted or used for any decision in this phase.
Report excerpts are not stored in this repository (the audit files live in git-ignored `outputs/`).

## 1. Preserved
Production manifest v0.1.0 (`b44de700…8818`, 1,701 studies), all baseline checkpoints (inventory compare: no changes), Phase 10 results and
the public leaderboard submission (0.536, no new submissions) are untouched. The default extractor profile still reproduces v0.1.0 exactly.

## 2. Label audit sample (98 excerpts, blinded)
`scripts/build_label_audit_sample.py` draws a deterministic sample from training-only reports (gold excluded), stratified by how a label
changed between profiles 0.1.0 and 0.2.1 (changed labels oversampled), language-balanced inside strata. It writes a **blind sheet**
(excerpt, empty adjudication columns) and a **key** (automated states, strata, a third-party reference). Strata: 14 not-mentioned to positive,
12 not-mentioned to negative, 14 uncertain to positive, 2 uncertain to negative, 6 not-mentioned to uncertain, 10 positive to uncertain,
6 negative to uncertain, 5 negative to positive, 22 unchanged (positive 8, negative 8, uncertain 6), 7 where an anatomy term is present
but nothing is decided.

**No human adjudication has happened, so no precision is claimed.** The sheet is ready for a qualified reader. For orientation only I did a
single-reader preliminary pass (stored separately as `label_audit_assistant_preliminary.csv`, not independent):
of the 63 sampled labels profile 0.2.1 had decided, my reading agreed with 60, called 3 uncertain, and contradicted none; I would have decided
8 labels that 0.2.1 left uncertain, and found that 6 of the 7 "anatomy mentioned but undecided" cases were something the rules missed (2 normal statements, 4 weak findings). Those 8+ observations pointed to four defects (section 4); because they were found on this sample, the improvement they give on
the same sample is circular and is **not** a validation.

**Unresolved cases** (reader uncertain or genuinely ambiguous, listed separately): partial ACL "injury", "suggesting a tear" phrasing in
Spanish, "continuous but non-specific signal", and region-limited statements; 17 of the 98 excerpts differ between my reading and profile 0.2.1 or are ambiguous.

## 3. Independent cross-check: public report-label datasets (investigated, not adopted)
Kaggle hosts many participant-built label sets for this competition. I read their metadata and, for one dataset (stevenleehans, CC0, 426 KB),
downloaded the label CSV to compare against our states; nothing was merged or used for training.

| Dataset (owner) | License | How labels were built (per its description) | Gold use |
|---|---|---|---|
| stevenleehans/rsna-knee-llm-report-labels | CC0 | LLM reads each report; 0.5 means "not addressed" | scored on the 58 gold studies (0.887; blend 0.893) |
| pilkwang/rsna-knee-llm-labels | CC0 | LLM, YES/NO/UNK verdicts plus severity, original language | scored on gold |
| nartaa/rsna-knee-hpo-assets | CC0 | Gemini re-read against the host's annotation definitions; 0.25 means "not addressed"; includes a per-label pick chosen on gold | picks chosen on gold |
| yunusgmsoy 2/3/4-source merged | CC0 | Gemini plus other public sets, consensus | benchmarked and merged on gold |
| laymond Qwen2.5-32B weak labels | CC0 | open-weights LLM, fillers from co-occurrence priors | tuned/validated on gold |
| gperdrizet pseudolabeled | **CC-BY-NC-SA-4.0** | Qwen2.5-7B + Phi4 ensemble, gating chosen with McNemar tests on gold | gate chosen on gold |
| riadmohamed42/jev-knee-labels | CC0 | hard-swap of two public sets chosen on gold, plus an ensemble | selected on gold; 74 "hand-verified" studies include the 58 |

**Concerns.** (1) Nearly every set was tuned, selected or scored against the 58 gold studies, so training on them would leak gold information
into any later gold evaluation. (2) Encodings differ (0.5, 0.25, 0.05 mean "not addressed" in different sets) and would be misread as soft
negatives if merged naively. (3) They are derived from competition reports built with hosted LLMs; the host allows hosted LLMs, but whether
redistributing derived labels publicly is compatible with the rules is not confirmed. (4) One set carries a non-commercial share-alike license.
**Decision:** do not replace or merge our labels with any of them.

**Use as an independent annotator (training reports only).** Where both our extractor and the stevenleehans labeler made a decision on a
Tier-1 target, they agree on 99.7-100% of target-study pairs (v0.1.0 and the later profiles alike); our decided labels are therefore very
likely accurate where decided, and the main limitation is coverage, not precision. For the labels that changed in profile 0.2.0, agreement was
250/251 (not-mentioned to positive), 130/130 (to negative), 113/113 (uncertain to positive), 5/5 (negative to positive) and a weak 14/17 for
uncertain to negative with 14 further pairs where the public labeler was uncertain or silent; that weak transition motivated profile 0.2.1.
This is agreement between two automated annotators on explicit statements, not ground truth.

## 4. Extraction profiles (separately versioned; counts and hashes)

| Profile | Change | Included studies | ACL pos/neg | Medial Meniscus pos/neg | Lateral Meniscus pos/neg | Manifest hash |
|---|---|---|---|---|---|---|
| 0.1.0 (production) | none | 1,701 | 193 / 1,276 | 468 / 794 | 225 / 1,217 | `b44de700…8818` |
| 0.2.0 | inline-heading split, weak findings subordinate to decisive ones, `R/O` hedge, negation after an explicit tear, "continuous" | 1,960 | 248 / 1,345 | 696 / 844 | 286 / 1,260 | `f7f21a54…3282` |
| 0.2.1 | 0.2.0 without the weak-to-negative promotion (public-labeler agreement 14/17) | 1,957 | 247 / 1,337 | 706 / 824 | 293 / 1,254 | `6443af24…` |
| 0.2.2 | 0.2.1 + Spanish "desgarro" as tear term, region-limited normal statements, `R/O` after an explicit tear, "no internal derangement" | 1,990 | 257 / 1,352 | 718 / 845 | 301 / 1,271 | `4d4cbc37…` |

The four 0.2.2 defects were visible in the audit sample and are plain logic or vocabulary errors, not judgment calls; the independent check
supports them: all 79 labels newly decided by 0.2.2 agree with the public labeler (47/47, 17/17, 12/12, 3/3). Uncertain and missing labels were
never converted to negatives by default, and the four-state representation and loss masks are unchanged. Profiles are opt-in
(`--extractor-profile`); each reproduces its hash.

## 5. Bounded Model C comparison on training-only validation (Kaggle T4, kernel `draculess/rsna-weak-label-phase13`)
Setup: C0 (ResNet-18 from scratch, 2.5D, Attention-MIL) vs C2 (C0 + GRU) x labels v0.1.0 vs v0.2.2 x seeds 0 and 1 = 8 runs, 6 epochs each,
no early stopping, FP16 AMP, identical hyperparameters and hash splits (paired by seed; the seed also changes the split). Both manifests were
regenerated inside Kaggle and matched the local hashes; 0 unreadable studies; 0 invalid samples in all runs; 299 new studies cached in 206 s.
Each run's weak-validation loss uses its own labels, which cannot rank label versions, so every run was also scored on a **common yardstick**:
validation studies shared by both manifests, on labels both manifests agree on (about 265-320 studies per target).

Common-yardstick macro AUC over ACL, Medial Meniscus, Lateral Meniscus (agreement with automated labels; not clinical accuracy):

| Condition | seed 0 | seed 1 | mean |
|---|---|---|---|
| C0, labels v0.1.0 | 0.666 | 0.532 | 0.599 |
| C2, labels v0.1.0 | 0.536 | 0.601 | 0.568 |
| C0, labels v0.2.2 | 0.601 | 0.504 | 0.553 |
| C2, labels v0.2.2 | 0.489 | 0.531 | 0.510 |

Paired differences (macro AUC): C2 minus C0 is -0.131 / +0.069 (v0.1.0 labels) and -0.112 / +0.026 (v0.2.2); v0.2.2 minus v0.1.0 labels
is -0.065 / -0.028 (C0) and -0.047 / -0.070 (C2).

Own-validation best loss (different label sets, so not comparable across label versions): v0.1.0 C0 1.138, C2 1.158; v0.2.2 C0 1.084, C2 1.083.
Best epochs ranged from 1 to 6, so the 6-epoch budget is not clearly sufficient or necessary.
Runtime: 21-26 s per epoch (full split), peak GPU memory 1.13 GB (C0) and 1.14 GB (C2), 8 runs plus caching in 2,434 s (about 0.68 GPU-hours).

**Reading.** Seed-to-seed variation within one condition (up to 0.13 macro AUC) is as large as every difference observed; two seeds cannot
separate the conditions. The only consistent pattern is that models trained on the 0.2.2 labels scored lower than their v0.1.0 counterparts
on the common yardstick in all four paired comparisons (-0.03 to -0.07), so the larger, corrected label set did not help this metric. A plausible but
untested reason is that the added studies come from different report styles and sites while the yardstick consists of studies that the earlier
rules also decided. Neither C2's GRU nor the corrected labels is supported by this evidence; neither is excluded.

## 6. Recommendation for the next experiment
1. Do not scale the C2 or the label-profile comparison at 128 px: variance dominates, and the label change did not help.
2. Test information content next, since the models see 27 slices at 128 px and the findings are small: C0 with v0.1.0 labels at 192 px and
   16 slices per series, with at least 5 seeds and the identical common yardstick, so seed noise can be estimated. Estimated cost: a
   cache rebuild of about 15 minutes (about 6 GB on disk) and roughly 10 minutes per run (about 1 GPU-hour in total), well inside the quota.
3. In parallel (no GPU): have a qualified reader adjudicate the 98-excerpt blind sheet so rule precision and the effect of each profile can be
   measured, and decide whether language coverage should be extended by rules or by an offline model.
4. Keep the gold studies unused until a candidate clearly beats the others on the common yardstick with multiple seeds.
