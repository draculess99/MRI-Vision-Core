# Phase 17: weak-label quality audit (CPU, local, text-free)

Research prototype; not a diagnostic device. Audited: the registered **1,200-study training selection x 12 targets = 14,400 target entries** (8,855 decided positive/negative labels).
Not used: the Phase 16 hold-out pool (501 studies), the 58 gold studies, test labels, model predictions or any model performance; the audit refuses those studies in code
(`assert_audit_scope`). Report text was read locally only for its length (never printed or stored). Original manifests (v0.1.0 `b44de700...`, v0.2.0 `f7f21a54...`, v0.2.2 `4d4cbc37...`),
dataset fingerprints, study partitions and checkpoints are untouched; outputs are new files in git-ignored `outputs/weak-labels/label_quality/` (`audit.json` sha `5fa60c6f...`,
`candidate_labels_tier1.npz`). Reproduce: `scripts/audit_weak_label_quality.py` (library `mri_core/weak_label_quality.py`, tests `tests/test_weak_label_quality.py`, 10 tests).

## 1. Findings per target (12 targets)
Columns: decided = positive or negative; high/medium/low = confidence tier of the decided labels (rules in section 3); stable = same decided state in profiles 0.1.0/0.2.0/0.2.2;
association = AUC of series count for the label (0.5 = none); en/es positive rate among decided.

| Target | decided (of 1,200) | positive rate | high / med / low | stable / unstable / flipped | label-vs-series-count AUC | en vs es positive rate (p) |
|---|---|---|---|---|---|---|
| ACL | 1,037 (86%) | 12.9% | 809 / 0 / 228 | 1,034 / 3 / 0 | 0.672 | 15.7% vs 1.9% (<0.001) |
| MCL | 967 (81%) | 2.8% | 767 / 0 / 200 | 961 / 6 / 0 | 0.811 | 3.5% vs 0% (0.008) |
| Medial Meniscus | 882 (73%) | 37.6% | 878 / 0 / 4 | 878 / 3 / 1 | **0.551** | 39.4% vs 32.1% (0.056) |
| Lateral Meniscus | 1,016 (85%) | 15.4% | 1,010 / 0 / 6 | 1,007 / 7 / 2 | 0.635 | 14.6% vs 18.3% (0.18) |
| Medial OA | 792 (66%) | 23.0% | 726 / 0 / 66 | 790 / 2 / 0 | 0.624 | 20.3% vs 32.9% (0.001) |
| Lateral OA | 800 (67%) | 18.0% | 733 / 0 / 67 | 795 / 5 / 0 | 0.606 | 14.1% vs 32.9% (<0.001) |
| PF OA | 700 (58%) | 32.0% | 641 / 0 / 59 | 699 / 1 / 0 | 0.610 | 29.1% vs 49.5% (<0.001) |
| Effusion | 952 (79%) | 57.8% | 550 / 402 / 0 | 948 / 4 / 0 | 0.489 | 62.2% vs 41.2% (<0.001) |
| Synovitis | 210 (17%) | 96.7% | 201 / 6 / 3 | 201 / 9 / 0 | 0.917 | 96.2% vs 98.1% (0.51) |
| Baker's | 533 (44%) | 32.5% | 171 / 350 / 12 | 529 / 4 / 0 | 0.485 | 39.6% vs 20.5% (<0.001) |
| Contusion | 574 (48%) | 18.6% | 107 / 77 / 390 | 570 / 4 / 0 | 0.607 | 24.2% vs 6.6% (<0.001) |
| Fracture | 392 (33%) | 11.7% | 45 / 345 / 2 | 389 / 3 / 0 | 0.436 | 9.0% vs 80% (<0.001; few decided Spanish) |

Per-state counts (positive / negative / uncertain / not mentioned) are in `audit.json`; e.g. ACL 134/903/120/43, Medial Meniscus 332/550/181/137, Lateral Meniscus 156/860/76/108.

## 2. Quality issues identified
1. **Missingness is not random and large for non-meniscal targets.** Decided fraction ranges 17% (Synovitis) to 86% (ACL). For Tier-1 targets, whether a label is decided correlates with
   series count (decided-vs-undecided AUC 0.39-0.48), i.e. studies with more series are less often decided. Undecided labels remain masked; none is treated as negative (verified in code and tests).
2. **Class structure:** MCL 2.8% positive and Synovitis 96.7% positive (7 negatives) cannot support learning; Fracture, Contusion, Synovitis and Baker's are decided in only 17-48% of studies.
3. **Confidence:** 22% of ACL and 21% of MCL decided labels are list-inherited negatives ("no X, Y, Z" lists, lower confidence); Contusion is 68% low confidence (global "normal" phrases and lists);
   Effusion, Baker's and Fracture negatives have no recorded rule reason (counted medium). Medial and Lateral Meniscus are 99% high confidence.
4. **Contradiction and unsupported labels:** no decided label lacks recorded evidence (0 unsupported). Label flips between extractor profiles are rare (Medial 1, Lateral 2 of ~900-1,000 decided;
   0 for the others); 1-9 labels per target become undecided in a later profile. Profile stability therefore removes almost nothing and is weak evidence of quality.
5. **Language (a site proxy) is confounded with labels:** ACL positive 15.7% (English) vs 1.9% (Spanish), OA, Effusion, Baker's and Contusion rates also differ strongly. An independent annotator
   (third-party LLM labels, descriptive only, not used for any rule) agrees with ours on 2,879 of 2,883 compared Tier-1 labels (e.g. Spanish ACL 206/206), so the Spanish ACL gap is not an extractor recall
   failure: it is a population/site difference that the labels carry.
6. **Acquisition correlation:** label-vs-series-count AUC is high for MCL 0.81, ACL 0.67, Lateral 0.64 and OA 0.61-0.62; lowest for **Medial Meniscus 0.55**. Cross-validated metadata-only logistic regression
   inside the 1,200 (labels of the audit set only): ACL **0.743** [0.694, 0.789], Lateral 0.649 [0.598, 0.702], Medial **0.555** [0.521, 0.594].
7. **Selection bias:** label-free comparison with all 4,349 eligible training studies: mean series count 5.52 vs 5.53 (no shift); but only English/Spanish reports with at least one decided label are
   included (1,701 of 4,349; 1,980 studies in unsupported languages are excluded), so all conclusions apply to that subset. Prevalence in the 1,200 (ACL 12.9% of decided) is close to the full manifest (13.1%).
   Precision of decided labels appears high (agreement with the independent annotator 99.9%); the problems are coverage, confounding and class structure, not rule precision.

## 3. Candidate higher-confidence subset (documented rules; label values unchanged)
A decided Tier-1 label is a candidate if: (a) positive or negative in the original manifest; (b) confidence **high**: positive with no modifier, or negative from `negated_lesion`, `normal_structure` or
`normal_cartilage`; (c) evidence recorded and not shared by 4 or more targets of the study (a global sentence); (d) the same decided state in profiles 0.1.0, 0.2.0 and 0.2.2 (a study dropped by a later profile counts as unstable).
No outcome, prediction, hold-out, gold or leaderboard information is used. The subset is a boolean mask over the original labels, never a rewrite. Fingerprint `e3cc0bd9f0806fc0...` (full SHA-256 in `audit.json`).

| Target | original decided (pos) | candidates (pos / neg) | retained |
|---|---|---|---|
| ACL | 1,037 (134) | 806 (132 / 674) | 77.7% |
| Medial Meniscus | 882 (332) | 874 (329 / 545) | 99.1% |
| Lateral Meniscus | 1,016 (156) | 1,001 (151 / 850) | 98.5% |
2,681 of 2,935 Tier-1 labels (91%); 1,181 studies keep at least one candidate. With rules (a)-(d) applied to all 12 targets: 6,586 candidates of 8,855 decided.
Effect on acquisition dependence (cross-validated metadata AUC, candidates only): ACL 0.704 [0.644, 0.752] (from 0.743), Medial 0.555, Lateral 0.647: the filter does **not** remove the confound; it mainly drops ACL list-inherited negatives.

## 4. Potential impact on training, and remaining uncertainty
- Filtering changes meniscus labels by 1-2% and ACL by 22%; expected effect on C0 is small. The dominant issues (coverage, language/site and acquisition confounding, class skew) are not repaired by confidence filtering.
- **Medial Meniscus is the cleanest target** (99% high confidence, weakest acquisition association 0.55, balanced classes) and was the only target where C0 beat metadata in Phase 16X (exploratory); it is the natural primary target.
- ACL labels are heavily acquisition- and language-linked (metadata AUC 0.74); a model can score well without images. Treat ACL and MCL as secondary/diagnostic.
- Uncertain: rule-based confidence is a proxy - no human adjudication exists; the independent annotator shares the same reports and was tuned on gold by its author; the profile-stability criterion is nearly vacuous;
  series count is one acquisition summary (header features are richer); the audit says nothing about whether labels match the images.
