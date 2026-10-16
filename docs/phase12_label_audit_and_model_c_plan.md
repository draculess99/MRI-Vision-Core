# Phase 12: weak-label audit, Model C readiness and experiment plan

Research prototype; not a diagnostic device. No gold-label result was used for any rule or design decision in this phase (the 58 gold
studies were neither re-evaluated nor consulted). Examples below are paraphrased; no report text is stored in this repository.

## 1. Label audit (training-only reports: the 4,349 studies with no structured labels)

Language coverage of the 4,349 eligible reports (script and stop-word identification): English 1,697, Spanish 672 (supported, 54%);
Turkish 539, Croatian 402, Greek 318, German 252, Cyrillic-script 217, Dutch 137, French 44, undetermined 71 (unsupported, 46%).
The current manifest (extractor `rules-en-es-0.1.0`) therefore contains 1,701 studies (39%).

### Systematic errors found (with rule profile `0.2.0` as the proposed correction)

| # | Finding | Evidence (training reports only) | Fix in profile 0.2.0 |
|---|---|---|---|
| 1 | **A leading "Technique:" or "Indication:" heading swallowed the whole report** when the report is one paragraph with later inline headings ("Findings:", "Resultados:", "Impression:") | 556 of 2,369 supported reports (23%) had under 20% of their text kept: 330 of 672 Spanish (49%) and ~226 English; 524 of those 556 contain an inline findings heading | split recognised headings that appear mid-line; add "resultados" |
| 2 | A weak finding (degeneration, extrusion, "grade 1 injury") **vetoed an explicit tear** found elsewhere in the same report (reason `mixed_certainty`, 159 medial-meniscus and 71 ACL reports) | many reports state a tear in the findings and mention extrusion or degeneration in the impression | weak statements are dropped when a decisive statement exists; hedges and history still veto |
| 3 | `R/O` ("rule out") is not treated as a hedge, so "R/O tear of the lateral meniscus" was **positive** | direct inspection of transitions | `R/O`, "rule out", "a descartar" are hedges |
| 4 | A negation cue after an explicit tear ("tear ... without extrusion") **negated the tear** (a false negative in the training labels) | 5 target-study pairs flipped from negative to positive; 8 more became uncertain, in spot checks all on this pattern | a cue that appears after the first explicit lesion term does not negate it |
| 5 | "continuous"/"continuo(s)" (ligaments reported as continuous = intact, common in Spanish) not read as normal | seen in Spanish reports | added as a normal cue |

Effect of the corrections (profile 0.2.0 vs 0.1.0, same 4,349 reports):

| | v0.1.0 | v0.2.0 |
|---|---|---|
| Studies with at least one decided Tier-1 target | 1,701 | **1,960** (+15%) |
| Studies with all three decided | 1,017 | 1,173 |
| ACL positive / negative | 193 / 1,276 | 248 / 1,345 |
| Medial Meniscus positive / negative | 468 / 794 | 696 / 844 |
| Lateral Meniscus positive / negative | 225 / 1,217 | 286 / 1,260 |
| Spanish decided fraction (ACL / MM / LM) | 0.45 / 0.44 / 0.46 | 0.51 / 0.60 / 0.50 |
| Spanish ACL positives among decided | 2.7% | 9.6% (English 17%) |

Transitions among Tier-1 target-study pairs: 255 not-mentioned to positive, 139 not-mentioned to negative, 116 uncertain to positive,
32 uncertain to negative, 57 not-mentioned to uncertain, 15 positive to uncertain, 8 negative to uncertain, 5 negative to positive.
Nothing was converted from uncertain or not-mentioned into negative by default; a negative needs an explicit negation or a normal statement.

**Validation without gold.** I read about 40 changed cases by eye (an informal spot check, not a measurement of precision):
the not-mentioned-to-decided and uncertain-to-positive changes read correctly; the first pass of the aggregation change produced a
few wrong negatives ("R/O ... injury" and "not all in a straight line") which fixes 3 and the negated-weak rule removed. This
is not a substitute for a labeled audit set (see section 5).

### Remaining gaps
- Spanish ACL is still under-detected (51% decided vs 73% English; 9.6% positive vs 17%). Likely causes: different wording ("lesion",
  "rotura parcial", "cicatrizacion") and ACL mentioned only inside generic "cruzados" statements.
- Grade 1 "injury" and mucoid degeneration of the ACL remain uncertain by design (the host overview defines the finding as tears).
- Contusion/effusion/synovitis/Baker's/fracture and OA targets are untouched (still Tier 2/3, masked).

### Opportunity: multilingual coverage (deterministic)
Share of reports in each unsupported language that mention a meniscus or cruciate term (keyword check, an upper bound on what a rule
set could decide):

| Language | Reports | Mentions meniscus or cruciate term | Mentions a lesion word |
|---|---|---|---|
| Turkish | 539 | 100% | 58% |
| Croatian | 402 | 100% | 94% |
| Greek | 318 | 100% | 67% |
| German | 252 | 91% | 82% |
| Cyrillic-script (Bulgarian/Serbian/Russian) | 217 | 99% | 77% |
| Dutch | 137 | 98% | 67% |
| French | 44 | 100% | 93% |
| **Total** | **1,909** | about 1,880 | |

Writing negation/uncertainty/lesion lexicons for these seven languages could roughly double the usable training set (1,960 to about
3,800 studies) and remove the English/Spanish-only site bias that is a prime suspect for the weak-to-gold gap. Proposed order (by size,
and by how regular the report style is): Turkish, Croatian, German, Dutch, French, Greek, Cyrillic. Each language needs its own negation
scope and a native-reader spot check; this is a larger job than the 0.2.0 fixes. Offline LLM extraction (host-permitted) is the alternative.

### Versioning
`rules-en-es-0.2.0` is opt-in (`RuleBasedExtractor("0.2.0")`, `--extractor-profile 0.2.0`). The default profile still reproduces the
production manifest exactly (`b44de700…8818`, verified by hash). The candidate v0.2.0 manifest is at
`outputs/weak-labels/weak_manifest_tier1_v0.2.0.jsonl` (1,960 studies, hash `f7f21a54…3282`, git-ignored). It has not been used for training.

## 2. Model C readiness

`RSNAKneePretrainedMIL` (`mri_core/weak_label_models.py`):
- input `[batch, 3 planes, 9 slices, 1, 128, 128]` with a boolean slice mask; output `[batch, 12]` logits; only 3 targets receive loss
  (the other nine are masked at the loss, verified by a gradient test);
- 2.5D context stacks the previous/current/next valid slice as three channels (edges repeat); ImageNet-style channel normalisation;
- ResNet-18 trunk (torchvision-compatible names), gated Attention-MIL over all plane-slice instances with plane embeddings;
- 11,762,933 parameters (of which the unused 513,000-parameter ImageNet classifier is frozen); optional bidirectional GRU over each plane's
  slice sequence adds 1.2 M (12.96 M total), packed so padding never enters it;
- random initialization is the default; pretrained weights load only from a local file (never downloaded), with strict shape checks.
Tests: shapes, padding invariance (with and without the GRU), parameter counts, gradient routing, exact interrupt-and-resume, no
download fallback.

## 3. Pretrained-weight eligibility (unresolved)
- Participants quote Rule 6.c as requiring external models to be OSI-licensed without commercial restrictions, and sections 2.5, 2.6, 2.8
  on delivery obligations (unverified; I could read only forum threads).
- Unanswered by the host (open threads checked 2026-10-15): torchvision ImageNet-1k weights (745666), RadImageNet ResNet-50, CC-BY-NC(-SA)
  (735121, 745283; the same file is listed as MIT on Hugging Face and CC-BY-NC-SA on a Kaggle dataset), winner-license wording (739905).
- Kaggle's model hub offers no first-party PyTorch ResNet-18 ImageNet file; the ResNet entries are TensorFlow/Keras collections or
  community uploads of unknown provenance.
**Decision:** no external weights are used. C1 stays blocked until the host answers or a clearly OSI-licensed PyTorch ResNet-18 file is
identified. C0 (random init) is fully supported. An in-domain alternative with no license issue is self-supervised pretraining on the
competition's own unlabeled MRIs (C3, below).

## 4. GPU smoke test of Model C (real training studies, Kaggle T4, private kernel `draculess/rsna-weak-label-phase12-smoke`)
Passed for C0 (random init) and C2 (C0 + GRU). Bundle commit `5ccfb27`; manifest hash matched the local reference.

| Measure | C0 | C2 |
|---|---|---|
| Parameters | 11.77 M | 12.96 M |
| 64-study smoke: epoch 1 then resume to epoch 2 | passed (step 7 to 14, 2 sessions in log) | passed |
| Losses | finite, 0 invalid samples | finite, 0 invalid samples |
| AMP scaler scale after training | 65,536 | 65,536 |
| Timed epoch, 199 train / 57 validation studies (5,373 images) | 4.8 s, 1,114 images/s | 5.0 s, 1,083 images/s |
| Peak GPU memory (batch 8) | 1.13 GB | 1.15 GB |
| Data wait per epoch | 0.2-0.3 s | 0.2-0.3 s |
| Exports | hashes match, re-verified locally | same |

Decoding: restored cache hits 64 of 64 studies (0.003 s each); cold decode 1.74 s per study single-threaded (95th percentile 6.0 s) on a
16-study empty cache. Session wall time 818 s (about 0.23 GPU-hours; weekly quota afterwards 1.56 h used of 30 h).
Checkpoints, logs and manifests are saved locally under `outputs/kaggle_smoke/out_p12/` (git-ignored; hashes verified).
The smoke test used only training studies: no gold evaluation, no test images, no pretrained weights, no submission.

## 5. Experimental design (development on training-only validation)

Common settings: Tier 1 targets, masked BCE with training-only class weights, FP16 AMP, early stopping on weak-validation loss
(patience 3, min delta 0.002), 20 epochs maximum, the existing 1,333/368 hash split (identical fingerprint across arms).

| Arm | Description | Status |
|---|---|---|
| C0 | ResNet-18 from scratch, 2.5D, Attention-MIL, 128 px | ready |
| C2 | C0 + bidirectional GRU over slices | ready |
| C0-192 | C0 at 192 px (about 2.3x compute) | ready (config only) |
| C1 | ImageNet-pretrained ResNet-18 | **blocked** by weight eligibility |
| C3 | self-supervised pretraining on unlabeled competition MRIs, then C0 | needs implementation |
| Label control | C0 trained on manifest v0.1.0 vs v0.2.0 | needs the 259 new studies cached |

Three seeds per arm. **Design limitation:** weak-validation loss is computed against the labels of the same manifest version, so it
cannot compare label versions or tell which labels are closer to the experts. Only the gold studies can adjudicate that, which is
why the gold set must stay reserved for one final look at a pre-registered candidate. For the label change itself I recommend a small
hand-labeled audit set drawn from training reports (about 100 reports, by someone qualified), scored for rules v0.1.0 and v0.2.0.

Decision rule fixed in advance: an architecture or label variant advances only if it beats the baseline arm B on weak-validation loss and
on mean per-target weak AUC in at least two of three seeds; the survivors get one paired gold look.

**Compute budget (measured throughput, extrapolated):** an epoch on the full 1,333/368 split is about 6.5x the 256-study epoch, so roughly
30-40 s at 128 px; a 20-epoch run is at most 13 minutes, usually less with early stopping. Nine runs (C0, C2, C0-192 x 3 seeds, the 192 px
runs at about 2.3x) are about 2.5 GPU-hours; the label control adds about 40 minutes; caching the 259 new studies takes a few minutes
with 4 workers. Total about 3-4 GPU-hours against 28.4 hours of remaining weekly quota. Memory is not a constraint (1.1 GB at 128 px).
The cold decode cost (about 0.44 s per study with 4 workers) is paid once per new study.

## 6. Risks and blockers
1. The gold evidence so far shows no signal; label quality and language coverage are the leading hypotheses, not facts.
2. Pretrained-weight eligibility is unresolved (C1 blocked).
3. No labeled audit set exists, so rule v0.2.0's precision is only spot-checked.
4. Spanish ACL and all non-English/Spanish languages remain weak or unlabeled.
5. Competition closes 2026-10-22; the hidden-test inference path (separate notebook) has not been rerun with a new model.

**Recommended next experiment:** build the multilingual (Turkish, Croatian, German first) and Spanish-ACL lexicons as profile 0.3.0,
create the hand-checked audit set, then run the C0/C2 comparison on the improved labels. If more labeled data does not help, stop adding
architecture.
