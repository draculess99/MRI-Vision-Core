# Phase 11: gold-label evaluation of the Phase 10 models, and a Model C proposal

Research prototype; not a diagnostic device. Everything here is private to this project; derived data stays out of Git.

## 1. Gold evaluation (one look, frozen, pre-registered)

Checkpoints (selected on weak-validation loss only): Model A `last_best.pt` `48c58177…` (epoch 5), Model B `last_best.pt` `ae0bf846…`
(epoch 7). Both trained on the same 1,333 weak-labeled studies (validation 368); none of the 58 gold studies, nor any test study,
was in either split (checked from the stored study IDs). Settings fixed before the look: trained targets only, threshold 0.5,
2,000 paired bootstrap rounds over studies, seed 0, same 58 eligible studies for both models (no study was unreadable).
Code: `mri_core/weak_label_gold_report.py`, `scripts/evaluate_weak_label_gold.py`; the ledger refuses a second look at either file.

| Target (gold pos/neg) | A: AUC [95% CI] | B: AUC [95% CI] | A - B [95% CI] |
|---|---|---|---|
| ACL (24/34) | 0.487 [0.34, 0.65] | 0.569 [0.41, 0.72] | -0.082 [-0.29, 0.12] |
| Medial Meniscus (26/32) | 0.578 [0.42, 0.73] | 0.542 [0.39, 0.69] | +0.036 [-0.18, 0.26] |
| Lateral Meniscus (23/35) | 0.645 [0.49, 0.78] | 0.550 [0.39, 0.70] | +0.094 [-0.12, 0.29] |
| **Macro of the three** | **0.570 [0.47, 0.67]** | **0.554 [0.46, 0.65]** | **+0.016 [-0.11, 0.15]** |

At the pre-specified 0.5 threshold, A predicts almost nothing positive for the menisci (sensitivity 0.00, specificity 1.00 for both) and
B predicts mostly positive (Lateral Meniscus sensitivity 0.87, specificity 0.26). The probabilities are not calibrated to the gold
prevalence (class weights shift them), so threshold-based numbers say little; AUC is the fairer summary.

**Reading.** Every confidence interval includes 0.5, and so does every A-B difference. The data show no detectable signal on the
gold studies and no detectable difference between A and B. The earlier local baseline (5-fold out-of-fold on the same 58 studies)
had pooled AUCs of 0.52 / 0.55 / 0.51 for the same three targets, so neither new model is distinguishable from it either.
These three-target gold AUCs must not be compared with the 12-target Kaggle leaderboard metric (public baseline 0.536).

**Weak validation vs gold.** The weak-validation AUCs (0.58-0.69 on 368 studies) measure agreement with rule-extracted labels, not
with expert labels. The drop on gold is the main finding to explain. Candidate causes (none tested yet):
1. rule labels differ from the expert labels (rule recall and precision are modest, see Phase 3);
2. domain shift: the weak training set is 100% English/Spanish reports (likely a limited set of sites) while 20 of the 58 gold
   studies have reports in other languages;
3. small sample (58 studies; CIs about +-0.15).
An exploratory, descriptive split of the stored probabilities (not used for any decision) shows Model A at 0.65 macro AUC on the 38
English/Spanish gold studies and 0.46 on the 20 others (Model B: 0.57 / 0.54). With 20-38 studies this is suggestive at most.

**Integrity notes.** The extractor rules were developed with some exposure to the gold labels (an exploratory prototype was scored on
all 58 studies, then DEV/EVAL halves were used). That is an indirect, mild path by which gold information could have shaped the weak
labels; the models themselves never saw gold studies. Local CPU evaluation used the same preprocessing code as the Kaggle training run
(same commit), but a different pydicom/torch build; pixel-level equivalence was not separately verified.

## 2. What transfers from the first-place solutions (read via the Kaggle forum API)

- **2024 lumbar (NANACHI, 1st):** two stages. Stage 1 predicts slice index and (x, y) coordinates per vertebral level (3D ConvNeXt and 2D
  encoders with level-specific heads, 5-fold, L1 + cross-entropy), using the competition's coordinate labels. Stage 2 classifies crops
  around each level. A community summary of the top teams says almost all used this two-stage crop design, treating each level as
  an extra sample (about 5x more training units), while a couple of single-stage teams (7th, 8th) matched it.
- **2025 aneurysm (tomoon33, 1st):** DICOM to NIfTI, coarse-to-fine nnU-Net (three models) for vessel segmentation and ROI, then a 3D
  classifier on the ROI with the vessel mask as an extra input; about 60 anomalous series removed; multilabel-stratified 5-fold.
  The 13 location classes plus aneurysm presence were predicted jointly; an auxiliary location-segmentation loss and a
  location-aware transformer were used (from a secondary summary of the write-up).

| Component | Transfers to our knee task? | Why |
|---|---|---|
| Pretrained 2D/2.5D encoder | **Yes, the most plausible gain** | Both winners relied on strong pretrained or nnU-Net backbones; our encoder is a 0.1 M-parameter net |
| Attention-MIL over slices | Yes (already in B) | Study-level labels only; B is the attention variant |
| Slice-sequence model (LSTM/transformer over slice embeddings) | Plausible, cheap add-on to C | Community notes use 2D CNN + LSTM |
| ROI localization / crops | **Not now** | The winners had coordinate or mask labels per structure; the knee data has study-level labels only, so localization would need weak localization supervision we do not have |
| Multi-fold ensembles | Yes, late | Cheap here (models train in minutes); only worthwhile once a single model shows signal |
| Label quality and coverage | **Likely the larger lever** | Public knee write-ups report that report-to-label extraction quality dominates; our rules cover only English/Spanish reports (about 56% of studies) |

## 3. Pretrained weights: licensing and rules status (not resolved)

- Participants quote Rule 6.c as requiring external models to be OSI-licensed with no commercial restriction, and sections 2.5, 2.6 and 2.8 on
  external tools and winner delivery. I could read only forum threads, not the rules text, so these quotes are unverified.
- Open questions with **no host answer as of 2026-10-14/15**: eligibility of torchvision ImageNet-1k weights (thread 745666), CC-BY-NC(-SA)
  RadImageNet weights (735121, 745283), and the winner-license wording (739905).
- The host did rule (post 733965) that hosted LLM services are allowed if reasonably accessible and low cost.
- Kaggle's model hub lists only community uploads of ResNet-18 (unknown provenance) plus Google's `google/resnet-v2`, `resnet50` and `bit` collections.

**Consequence.** For leaderboard participation, any publicly available weights are probably acceptable; for prize or winner eligibility
they are not clearly so. Until the host answers, plan Model C so that it works without external weights:
C0 = same architecture, random init; C1 = ImageNet-pretrained, kept as an optional arm; C2 = self-supervised pretraining on the
competition's own unlabeled MRIs (no external license). Do not use CC-BY-NC weights.

## 4. Model C proposal (not implemented beyond what exists; nothing to be trained until approved)

The architecture already exists (`RSNAKneePretrainedMIL`: ResNet-18 trunk, 2.5D previous/current/next slices, gated Attention-MIL with plane
embeddings; weights loadable from a local file). Proposed changes before any run:
1. a bidirectional GRU (or small transformer) over each plane's slice embeddings before attention (slice-sequence modeling);
2. larger inputs (160-224 px) as an option;
3. keep masked BCE, Tier 1 targets, identical split and fingerprint as A and B.

**Ablation plan** (all on the same 1,333/368 split, 3 seeds each, selection on weak-validation loss only):

| Arm | Question answered |
|---|---|
| C0 random-init ResNet-18, 2.5D, attention | Is capacity alone enough? (vs B) |
| C1 = C0 + ImageNet weights | Does pretraining help? (needs a permitted weight source) |
| C0 without the 2.5D context | Value of the neighboring slices |
| C0 with GRU over slices | Value of sequence modeling |
| best of the above at 192 px | Resolution |

Decision rule, fixed in advance: only a candidate whose weak-validation AUC beats B in a majority of seeds goes to a single gold look
(paired bootstrap against B and the baseline).

**Compute (estimates, not measurements).** On this laptop CPU one training step costs about 16x model A (2.0 s vs 0.12 s for 4 studies).
By operation count (about 0.6 GFLOPs per 128-px slice forward, 27 slices per study, roughly 3x for backward) one epoch of 1,333 studies is
about 64 TFLOPs, i.e. roughly 15-40 s per epoch on a T4 once the decode cache exists; 20 epochs about 5-15 minutes per run; 5 arms x 3
seeds roughly 1.5-3 GPU-hours. Peak memory is likely 3-6 GB at 128 px with batch 8 (216 images per batch); use batch 4 if needed.
A 2-minute synthetic GPU smoke test of arm C would replace these guesses with measurements. The 1.5 GB decode cache already exists
in the Phase 10 kernel output and the account has about 28.7 GPU-hours left this week.

## 5. Recommendation

1. Do not simply train A or B longer on the same recipe: weak-validation loss improves, but gold shows no detectable signal.
2. Before more architecture work, diagnose the weak-to-gold gap: label coverage across languages (non-English/Spanish reports are
   unlabeled) and rule label quality are the likeliest causes. An offline multilingual extractor (host-permitted, evaluated on the same
   DEV/EVAL protocol) addresses both.
3. Then run the Model C ablation above, starting with the license-clean C0/C2 arms.
