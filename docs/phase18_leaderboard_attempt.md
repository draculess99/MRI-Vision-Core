# Phase 18: one controlled leaderboard-improvement attempt (RSNA Knee Abnormality Detection)

Research prototype; scores here are competition metrics, not clinical accuracy. Written before the kernel was launched.

## Competition facts (Kaggle CLI + public pages)
- Competition: `rsna-knee-abnormality-detection` (RSNA Knee Abnormality Detection AI Challenge 2026), deadline **2026-10-22 23:59**, 6,350 teams, our rank 5991, prize pool $77,000 ([RSNA announcement](https://www.rsna.org/news/2026/august/ai-challenge-knee-mri)).
- Task: 12 binary findings per knee MRI study; metric macro ROC-AUC over the 12 targets (the RSNA page itself names no metric; the metric and the code-competition format come from the competition page / our earlier phases and a public project README, [homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)).
  Submission is by notebook (`kaggle competitions submit -k`), re-run on the hidden test (about 1,300 studies, internet off).
- Our best public score: **0.536** (two submissions, 2026-10-02 and 2026-10-04: the 2-checkpoint baseline ensemble `fold0` + `fold1`). No submission has been made since 2026-10-04; the daily allowance (5/day per the competition rules as recorded earlier) is fully available.
- Only 58 of 4,407 training studies have structured labels; 4,349 have only multilingual reports. No external-data or pretrained-weights rule is stated on the RSNA page; our standing policy forbids unverified external weights, so none are used.

## Baseline evaluated on the 58 labelled studies
The 5-fold family of the same architecture (`rsna-knee-ensemble`, MultilabelStratifiedKFold seed 42), re-derived so that every labelled study is scored by the fold checkpoint that did not train on it (fold sizes 12/10/12/11/13 match the checkpoints):
**pooled out-of-fold macro AUC 0.554**, consistent with the public 0.536. The exact submitted checkpoints (`rsna-knee-cv` fold0 and fold1, byte-identical to the Kaggle baseline dataset) can only be scored out-of-fold on their own validation folds (a small subset), so the 5-fold family is the like-for-like reference on all 58.

## Research reviewed (public, approaches only; no code, labels or predictions used)
| Source | Technique | Compatible here? |
|---|---|---|
| [Zenn summary of the RSNA 2024 Lumbar Spine competition](https://zenn.dev/mkj/articles/ba630a8837ee72) (1st: ConvNeXt/EfficientNet-v2 with bi-LSTM and attention-based MIL; 2nd/3rd: attention pooling, pseudo-labels, ensembles of different models, TTA, label cleaning) | multi-view attention-MIL aggregation, 2.5D neighbour-slice inputs, pseudo-labels used with real labels, seed/model ensembles, flip/rotate TTA | MIL aggregation and 2.5D already in C0; pseudo-labels map to our report-derived weak labels; ensembling applies; ROI cropping needs keypoint labels we do not have |
| [RSNA 2025 aneurysm 3rd place slides](https://speakerdeck.com/k951286/kaggle-rsna-intracranial-aneurysm-detectionkonpe-fan-sheng-hui) | two-stage ROI detection on projections, 3D ResNet-18 crops, 11-model ensemble | ROI/3D pipeline not compatible (no localisation labels, knee is not a detection task); ensembling is |
| [RSNA 2024 results](https://www.rsna.org/news/2024/november/2024-ai-challenge-winners) / [2025 results](https://www.rsna.org/news/2025/november/2025-ai-challenge-winners) | ranking only | no technique detail used |
| [homeshwarnelakurthi README](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection) (public project for this competition) | weak supervision from report text because only 58 studies are labelled; grouped folds because random folds inflate AUC through scanner memorisation; reports text labeler 0.79 and a first vision model 0.67 on the 58 labelled studies (its own measurements, metric not stated) | directly relevant to our setting |
No top-10 claims are made for the knee competition: it was still open when searched and no winner write-ups exist. The techniques above come from other RSNA MRI/CT competitions.

## Selected improvement (smallest, evidence-based)
**Extend the report-derived weak supervision from the 3 meniscus/ACL targets to every target with usable class balance (11 of 12; Synovitis is excluded because only 7 of 210 decided labels are negative), and average two seeds.**
Why: the baseline sees 58 labelled studies per fold and is near chance; pseudo/weak labels with ensembling are the documented lever in the reviewed solutions and in the public project for this competition; our C0 (2.5D ResNet-18, attention-MIL over three planes) already learns from these labels (Phases 13-17). Trade-off: the labels are automated and acquisition-linked (Phase 16X), so held-out gold performance may not follow weak-label performance; that is exactly what the gate measures.
Candidate: C0, 128x9, lr 1e-4, 12 epochs, clip 1.0, FP16 AMP (the Phase 16/17A recipe, unchanged), trained on all eligible report-labelled studies (gold and test studies excluded by construction), seeds 701 and 702, mean of the two for the 11 targets; Synovitis from the preserved baseline.

## Pre-registered submission gate (computed locally from retrieved artifacts; `mri_core/phase18_gate.py`)
Gold = the 58 studies the candidate never trained on, scored once. Submit exactly one CSV only if all hold: (G1) paired bootstrap 95% CI lower bound of candidate minus baseline macro AUC > 0; (G2) each seed alone beats the baseline OOF macro AUC; (G3) candidate macro AUC >= 0.58;
(G4) no gold/test study in training or validation, resume check passed, finite values, valid schema; (G5) submissions remaining. If any fails: no submission, artifacts preserved.
Limits: n = 58 gives wide intervals; the gate favours not submitting. A kernel commit-run sees only the visible test sample, so the full-size submission is produced when Kaggle re-runs the notebook (retraining deterministically, about the same recipe and seeds).

## Result (kernel `draculess/rsna-knee-phase18-w11`, version 3 complete; versions 1 and 2 failed on driver bugs)
Version 1 stopped after 2 s (bundle path); version 2 trained both seeds but my resume-check parser misread the output and stopped before scoring; version 3 (same kernel, parsing fixed) completed: 1,823 trainable studies (gold and test excluded), 2 seeds x 12 epochs
(weak-validation macro AUC 0.711 / 0.708), resume checks passed, checkpoints SHA-256 `9d6cbaae...` (seed 701) and `95f51971...` (seed 702), 58 gold studies scored once (never trained on), GPU about 1.2 h.
| | macro AUC on the 58 labelled studies |
|---|---|
| Baseline family, out-of-fold | 0.554 |
| Candidate, seed 701 / seed 702 | 0.607 / 0.608 |
| Candidate, 2-seed mean | **0.618** |
Paired difference +0.063, 95% bootstrap CI **[-0.009, +0.135]** (96% of resamples positive). Largest per-target gains: Effusion 0.88 vs 0.60, Lateral OA 0.70 vs 0.59, ACL 0.66 vs 0.51, Lateral Meniscus 0.65 vs 0.56, PF OA 0.69 vs 0.60; no gain for MCL, Medial Meniscus, Baker's.
Gate: G1 FAILED (CI lower bound -0.009 is not above 0); G2, G3, G4, G5 passed. **No submission was made.** Artifacts are preserved in `outputs/kaggle_smoke/out_p18_w11_v3/` and `outputs/phase18/` (baseline out-of-fold predictions, gate result).
