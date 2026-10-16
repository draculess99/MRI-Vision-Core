# Phase 16 pre-registration: paired real-vs-shuffled weak-label comparison for C0 (prepared, NOT launched)

Research prototype; not a diagnostic device. Design fixed before any run; machine-readable copy in `configs/phase16_registry.json`,
decision rules in `mri_core/weak_label_prereg.py` (unit-tested), kernel driver `scripts/kaggle_phase16_driver.py`.

## Question
Phase 15 showed C0 (128x9) reaches 0.62-0.66 macro AUC against 0.52-0.54 for a shuffled-label control, and only lr 1e-4 beat the
constant-prediction loss floor. That lr was found by inspecting one validation split. Phase 16 asks whether the effect repeats on new
partitions and on studies unused in Phases 14-15, and whether gradient clipping at 5.0 differs from 1.0.

## Design (all fixed in advance)
- Model/data: C0 128x9, FP16 AMP, AdamW, batch 8, weight decay 1e-4, warmup 1, labels v0.1.0, lr 1e-4, **12 epochs, no early stopping**.
- Arms: gradient clip {1.0, 5.0} x labels {real, shuffled}. Shuffled twin: labels and masks permuted jointly across training studies with
  seed 1000+seed; validation and hold-out labels stay true.
- Seeds: 101-105 (never used before; Phases 13-15 used 0 and 1). The seed sets the initialisation, data order and the 80/20 hash split of the
  1,200-study pool; real and shuffled twins share seed, split and initialisation. 5 seeds x 2 clips x 2 labels = **20 runs**.
- Training pool: the Phase 14/15 selection of 1,200 studies (rebuilt in the kernel; SHA-256 must equal the registered value).
- Registered validation partitions: SHA-256 of each seed's sorted validation IDs (258, 243, 248, 249, 256 studies). The driver stops if a run's
  validation set differs. Each is a different 20% of the same 1,200 studies (about 20% overlap with the seed-0 validation set, 17-24% with seed 1).
- Hold-out pool: v0.1.0 manifest studies **outside** the 1,200 (501 before series/readability filters; about 59/136/69 positive for
  ACL/Medial/Lateral). Never trained or validated on in Phases 14-15. Each run's **final-epoch** checkpoint is scored there once; a run is
  refused if any hold-out study was in its training/validation set.
- Fixed reporting epoch: the final one (12). Best-validation-loss epoch is reported descriptively only. No epoch, seed or arm may be chosen after seeing results.

## Decision rules (per clip arm, judged separately)
- C1: validation macro AUC (real) minus its shuffled twin >= 0.10 (about the width of the shuffled chance band seen in Phase 15).
- C2: hold-out macro AUC (real) minus the twin's hold-out AUC >= 0.05 (hold-out has few positives, so its CI is wide, roughly +/-0.08 per run).
- C3: real final validation loss below the constant-prediction floor (p = 0.5) of that split.
- C4 control sanity: mean shuffled validation AUC and mean shuffled hold-out AUC both within [0.45, 0.55], else verdict `invalid_control`.
- **supported** = C1, C2, C3 each in 5/5 seeds; **promising** = each in >= 4/5 (criteria counted separately); otherwise **not_supported**;
  fewer than 5 complete seeds = **incomplete** (descriptive only). Clip arms differ only if the sign of (clip5 - clip1) real validation AUC
  agrees in 5/5 seeds with mean magnitude >= 0.03.
- With n = 5 these are decision rules, not significance tests (5/5 has a one-sided sign-test p of 0.031, 4/5 of 0.19).

## Stopping rules
Abort on non-finite loss; stop after two consecutive run failures; stop on partition mismatch, hash mismatch or any leakage-check failure;
stop and investigate if a shuffled run scores >= 0.60 on both validation and hold-out; no seed block is started unless it can finish within
6,900 s of the 7,200 s kernel limit (blocks are the unit: seeds are never half-finished unless a failure stops the run).

## Leakage prevention and what the validation can and cannot show
Gold (58) and hidden-test studies are excluded by the manifest and re-checked in every run (`leakage.json`) and every hold-out evaluation;
no leaderboard score is consulted or produced; no test images are read; train/validation overlap is checked per run.
**Alternative splits of previously examined data are not an independent external validation set.** The 1,200 studies were used in Phases
14-15 and the lr was chosen on one split of them; fresh partitions only remove the dependence on that one split. The hold-out pool is disjoint
from this experiment's training and validation studies, but it comes from the same source, the same automated labeler and earlier phases'
data (it was inside the larger Phase 9-13 runs), so it is a stricter check, not external validation. Real external validation would need
labels from a different annotator or site; the 58 gold studies stay reserved for a single final look.

## Cost and capacity (estimates from Phase 15 timings)
| Item | Estimate |
|---|---|
| Manifest + decode 1,200 studies | about 11 min |
| Decode hold-out pool (about 500 studies) | about 4 min |
| 20 runs x (12 epochs x 16.5 s + about 25 s start-up) | about 71 min |
| 20 hold-out evaluations | about 8 min |
| **Total** | **about 95-100 min wall (about 1.65 GPU-hours), under the 2-hour cap** |
| Disk | cache about 1.5 GB (outside the output); output about 2.8 GB (10 real runs x 275 MB of `last.pt` + `last_best.pt`); free disk about 20 GB |

Quota: 26.13 h remain; about 24.5 h would remain afterwards. Checkpoints: every run writes a full resumable `last.pt` each epoch
(model, optimizer, scheduler, scaler, RNG, loader RNG, config, history, study IDs); every result carries SHA-256 of its checkpoints, the
driver re-hashes all kept files at the end, and after download I verify them again. Control (shuffled) weights are not kept, only hashes.
If the kernel dies, re-push with the previous output attached (`P16_PREV`): completed runs are skipped and the interrupted run resumes at its last epoch.

## Tests (all passing)
Registry consistency and the verdict rules (supported / promising / not_supported / incomplete / invalid_control / clip comparison);
shuffled-label control, constant baselines, leakage report, gradient diagnostics, CLI `--shuffle-train-labels`, `--val-equals-train`
and `holdout-eval` (including refusal of studies the checkpoint trained on); a CPU dry run of the whole driver with a resume pass.

## Risks
1. 5 seeds can neither confirm nor exclude a modest effect; a hold-out AUC has about +/-0.08 sampling error per run.
2. lr 1e-4 and 12 epochs were suggested by the Phase 15 data; the other arms (clip 5.0) are untested, and clip 1.0 saturated in Phase 15.
3. Weak labels come from one automated labeler and cover only English/Spanish reports; the hold-out shares those biases.
4. Site or acquisition cues correlated with labels would raise real-vs-shuffled AUC without anatomy; this design cannot separate them.
5. Run time could exceed the estimate (Kaggle T4 variation); blocks are skipped rather than truncated, giving `incomplete`.
6. Disk or output-size surprises (about 2.8 GB expected); downloads need `--page-size 200`.
7. A positive result is agreement with automated labels, never clinical accuracy, and would not justify replacing the deployed inference weights.

## Kaggle preparation (added before launch)
- **Bundle**: built from commit `d2fa2b3` (every file hash-verified in the kernel): includes the undefined-AUC handling
  (`final_epoch_record`, `macro_auc`, None-safe verdict), the registry, the hold-out evaluator and all model/data/checkpoint modules.
- **Undefined AUC** is recorded as unavailable (`None`, flagged in `auc_unavailable_runs`); never replaced by 0.5 or 0.0. On Kaggle an
  undefined validation or hold-out AUC stops the run; the verdict counts such a seed pair as missing (arm becomes `incomplete`).
- **GPU smoke test** (`P16_SMOKE=1`, kernel `rsna-weak-label-phase16-smoke`, private, T4, limit 25 min): 120 studies, seed 102, 2 epochs,
  60 hold-out studies, the four arms (clip 1.0/5.0 x real/shuffled) plus an interrupt-at-epoch-1-then-resume run on the FP16/AMP path that must
  reproduce the uninterrupted run exactly (history, model weights, scaler/optimizer/scheduler/RNG states present). Registry-equality checks
  are relaxed in smoke mode only; leakage checks, hold-out disjointness and hash re-verification stay on. Estimate: about 7-9 min (about 0.15 GPU-hours).
- **Full run** (`rsna-weak-label-phase16`, push with a 7,200 s limit): about 95-100 min as tabulated above.
- **Export/persistence**: per-epoch full-state checkpoints in the work directory; each completed run writes `result.json` (metrics, partition
  hash, leakage report, hold-out report, SHA-256 of kept checkpoints); real-run `last.pt` and `last_best.pt` are kept (about 2.8 GB), shuffled
  controls keep hashes only; `final_summary.json` and `prereg_verdict.json` are written after every run so a killed kernel still leaves partial results.
