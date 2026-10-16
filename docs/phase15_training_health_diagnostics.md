# Phase 15: training-health diagnostics for Model C0 (Kaggle T4, kernel `draculess/rsna-weak-label-phase15`)

Research prototype; not a diagnostic device. The 58 gold studies, the test images and the leaderboard were not used. All AUCs are agreement
with automated weak labels on a training-only validation split, not clinical accuracy.

## Setup
Code commit `f2867cb` (bundle hash-verified in the kernel). C0 (ResNet-18 from scratch, 2.5D, Attention-MIL), 128 px x 9 slices, FP16 AMP,
AdamW, batch 8, labels v0.1.0 (manifest hash matched `b44de700…`), the Phase 14 selection of 1,200 studies and the same hash split
(920 train / 280 validation, seed 0). Dataset fingerprint `955416d9b42948d3a9c110863265d7feb832ad26eb8c4d640c3a8c80e3789c39` is identical to
Phase 14. Leakage report for every run: no duplicates, no train/validation overlap, no gold or test study in either set (the overfit runs
validate on their own training studies by design and say so). 1,200 studies decoded in 646 s, 0 unreadable. Wall time 60.6 min (budget 90);
GPU quota used about 1.0 h. Peak GPU memory 1.13 GB.

New tooling: `--shuffle-train-labels SEED` (labels and masks permuted jointly across training studies; validation labels stay true),
`--val-equals-train`, per-run `baselines.json` and `leakage.json`, and per-epoch gradient-norm, clipping, non-finite-gradient and
validation-prediction-spread records. These add logging only; the training computation is unchanged.

## A. Mechanics and overfit (validation = the 49 training studies)
| Check | Result |
|---|---|
| 64-study overfit, 60 epochs | train loss 0.984 to 0.0008, fit macro AUC 1.000 (passed). Learning only takes off after about 30 epochs; before that predictions are constant (std about 0.001) |
| 199-study overfit, 40 epochs | train loss 1.086 to 0.994, fit AUC 0.68: it begins to fit but is far from memorising in 40 epochs |
| Resume equivalence | run stopped after 5 epochs and resumed (2 sessions) vs uninterrupted: loss histories identical (max difference 0.0 for train and validation) |
| Reproducibility | Phase 14 128x9 configuration re-run: best validation loss 1.0480 (Phase 14: 1.048), best macro AUC 0.6025 (0.603) |
| Non-finite gradient steps | 0-5 per run, AMP loss-scale warm-up skips; no NaN losses |

## B. Constant-prediction baselines (same loss, same split)
Because the training loss is pos-weighted, the best image-blind prediction is p = 0.5 and its validation loss is **1.053** (seed 0 split;
1.143 for the seed-1 split). Predicting the training prevalence is worse (1.411). Any constant predictor has AUC 0.5. So the earlier
"validation loss of about 1.05" in Phase 14 was the constant-predictor floor: those models were not beating a model that ignores the images.

## C/D. Real labels vs shuffled-label control (12 epochs, no early stopping)
| Run | Best val loss (epoch) | Final val loss | Final macro AUC | Max macro AUC | Final prediction std (ACL) |
|---|---|---|---|---|---|
| real, lr 1e-3, seed 0 | 1.052 (1) | 1.167 | 0.639 | 0.640 | 0.18 |
| shuffled, lr 1e-3, seed 0 | 1.050 (2) | 1.052 | 0.540 | 0.560 | 0.000 |
| real, lr 1e-3, seed 1 (other split) | 1.153 (3) | 1.154 | 0.615 | 0.617 | 0.001 |
| shuffled, lr 1e-3, seed 1 | 1.155 (5) | 1.155 | 0.516 | 0.545 | 0.000 |
| real, lr 3e-4, seed 0 | 1.051 (1) | 1.053 | 0.494 | 0.598 | 0.000 |
| real, lr 1e-4, seed 0 | **1.016 (8)** | 1.038 | **0.660** | 0.661 | 0.27 |

Baseline loss to compare with: 1.053 (seed 0), 1.143 (seed 1). Epoch-wise macro AUC, real lr 1e-3 seed 0: 0.587, 0.539, 0.556, 0.528, 0.493,
0.504, 0.485, 0.534, 0.585, 0.627, 0.640, 0.639. Shuffled seed 0 stays between 0.44 and 0.56 for all 12 epochs.

What the diagnostics show:
- **The shuffled-label control behaves like an untrainable problem**: loss pinned at the constant floor, predictions constant (std 0.000),
  AUC in a 0.42-0.56 band around 0.5. The "AUC 0.60 at epoch 1" seen in Phase 14 was not reproduced by the control (0.51).
- **Real labels do separate from the control at the end of training**: final macro AUC is 0.10 higher in both seeds (0.639 vs 0.540; 0.615 vs
  0.516) and 0.12-0.14 higher at lr 1e-4. But the real runs spend the first 8 epochs at 0.49-0.59, inside the control's chance band, and only
  the lr 1e-4 run beats the constant-loss floor (1.016 vs 1.053). With seed 1 the real validation loss (1.154) never goes below the floor (1.143).
- **lr 1e-3 is unstable**: loss spikes (validation loss 1.97 at epoch 4 with mean ACL probability 0.018), and from epoch 10 the
  gradient norm is 3-9 with 60-97% of steps clipped at 1.0, i.e. the update is dominated by clipping. lr 3e-4 collapsed to constant
  predictions; lr 1e-4 trained smoothly (AUC rose monotonically 0.59 to 0.66) but its gradient norms also grew to 16 with 100% clipped, so the
  clipping threshold itself is a further untested suspect.
- A small AUC above 0.5 with prediction spread of about 0.001 (epoch 1) means the AUC is ranking noise-sized differences; it should not be read as learning.

## Interpretation and limits
The pipeline is mechanically sound: masks, loss, optimiser, AMP, checkpoints and resume work, the model can memorise a small set, and a
clean negative control produces chance-level behaviour. The weak-label problem itself appears **learnable only weakly**: 0.62-0.66 final
macro AUC against 0.52-0.54 for the control, with large epoch-to-epoch swings, one seed per lr value, and Phase 13 seed noise of up to 0.13.
The lr 1e-4 result was found by looking at this same validation split, so it is a lead, not a confirmed effect. Possible explanations not
tested: label noise and coverage (1,200 of 4,349 reports, English/Spanish only), site or acquisition cues correlated with labels, optimiser
settings (lr, gradient clipping, BatchNorm with batch 8), too few studies. No clinical claim is made.

## Recommendation (separate experiment; not launched)
Run a pre-registered comparison on fresh splits (new split seeds, so the lr found here is not re-scored on the split that suggested it):
C0 128x9, lr 1e-4, 20 epochs, gradient clip 1.0 vs 5.0, five seeds each of real and shuffled labels = 20 runs, about 5-6 min each plus an
11-min cache build (about 2 GPU-hours). Compare each real run with its shuffled twin and with the constant-loss floor of its split. Continue
only if real beats shuffled by more than the observed shuffled band (about 0.1 AUC) in at least 4 of 5 seeds and beats the floor on loss;
otherwise label coverage and quality, not model or input size, is the bottleneck. The gold studies stay unused.

## Artifacts and hashes (verified after download; weights stay local, git-ignored)
| Run | last.pt SHA-256 | last_best.pt SHA-256 |
|---|---|---|
| overfit64_resume | `7968af1051725f18bacd1f5f0c7c62eb9ace69ccec17ad475ae88a23d6fc2868` | `b2e905acb6f61d112a3c2a24848c316fa7eddd435e36426ccddc1ab087a75277` |
| real_e6_s0 | `a018dddb05e972c560bafd3017abed5f1d2bc879e087ca6e2cdc86b9777dd91d` | `94e525f753dae7ab3f57b3c1bfa712ff1b7fe97cd894e53b2c27d942be3cb2ce` |
| real_e12_s0 | `b5c1f275f34c5fc811a6122f6c4c5b59badb834d89a526a5481965ebc181f511` | `abe8bed0c8e0accc84dd5a3164fd83b6756a7fbf481b40574ae9a7b37ff1953f` |

Controls' weights were not kept (their hashes are in `final_summary.json`, local `outputs/kaggle_smoke/out_p15/`). The three exports passed
their own MANIFEST hash checks. Kaggle GPU quota after the phase: 3.87 h used, 26.13 h remaining (resets 2026-10-17 00:00).
