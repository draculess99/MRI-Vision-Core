# Phase 14: resolution and slice-count ablation (Kaggle T4, kernel `draculess/rsna-weak-label-phase14`)

Research prototype; not a diagnostic device. The 58 gold studies were not evaluated, consulted or used. No leaderboard submission was made.
All metrics are agreement with automated weak labels on a training-only validation split, not clinical accuracy.

## 1. Preserved
Clean branch `hackathon/opencv-aws`, Models A/B/C and earlier checkpoints, manifests v0.1.0 (`b44de700…8818`, production) and corrected
profiles, the public submission (0.536) and the AMD project were not touched. The default preprocessing is unchanged. The Phase 13 checkpoint
that was still on Kaggle (`v022_C2_s0`) was recovered and its SHA-256 verified before this phase started.

## 2. Preprocessing verification (`tests/test_weak_label_resolution.py`)
For each of 128/9, 192/9, 128/15, 192/15 the production `RSNAKneeDicomDataset` is tested on synthetic DICOM studies for: tensor shape
`[3 planes, S, 1, N, N]`; plane assignment from `Anatomical_Plane`; slice order (strictly increasing along the position-validated order,
recomputed independently and compared with `torch.equal`); masks (a 12-slice coronal series gives 12 valid slots at S=15, padding is exactly
zero); finite values in [0, 1]; all three model arms accept the shape. Normalisation is **per slice** (1-99 percentile) and therefore does not
depend on resolution or slice count.

To avoid decoding every study four times, `mri_core/weak_label_multicache.py` reads each plane once and derives all four variants with the
same operations as production. Tests assert bitwise equality with the production pipeline for every variant, and that its cache entries are
hits for the normal training dataset. One production-style observation (not changed): slice sampling always includes the first and last slice,
which at S=9 can include end-of-stack slices with little anatomy.

## 3. Data and fingerprint
- Manifest v0.1.0 regenerated in Kaggle; matches the local reference hash. Bundle commit `e20ef14`.
- 1,200 studies selected deterministically and stratified by class (`--smoke-subset 1200 --min-per-class 60`); 0 unreadable, 1,200 decoded.
- Split (hash-based, seed 0, identical for all four runs): 920 train, 280 validation.
- Class counts (positive/negative) train: ACL 106/685, Medial Meniscus 260/418, Lateral Meniscus 120/657; validation: ACL 28/218,
  Medial Meniscus 72/132, Lateral Meniscus 36/203.
- Dataset fingerprint (identical in all runs): `955416d9b42948d3a9c110863265d7feb832ad26eb8c4d640c3a8c80e3789c39`.
- One decode of all 1,200 studies: 582 s (0.485 s/study, 4 workers); cache sizes 1.06 / 2.39 / 1.77 / 3.98 GB for 128x9 / 192x9 / 128x15 /
  192x15 (9.2 GB total); 11.7 GB of disk remained.

## 4. Feasibility (64-study subset, 49 train / 15 val, 2 epochs; stop after epoch 1, resume, export)
All four passed: forward/backward with masked loss and FP16 AMP, epoch-boundary resume (epoch 1 to 2, step counter continued), checkpoint
export and reload; GradScaler stayed at 65,536; 0 invalid studies. Peak GPU memory 1.13 / 2.33 / 1.76 / 3.72 GB. Checkpoint 137.2 MB each.

## 5. Controlled comparison (C0, labels v0.1.0, seed 0, same split, 6 epochs, no early stopping)
Only `image_size` and `slices_per_series` varied. Selected epoch = lowest weak-validation loss.

| Config | Best epoch | Val loss | ACL AUC | Med. Men. AUC | Lat. Men. AUC | Macro AUC | Epoch s | Imgs/s | Peak GPU GB | Run s |
|---|---|---|---|---|---|---|---|---|---|---|
| 128x9 | 5 | 1.048 | 0.617 | 0.592 | 0.598 | **0.603** | 15.1 | 1,645 | 1.13 | 111 |
| 192x9 | 1 | 1.051 | 0.620 | 0.572 | 0.601 | **0.598** | 33.5 | 740 | 2.33 | 229 |
| 128x15 | 5 | 1.024 | 0.646 | 0.603 | 0.568 | **0.606** | 24.6 | 1,683 | 1.76 | 172 |
| 192x15 | 1 | 1.051 | 0.615 | 0.575 | 0.601 | **0.597** | 55.7 | 742 | 3.72 | 368 |

Differences from 128x9 (macro AUC): 192x9 -0.004, 128x15 +0.003, 192x15 -0.005. Differences in best validation loss: +0.003, -0.024, +0.003.

Epoch-to-epoch behaviour (macro AUC by epoch 1..6): 128x9 0.587 0.539 0.499 0.577 0.603 0.623; 192x9 0.598 0.558 0.500 0.503 0.501 0.479;
128x15 0.519 0.562 0.544 0.584 0.606 0.626; 192x15 0.597 0.619 0.558 0.637 0.650 0.655. Validation loss by epoch stays at about 1.05 in
nearly every epoch of every run (range 1.02-1.27); for 192x9 the six epochs are flat at 1.051-1.054 and AUC falls to 0.48 (about chance), and
192x15 validation loss rises to 1.21 by epoch 6 while AUC rises. No decoding or invalid-sample failures in any run (0 invalid).

## 6. Reading
- The four configurations are indistinguishable at the selected epoch: macro AUC 0.597-0.606, a spread of 0.009. Within a single run the macro
  AUC moves by 0.10-0.12 between consecutive epochs, and Phase 13 measured up to 0.13 between seeds, so a 0.009 spread cannot be called a
  difference. The best-epoch pick is itself unstable: two of four runs selected epoch 1, i.e. essentially before training, and those two
  epoch-1 values (0.598, 0.597) are no different from the later-epoch values of the other runs.
- Validation loss barely moves from epoch 1 in any run, and loss and AUC disagree at the end of training (192x15: AUC rising, loss rising).
  I have not established why; candidates (not tested) are an under-tuned optimiser/learning rate for this model on 920 studies, loss
  dominated by pos_weight, or too little signal in the labels. This is a bigger issue than resolution.
- Cost scales sharply: 192 px takes 2.2x the epoch time and 2.1x the memory of 128 px (192x15: 3.7x time, 3.3x memory) with no measurable
  benefit; 15 slices at 128 px costs 1.6x time for +0.003 AUC.
- Preprocessing: no inconsistency found. Caveats: normalisation is per slice (not per volume); 5 of 64 series flagged coronal in metadata are
  oblique in DICOM (Phase 8), unchanged; 15 slots exceed the coronal series length in some studies (masked as padding).
- One seed per configuration, 6 epochs, 920/280 split: this is a feasibility-grade comparison, not an estimate of effect sizes.

## 7. Decision against the stated criteria
| Criterion | Assessment |
|---|---|
| Reliable | No: differences are smaller than epoch-to-epoch and seed-to-seed variation |
| Fits GPU budget | Yes (whole phase 0.62 h wall; 3.7 GB peak at the largest) |
| Promising improvement | No: macro AUC spread 0.009; best-loss change -0.024 at 128x15 comes with a 1.271 loss spike in epoch 4 |
| No preprocessing inconsistency | Yes |

**Recommendation:** do not launch a multi-seed resolution/slice sweep; resolution and slice count are not the bottleneck on this evidence.
The more informative next step (cheap, one GPU run of minutes) is a training-health diagnostic at 128x9: (a) can C0 drive the training loss
down on a 64-study subset (overfit test), and (b) validation loss against a constant-prediction baseline, learning-rate and epoch sensitivity,
and shuffled-label control. If the model cannot fit or beat the controls, label coverage and optimisation, not input size, are the
constraint. Pretrained weights remain unavailable pending the unresolved rules question.

## 8. Artifacts and release checks
- Verified SHA-256 for `last_best.pt` of each comparison run (downloaded; match `final_summary.json`): 128x9 `e3fcb278…`, 192x9 `1b679d7f…`,
  128x15 `dda95cc3…`, 192x15 `268727b3…` (137.3 MB each). Local copies and the summary are in git-ignored `outputs/kaggle_smoke/out_p14*`.
- Kaggle GPU quota after the phase: 2.86 h used, 27.14 h remaining of 30 h (resets 2026-10-17 00:00).
- Download tip: `kaggle kernels output` needs `--page-size 200` for outputs with thousands of files; use `--file-pattern`.
