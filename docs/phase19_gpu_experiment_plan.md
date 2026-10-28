# Phase 19A GPU experiment plan: plain ConvNeXt-T versus the existing CNN baselines (prepared; NOT launched)

Research prototype. Labels are the project's automated weak labels, so every result is descriptive agreement with those labels, not accuracy; no claim is possible without an independent adjudicated reference (Phase 17A protocol, section 5). Nothing here has been run on a GPU.

## Readiness
| Experiment | Ready? | Why |
|---|---|---|
| **D**: plain ConvNeXt-T, ACL + Medial Meniscus, registered seed 201 | **Technically ready; scientifically weak without pretrained weights** | code, tests, driver, bundle and CPU rehearsal done; random initialisation only; pretrained weights need an approval decision |
| **E**: ConvNeXt-T + ROI box pooling | **Not ready (blocked)** | no independent ROI boxes or detector exist; ground-truth boxes do not exist either, so not even an oracle bound can be run; see `phase19_convnext_reproduction.md` |
The driver therefore trains arm D only and records that E was not run. A comparison D versus E must not be launched until an ROI source with recorded provenance is approved.

## Design (fixed before any run)
- Data: the Phase 17A frozen inputs (manifest v0.1.0 `b44de700...`, 1,200-study pool `83a416a0...`, original B0 labels), partition **seed 201** (962 train / 238 validation, hash `b97c2728...`, registry-enforced). Targets ACL and Medial Meniscus; Lateral Meniscus is masked from the loss with reliability weight 0 so that the manifest and splits stay identical to Phase 17A.
  Gold, test and the inspected Phase 16 hold-out are excluded and re-checked; no protected result is used for selection.
- Preprocessing (identical to Phase 17A except resolution): per-slice 1-99 percentile normalisation, nine evenly spaced slices per plane including first and last, bilinear resize to **256 x 256**, plane order Axial / Coronal / Sagittal, one grey channel replicated to three and standardised with (x - 0.45) / 0.225.
- Model D: 29.015 M parameters, drop-path 0.1, no pretrained weights (unless a local ImageNet file with a recorded SHA-256 is approved and attached). Training: AdamW lr 1e-4, weight decay 0.05, cosine with one warm-up epoch, gradient clip 1.0, batch 2 studies (54 slice images), FP16 AMP, 12 epochs, no early stopping, final epoch reported. Recipe is untuned; no search on any validation set.
- Checkpoints (existing resumable format): model, optimizer, scheduler, GradScaler, torch/python/numpy/CUDA/loader RNG, epoch and global step, full config, metrics history, train/validation study IDs, dataset fingerprint (includes manifest hash); written each epoch, resumable at epoch boundaries (resume equality tested on CPU for this model), SHA-256 recorded in the run summary; `last_best.pt` is dropped to save space.
- Controls and context (same validation studies, same labels): metadata-only logistic regression and boosted tree on header features fitted on the same training studies; untrained ConvNeXt-T (ten fixed seeds, forward only) and optionally a label-shuffled twin; the preserved Phase 17A C0 predictions (B0, C1, shuffled) for seed 201 as the "existing CNN baseline for context".
- Statistics: per-target AUC with 1,000-resample study-level bootstrap CIs and paired differences (D minus B0, D minus C1, D minus metadata, D minus untrained) using `mri_core/weak_label_compare.py`; checks: validation hash equals the registry, leakage report clean, checkpoint hashes, finite losses.
- Not done by the driver: any ROI model, gold scoring, test inference, submission.

## Resource estimates (estimates, not measurements on a GPU)
| Item | Estimate |
|---|---|
| Decode 1,200 studies at 256 px x 9 slices (CPU bound) | 11-14 min; cache 4.2 GB (384 px would be about 9 GB, so 256 px only) |
| Training 12 epochs on 962 studies | about 0.5 TFLOP per study (27 slices x 5.9 GFLOP x 3); at 30-40% of a T4's AMP throughput roughly 0.12-0.16 s per study, 2-2.5 min per epoch, **25-30 min** |
| Scoring, metadata baselines, hashing | about 2 min |
| **Total one D run** | **about 45-60 min, 0.8-1.0 GPU-hours** (quota left 28.4 h); a shuffled twin would add about 30 min |
| Peak GPU memory | **unmeasured**: roughly 6-10 GB at batch 2 by activation-size reasoning (batch 4 may not fit a 15 GB T4); a bounded 5-minute GPU smoke should measure it first |
| CPU evidence | one study at 256 px: forward 1.9 s, forward+backward 6.1 s (384 px: 4.3 s / 14.1 s) on this laptop |

## Preparation state
- Driver `scripts/kaggle_phase19_driver.py` (single seed 201 only; stops cleanly with a reason; no pretrained weights unless attached with provenance), kernel inputs = the existing private `rsna-weak-label-code` dataset (a new version containing `mri_core/convnext_roi.py` is needed) and `rsna-weak-label-phase17a-inputs`. No upload and no Kaggle push has been done.
- Rehearsal: the driver was run end to end on the CPU with a tiny ConvNeXt (depths 1-1-1-1, widths 8-64) on the few local studies; it completed (manifest, 32 px cache, two epochs of arm D through the standard CLI, validation scoring, metadata baselines; the AMP scaler key is absent on CPU as expected).
- To launch D later: new code-dataset version, a Kaggle kernel folder from the driver (T4, internet off), explicit approval; to run E: an approved ROI table (`roi-table/1`, independent provenance) and a small in-process training script that wraps the datasets in `RoiDataset` (the loop already forwards boxes).
- Decisions needed from you: (1) allow pretrained ConvNeXt-T weights (source, licence, eligibility) or accept a from-scratch comparison; (2) whether and how an ROI source may be built; (3) approval to run the GPU memory smoke and then arm D.
