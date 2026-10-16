# Phase 18 protocol: read-only evaluation of preserved C0-family checkpoints on MRNet (prepared; NOT run)

Research prototype; not a diagnostic device. Written before any MRNet metric has been computed. Code: `mri_core/mrnet_eval.py`, `scripts/evaluate_mrnet_readonly.py`, tests `tests/test_mrnet_eval.py`.
No training, no model or dataset modification, no leaderboard, no gold studies, no Phase 16 hold-out, no upload.

## Reference
MRNet (Stanford) knee MRI: 1,250 exams (1,130 train + 120 valid), three planes each, labels abnormal / ACL tear / meniscus tear. Per its publication the labels are clinician-derived (not our extractor, not an LLM); this is to be confirmed
against the dataset documentation, which is not present in the local folder (no README or licence file). Different institution and identifier scheme from RSNA; no shared studies are known but this cannot be verified from the files.
Positives: ACL 208 + 54, meniscus 397 + 52 (train + valid). The C0 family (B0, C1, Phase 15/16 checkpoints; random initialisation) never trained on MRNet. The older MRNet-pretrained RSNAKneeCNN lineage did (train) and was selected on valid, so it is out of scope.

## Endpoints and models (fixed now)
- ACL: the ACL output. Meniscus: the larger of the Medial Meniscus and Lateral Meniscus outputs (MRNet has one meniscus label). Medial and Lateral are reported separately against the meniscus label as descriptive context only.
- Models: B0 and C1 from Phase 17A seed 201 (SHA-256 recorded in `result.json`); optionally the Phase 16 seed-101 and Phase 15 real-label checkpoints, reported separately.
- Preprocessing: the RSNA recipe applied to MRNet NumPy volumes (per-slice 1-99 percentile normalisation, up to 9 evenly spaced slices including first and last, bilinear resize to 128, plane order Axial, Coronal, Sagittal; test-verified against an independent recomputation).
  MRNet slice order is not guaranteed anatomical; its sequences (fat-suppressed PD axial, T1 coronal, fat-suppressed T2 sagittal) differ from RSNA's fluid-sensitive preference, so scores are cross-domain transfer.

## Controls and statistics
- Metadata-only: logistic regression and a small boosted tree on six acquisition features (slice count per plane, mean intensity per plane); 5-fold out-of-fold on all 1,250 exams (primary) and fitted on the train split / scored on valid (secondary).
- Untrained C0: ten fixed seeds (9001-9010), forward pass only. The Phase 17A shuffled-label weights were not kept, so no shuffled control is available (stated limitation).
- Per endpoint and mean of the two: AUC with 95% study-level bootstrap CI (1,000 resamples); paired differences C1 - B0, each checkpoint - metadata (both models) and - untrained mean, on the same resamples. Primary set: all 1,250 exams; secondary: the 120 valid exams.
- Throughput (forward-only exams/s and images/s) and total times are recorded; checkpoint, predictions and summary SHA-256 are recorded, and the dataset file signature plus checkpoint hashes are re-checked after the run (`read_only_verified`).

## Interpretation rules (descriptive; no p-values)
A checkpoint shows external transfer only if both its ACL and its meniscus AUC CIs exclude 0.5 and its mean AUC exceeds the metadata-only and untrained-mean arms with a paired-difference CI above 0. A result at or below the metadata control means no evidence that the model uses more than acquisition cues.
A positive result states "transfer performance on MRNet", not RSNA accuracy and not clinical accuracy; a null or low result cannot separate model failure from domain shift. B0 versus C1 is a comparison of two training label masks only.

## Preflight (done, labels not used)
24 exams scored on CPU with the real B0 and C1 checkpoints: tensor [3, 9, 1, 128, 128], probabilities finite, about 150 images/s (5.5 exams/s) per model on this Dell CPU. No AUC was computed in preflight. Full CPU run: about 4-5 min per checkpoint
plus about 4 min per untrained seed (ten seeds about 40 min; five seeds about 20 min); a T4 would need roughly 1-2 min per pass.

## Blockers and approvals needed
1. MRNet licence and label provenance unconfirmed (no documentation locally). 2. A Kaggle GPU run needs MRNet (7.3 GB) and the two checkpoints (137 MB each) uploaded as private datasets - not approved and not done; the prepared driver
(`outputs/kaggle_smoke/p18_mrnet_eval_BLOCKED`) stops cleanly if they are absent. 3. The CPU run needs no upload and is the practical route.
