# Phase 19A: ConvNeXt-T and ROI-box-pooling reproduction (prepared; nothing trained, no Kaggle job, no GPU used)

Research prototype; scores in this project are agreement with automated weak labels unless stated, never clinical accuracy.

## 1. What the winner published, and how much of it I could verify
Reference: the Kaggle write-up "1st Place Solution" ([page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution), dated 2026-10-25, Team Kangaroo).
**I could not read the write-up body.** The page is rendered client-side and my fetch returned only its title. Everything below about the method comes from search-engine summaries of that page and from third-party repositories that cite it
(for example [Bassel1000/rsna-knee-abnormality-detection](https://github.com/Bassel1000/rsna-knee-abnormality-detection), a bronze-medal repository that quotes the method), so it is **secondary and unverified**. **No official code was found** (no Team Kangaroo repository in the results),
so nothing here is "verified published code": the backbone is a published architecture (ConvNeXt, Liu et al. 2022) and every competition-specific component below is my own implementation of a description.

Described (secondary): the competition was treated as a labelling problem (only 58 of 4,407 studies carry image labels); LLM agents produced finding probabilities, **one 3D ROI box per series and finding** (a box around the anatomy the finding concerns, not around the lesion) and per-slice lesion boxes.
Models: a per-slice ConvNeXt-T trained on the 12 labels gave a soft score of 0.8721; adding **ROI box pooling** gave 0.8746; later steps (112 mm crops at the ROIs, **in-box attention**, temperature-calibrated labels, lesion and report heads) reached 0.8789 (the authors' own ablation, on their soft metric).
The efficiency-track entry was a single ConvNeXt-T at 256 px (single EMA checkpoint, detector plus box pooling); the main entry used mainly ConvNeXt-T at 384 px with some differing runs. A third-party repository reports 0.964 public / 0.954 private for the single model and 0.967 for the ensemble (not confirmed from the page).
Not available to me: the exact slice/series aggregation, the detector architecture, preprocessing, training settings and checkpoint structure of the winner.
Note the size of the effect: in the quoted ablation, box pooling added about 0.0025; most of the gain came from the labelling pipeline, which this project cannot reproduce (section 4).

## 2. What was implemented (`mri_core/convnext_roi.py`, tests `tests/test_convnext_roi.py`, 11 tests)
| Component | Implementation | Status |
|---|---|---|
| ConvNeXt-T trunk | from-scratch PyTorch: stem 4x4/4 + LayerNorm2d, stages 3-3-9-3, widths 96-192-384-768, 7x7 depthwise blocks with layer scale 1e-6, stochastic depth, stride 32; **state-dict keys follow torchvision `convnext_tiny`**; trunk parameters 27,820,128 = torchvision's 28,589,128 minus its 769,000-parameter ImageNet head (tested) | published architecture, verified by parameter count and key names |
| Local pretrained loading | `load_trunk_weights(path)` accepts a local torchvision-style file and refuses mismatches; nothing is bundled or downloaded | no weights available (section 5) |
| Arm D (plain) | per-slice trunk features, global average pool, plane embeddings, **target-specific gated attention-MIL over all valid slices of all three planes**, one linear head per target; padded slices provably do not affect outputs (tested) | our design (the winner's aggregation is unknown) |
| Arm E (ROI box pooling) | for the chosen targets (default ACL and Medial Meniscus) each slice's feature map is sampled on a 4x4 bilinear lattice **inside that target's box**, pooled with a learned **in-box spatial attention**, and only slices whose position lies inside the box's z-range enter that target's attention; other targets are unchanged (tested); a missing box falls back to the whole image; `strict_rois=True` refuses to run without boxes | our reading of "one 3D box per series and finding" + "in-box attention" |
| ROI interface | `roi-table/1` JSON with provenance; boxes validated (0 <= x0 < x1 <= 1, same for y, 0 <= z0 <= z1 <= 1); **ground-truth/oracle sources are refused unless `allow_oracle=True`**; `RoiDataset` adds `rois`/`roi_valid` to samples; the training loop forwards them (3 lines; earlier arms untouched) | complete, tested |
| Pipeline integration | arms `D`/`E` reachable from `build_model` without changing `ARMS`; trained end to end through the standard resumable `train_weak_label` (model, optimizer, scheduler, scaler, RNG, config, metrics, study IDs, dataset fingerprint) with resume verified | tested on CPU |
Full-size smoke (random weights, CPU, this laptop): D 29.015 M and E 29.017 M parameters; one study (3 planes x 9 slices) at 256 px forward 1.9 s, forward+backward 6.1 s; at 384 px 4.3 s and 14.1 s; outputs finite. Regression suite 927 passed, 8 skipped.

## 3. Reproduced faithfully vs. different from the winning solution
Faithful (to what is described): ConvNeXt-T backbone; per-slice processing with attention aggregation across slices and series; ROI box pooling per series and finding; in-box attention; 256 px input option.
Different: (1) **no pretrained weights** (the winner almost certainly used ImageNet-pretrained ConvNeXt-T; unverified); (2) **no ROI supervision or detector**; (3) **labels**: rule-based report extraction on two targets, not agent-produced image+report probabilities, no temperature calibration, no lesion/report heads; (4) one grey channel replicated to three; (5) no 112 mm physical-scale crops, no EMA, no TTA, no ensemble; (6) our data pipeline (per-slice 1-99 percentile normalisation, nine evenly spaced slices per plane); (7) training recipe untuned (AdamW lr 1e-4, weight decay 0.05, 12 epochs); (8) evaluation against weak labels on the Phase 17 partition, not the competition metric.

## 4. Is valid ROI supervision available?  **No.**
There are no ROI annotations in the competition data or in this repository, and no detector. The winner's boxes, as described, were produced by LLM agents reading images and reports. This project cannot reproduce that: its standing policy forbids sending report text to hosted models,
no open vision-language model has been approved or verified here, and boxes from any such source would be unvalidated. I also did **not** substitute heuristics (fixed central crops, DICOM-geometry guesses): they would be unverified boxes, and ground-truth boxes do not exist, so even an oracle upper bound is impossible.
Consequence: arm E is implemented and tested but **cannot be given a valid comparison**; the GPU plan covers arm D only.

## 5. Other missing dependencies
Pretrained ConvNeXt-T weights are not on this machine (torchvision and timm are not installed) and downloading them (about 110 MB) was not approved; their eligibility under our standing policy (unverified external weights) is unresolved.
A from-scratch ConvNeXt-T on about 1,000 studies is expected to be weak, so a meaningful comparison needs both an approval decision on pretrained weights and, for E, an ROI source.
