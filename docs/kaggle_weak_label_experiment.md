# Kaggle weak-label experiment (Tier 1: ACL, Medial Meniscus, Lateral Meniscus)

Research prototype; not a diagnostic device. Nothing here trains until you run a `train` step in a Kaggle notebook.

## What is compared
| Arm | Model | Notes |
|---|---|---|
| A | `RSNAKneeCNN` (unchanged) | max-pool over slices; ~0.10 M parameters |
| B | `RSNAKneeAttnCNN` | same encoder, gated-attention pooling per plane; ~0.12 M |
| C | `RSNAKneePretrainedMIL` | ResNet-18 trunk (standard layout: 122 state-dict tensors, 11,689,512 parameters, checked in tests), 2.5D context, gated Attention-MIL; ~11.8 M. Loads torchvision-style ImageNet weights from a local file; random init otherwise |

Same pixels for every arm, masked BCE (normalised by the summed mask x reliability weight of the batch), `pos_weight` from the
training mask only (clipped to 10), AdamW + warmup/cosine, a full-state checkpoint after every epoch.

## Competition rules status (checked 2026-10-15 through the Kaggle API: `kaggle competitions topics list/show`)
| Question | Status | Source |
|---|---|---|
| Commercially hosted LLMs / sending report text to external APIs | **Verified permitted** by the host, if the service meets Rule 2.6.b (reasonably accessible, minimal cost); this is not "private sharing" | Host post 733965, 2026-08-09 |
| Report-derived (weak) labels | **Consistent with the host's description** ("models may learn from both the images and the report text"; labels from reports are explicitly contemplated) but no explicit rule text read | Overview post 733343 |
| Offline open-weights LLM inside a notebook | Follows from the above; not separately confirmed | inference |
| Private derived dataset (manifest) | **Unresolved**. The host defines PRIVATE SHARING as sharing with other participants/teams/third parties; a dataset kept private to our own team is not that, but nothing explicitly confirms it | host post 733965 |
| Pretrained weights / winner license | **Unresolved**. The winner license text ("CC BY-NC 4.0" vs. an OSI-approved license) is being questioned in thread 739905 with no host answer as of 2026-10-14; weights of unknown license should not be used | thread 739905 |
| External datasets | Case-by-case host rulings (KneeCoT prohibited; click-through agreements generally acceptable, institution-specific approvals not). We use none | thread 743416 |
| Code competition / hidden test size / runtime | Hidden test is about 1,300 studies; notebooks are re-run privately; 9 h and internet-off come from secondary sources only | threads 744230, 749623 |
| T4/P100 have no bf16 | **Community-verified hazard** (bf16 emulated ~50x slower, silently degraded a score). Our AMP path uses float16 on CUDA only | thread 744230 |
| Submissions | 5 per day; we have 5 remaining today | `kaggle competitions submission-limits` |

Our own constraint (separate from the rules): no radiology report text is sent to any external service.

## Data rules enforced in code
- The manifest excludes all 58 gold studies and test/sample-submission IDs; `train` re-checks this and stores the exact
  train/validation IDs and a dataset fingerprint (manifest hash + the study set actually used) in every checkpoint.
- Tier 2/3 targets keep their extracted state in the manifest but are masked (`tier_masked`).
- Checkpoint selection uses weak-label validation loss only. `evaluate-gold` refuses a checkpoint whose training IDs include a
  gold study and allows one look per checkpoint hash (ledger).
- Studies that fail to decode are masked, counted, and listed by ID in `invalid_studies.json`; the run aborts above 5 %.
  Missing series found by the pre-flight are listed in `preflight_excluded.json`.
- The decode cache is float16, versioned, atomically written, and any unreadable or wrong-shape entry is re-decoded.
- Submission: only approved targets (default Tier 1) can come from the new model; every other target is the baseline value,
  and every fallback study is listed in `<output>.audit.json` together with checkpoint hashes and seconds per study.

## Kaggle inputs (all offline; add each via *Add Input*)
1. The RSNA Knee competition (accept the rules first).
2. Private dataset `rsna-weak-label-code` = `kaggle_upload/rsna_weak_label_code.zip` (source only, no data or checkpoints).
3. Private dataset `rsna-weak-manifest-tier1` = `weak_manifest_tier1_v0.1.0.jsonl` + `.summary.json` (study IDs and report-derived
   states; no report text). See the "private derived dataset" row above.
4. Baseline checkpoints dataset (submission only; for example the two folds in `kaggle_rsna_baseline_checkpoints_v2.zip`).
5. Arm C only: an ImageNet ResNet-18 state dict with a verified license. **Not sourced; do not run arm C until it is.**

## Exact instructions
Setup cell (run once per notebook):
```python
import glob, zipfile, pathlib, subprocess
dest = pathlib.Path("/kaggle/working/code")
z = glob.glob("/kaggle/input/**/rsna_weak_label_code.zip", recursive=True)
if z: zipfile.ZipFile(z[0]).extractall(dest)
else:
    import shutil; shutil.copytree(pathlib.Path(glob.glob("/kaggle/input/*/scripts")[0]).parent, dest, dirs_exist_ok=True)
MANIFEST = glob.glob("/kaggle/input/**/weak_manifest_tier1_v0.1.0.jsonl", recursive=True)[0]
RUN = f"python {dest}/scripts/kaggle_weak_label_experiment.py"
CFG = f"--config {dest}/configs/weak_label_tier1.json --manifest {MANIFEST}"
```
**Step 1: environment and data pre-flight (read-only, no training).** Notebook settings: Accelerator = GPU (T4 or whatever is offered),
Internet = Off, Persistence = Files.
```
!nvidia-smi
!{RUN} preflight {CFG}
```
Expected: 1,701 studies, active targets ACL / Medial Meniscus / Lateral Meniscus, `gold_overlap 0`, `studies_with_series_problems`
near 0, the GPU listed.

**Step 2: synthetic CUDA/AMP smoke test (about 1-3 minutes, no data).**
```
!{RUN} smoke --work-dir /kaggle/working/smoke --arms A B
```
Pass = every check true for each arm: interruption mid-epoch, resume from epoch 1, finite losses, AMP scaler state restored,
durable copy identical. It also reports peak GPU memory, studies/second and the GPU model. Add `C` to `--arms` only once weights exist
(the smoke test uses random weights and is fine for C, but C's real run needs approved weights).

**Step 3: one-epoch real-data smoke test (only after approval).** 64 studies, one epoch, then rerun to prove resume and export:
```
!{RUN} train {CFG} --arm A --epochs-override 1 --max-studies 64 --work-dir /kaggle/working/real_smoke_A --export-dir /kaggle/working/export_smoke
!{RUN} train {CFG} --arm A --epochs-override 1 --max-studies 64 --work-dir /kaggle/working/real_smoke_A   # no-op resume
!{RUN} export --work-dir /kaggle/working/real_smoke_A --export-dir /kaggle/working/export_smoke_final
```
Read `real_smoke_A/runtime_log.jsonl`: seconds per epoch, peak GPU GB. Extrapolate to 1,701 studies before any real run.
To test interruption on real data, stop the cell part-way through (Interrupt) and rerun the same command; it resumes from the last
completed epoch (a partial epoch is redone).

**Step 4 (after review, not part of the smoke test):** the full per-arm runs use the same command without `--max-studies`, with
`--max-seconds` to leave headroom before the session limit, followed by `evaluate-gold` and `export`.

**Quota:** Kaggle does not expose the remaining GPU quota to code. Read it from the notebook UI before and after each run and
compare with `summarize_runtime` / `runtime_log.jsonl` (the smoke report prints the same figures).

## Estimated GPU resources (estimates, not measurements on Kaggle)
- Manifest: 1,701 studies (about 1,530 train / 170 validation); a 64-study smoke test reads about 10 GB.
- Decode: about 0.5 s per study per CPU core (measured locally on 6 studies); cache entries about 0.9 MB (about 1.5 GB total).
- Arm C is about 16x the per-step CPU compute of arm A (2.0 s vs 0.12 s for 4 studies on this laptop); on a GPU all arms should be
  dominated by decoding until the cache is warm. Peak GPU memory is unmeasured until the Step 2 report.
- Submission inference over about 1,300 test studies: roughly 10-15 minutes of single-core decoding plus negligible model time
  (check `seconds_per_study` in the audit file; the T4 bf16 hazard above is why autocast is float16 only).

## Observed on Kaggle (synthetic GPU smoke test, 2026-10-15, kernel `draculess/rsna-weak-label-smoke`, commit a83b98c)
- Environment: 2x Tesla T4 (15,360 MiB each, capability 7.5), driver 580.178.04, Python 3.13.15, torch 2.11.0+cu128, cuDNN 9.19.
  Training used one GPU (`cuda:0`). `torch.cuda.is_bf16_supported()` returns True on the T4 although the hardware has no bf16
  units, so it is not a usable guard; autocast is hard-wired to float16 on CUDA.
- The Kaggle image has no `iterative-stratification` and internet is off. The first runs failed on that import (fixed by importing it
  lazily). Kaggle also auto-extracts uploaded zip files, so the driver accepts either the zip or the extracted tree and verifies
  every file hash against the verified bundle.
- Arms A and B: all checks passed (interrupt inside epoch 2 -> restored epoch 1 / step 3 -> completed; AMP float16 with scaler
  state restored, final scale 65536; exports re-hashed identical; checkpoints load on CPU). The resumed model was bitwise equal to an
  uninterrupted run on this hardware (informational on CUDA). Peak GPU memory 0.324 GB at the default synthetic size; throughput
  is not representative of real DICOM decoding.

## Not verified
Real-data CUDA execution and Kaggle I/O throughput (only the synthetic smoke test has run on a GPU), the exact GPU offered,
official ImageNet weights, numerical equality with torchvision's ResNet-18 (layout and key names are tested, outputs are not), and
the unresolved rules rows above.
