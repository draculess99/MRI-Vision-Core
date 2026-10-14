# Kaggle-native RSNA Knee workflow

Runner: `scripts/kaggle_rsna_runner.py`. It reads the competition from `/kaggle/input` in place and
reuses the unmodified `train_rsna_resumable` / `RSNAKneeCNN` / study-level split code.
`discover` and `inspect` are read-only. Only `benchmark` and `train` train.

## Rules and data handling
- Accept the competition rules on Kaggle first; attach the competition via **Add Input** (never re-upload it).
- Do not copy DICOMs or CSVs into `/kaggle/working`, a dataset, or a public notebook output. `export` only
  bundles `last.pt`, `last_best.pt` and small JSON run files, and its tests assert no `.dcm`/`.csv` in the zip.
- Results are research outputs, not diagnostic.

## Steps to start a real inspection session (nothing trains)
1. Create a new Kaggle notebook. Settings -> Accelerator: choose whatever is offered (GPU optional for
   `inspect`; "None" is fine). Add Input -> the RSNA knee competition.
2. Add the code as a private Kaggle dataset: zip `mri_core/` and `scripts/` (existing `mri_core_kaggle*.zip`
   convention) and attach it. Do not include data or checkpoints you are not permitted to share.
3. Cells:
   ```python
   import sys, glob
   code = glob.glob("/kaggle/input/*/scripts/kaggle_rsna_runner.py")[0]
   !python {code} discover
   !python {code} inspect --max-list 20
   ```
   `discover` finds the folder holding `train.csv` + `train_series.csv` (any name, DICOM folder may sit
   beside or below it) and errors with the candidate list if there is not exactly one. Override with
   `--data-dir/--dicom-root`.
4. Check the output: 12 targets, `complete_label_studies`, `labeled_studies_available_on_disk`, plane
   counts, sampled MB per study, and the detected GPU(s).

## Benchmark (only after approval)
`python {code} benchmark --studies 24 --epochs 2` -> `/kaggle/working/outputs/rsna-knee/benchmark/benchmark.json`
with seconds/epoch and peak CUDA memory (null on CPU). Extrapolate runtime from it before any long run.
Metrics on 24 studies are a smoke test only.

## Training, checkpoints and surviving session loss
- `last.pt` is written after every epoch (model, optimizer, scheduler, scaler, epoch, global step, Python/NumPy/
  torch/CUDA/loader RNG, config, metrics history, exact train/valid study IDs, dataset fingerprint).
  The runner does not pass a scheduler or AMP scaler, so those entries are `None`, as in the existing trainer.
- `/kaggle/working` is ephemeral unless the version is committed. Run long jobs as **Save & Run All**
  (commit) in time-boxed segments: `train --epochs-this-session 3`. The commit's output keeps `last.pt`.
- Next session: attach the previous notebook's output as an input and run
  `train --resume-from /kaggle/input/<previous-output> --epochs-this-session 3`. An existing `last.pt` is never overwritten.
  The saved `selection.json` pins the study list; the trainer refuses to resume if the label fingerprint differs.
- `export` writes `/kaggle/working/export/` (checkpoints, JSON, `MANIFEST.json` with SHA-256, one zip).
  Download via the Output tab, verify the hash with `Get-FileHash`, then follow `docs/kaggle_checkpoint_portability.md`.

## Leakage-free evaluation plan
1. Study-level splits only (`split_labeled_studies_cv`, multilabel-stratified, fixed seed); train and validation
   study IDs are stored in the checkpoint and tested to be disjoint. Positive-class weights come from the training fold only.
2. Per-volume normalisation; no statistics are fit on validation or test data.
3. Choose hyperparameters/epochs on validation folds; report 5-fold mean +- spread, never the best fold.
4. The Kaggle test set is touched only for the final submission and never for selection.
5. RSNA has no patient ID column, so patient-level leakage cannot be ruled out from the metadata; state this limit.
6. Report per-target AUROC with prevalence and flag targets with too few positives instead of averaging them silently.

## GPU and quota notes
- Do not assume an accelerator: the runner prints what `torch` actually sees and falls back to CPU (`device: auto`).
- Accelerator choices, weekly GPU/TPU quota and per-session time limits are shown in the notebook UI and
  change over time; check them before planning. Commit runs can be stopped at the session limit, hence the
  segmented `--epochs-this-session` design and per-epoch checkpoints.
- Start with `batch_size`/`num_workers` from `configs/rsna_knee.json` and let the benchmark's peak-memory
  figure decide whether to raise them.
