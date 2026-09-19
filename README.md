# MRI Vision Core — V0.4

Python/OpenCV MRI exploration with a Streamlit UI, Stanford MRNet label integration, and a lightweight CPU predictive baseline.

Research and educational prototype. Not for medical diagnosis or clinical decision-making.

## Milestones

- **V0.1 completed:** Standard image loading, OpenCV preprocessing, threshold segmentation, overlays, and image features.
- **V0.2 completed:** DICOM/NIfTI loading, `MRIVolume`, slice navigation, and MRI-aware preprocessing.
- **V0.3 completed:** Stanford MRNet `.npy` loading, local plane/exam discovery, and UI exploration. The verified V0.3 baseline was **21 passing tests**, including a local MRNet check.
- **V0.4 completed:** Stanford MRNet label integration, validated examination manifests, and a first predictive baseline for abnormality, ACL tear, and meniscal tear. ROI selection is not this milestone.

## Architecture

The Streamlit application (`app.py`) uses the existing slice-processing core:

`loader.py` / `dataset_discovery.py` → `mri_volume.py` → `preprocessing.py` → `segmentation.py` → `features.py` / `visualization.py`, orchestrated by `pipeline.py`.

Supported inputs include DICOM, NIfTI, PNG/JPG, and MRNet NumPy volumes. The UI provides plane/exam selection, slice navigation, technical metadata, Otsu/adaptive masks, and overlays. Predictive modeling runs through a separate CLI:

| Module | Responsibility |
|---|---|
| `mri_core/labels.py` | Strict label parsing, explicit anomaly recovery, task alignment, split validation |
| `mri_core/manifest.py` | ID-based joins, plane availability, separate external image paths, missing/ambiguous image checks |
| `mri_core/baseline_features.py` | Read-only sampled-slice handcrafted features |
| `mri_core/baseline.py` | Train-only fitting, held-out metrics, reproducible CLI and reports |

## Setup and execution

PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Modeling uses scikit-learn (`StandardScaler`, `LogisticRegression`, and `DummyClassifier`); no deep-learning framework is required.

## Tests

The suite collects **74 tests**. The portable default runs **72 synthetic tests**, with **2 optional local MRNet checks skipped**. All 74 passed locally with the dataset enabled before the real baseline was run.

```powershell
# Portable CI: no real dataset required or read.
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider

# Opt-in, read-only integration checks at D:\MRI_DATASETS\MRNet.
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --run-mrnet
```

Synthetic tests cover shuffled label joins, leading zeros, malformed rows, explicit recovery, duplicates, task mismatch, split overlap, missing planes, ambiguous images, deterministic features, invalid volumes, metric calculations, and the CLI. A leakage regression test changes validation features and labels and confirms that fitted scaler statistics, logistic coefficients, and dummy priors do not change. The synthetic CLI test verifies source-file hashes remain unchanged.

## Local MRNet inputs and label validation

The current local dataset is external to the repository:

```text
D:\MRI_DATASETS\MRNet\
  MRNet_ Knee MRI's_files\
    axial\       0000.npy ... 1249.npy
    coronal\     0000.npy ... 1249.npy
    sagittal\    0000.npy ... 1249.npy
  labels\
    train_abnormal.csv
    train_acl.csv
    train_meniscus.csv
    valid_abnormal.csv
    valid_acl.csv
    valid_meniscus.csv
```

Label files are headerless, two-column `exam_id,label` tables. IDs must be exactly four ASCII digits and remain strings, preserving leading zeros. Labels must be exactly `0` or `1`. Each task must contain the same IDs within its split; duplicate IDs, conflicting records, missing tables, malformed or blank rows, and train/validation overlap are rejected. Row order is irrelevant. Split membership comes from the tables, never ID ranges or folder names.

The local CSVs have a known first-row anomaly: `"_0000","_1"` in training abnormal, `"_0000","_0"` in the other training tables, and `"_1130","_0"` in all validation tables. Strict mode rejects these rows. **`--recover-first-row` explicitly permits this exact first-row pattern**, removes one leading underscore from both fields in memory, and records the original and recovered values in console output and `report.json`. It never silently discards a row, treats it as a header, or edits a source CSV. Ordinary headerless tables need no compatibility option.

The resulting manifest has one row per examination with:

```text
exam_id,split,axial_available,coronal_available,sagittal_available,abnormal,acl,meniscus
```

Image paths are resolved separately from all matching plane directories, including nested split layouts. Duplicate ID/plane files are rejected. Missing and unlabeled images are reported; the baseline refuses to fit until the mapping is complete. CSV consumers must read `exam_id` as text to retain leading zeros.

## First predictive baseline

```powershell
.\.venv\Scripts\python.exe -B -m mri_core.baseline `
  --dataset-root D:\MRI_DATASETS\MRNet `
  --labels-dir D:\MRI_DATASETS\MRNet\labels `
  --output-dir outputs\v0.4 `
  --recover-first-row
```

Omit `--recover-first-row` for standard CSVs. `--labels-dir` defaults to `<dataset-root>/labels`. The dataset is optional for installing the project and running CI.

Feature extraction opens external `.npy` arrays with `mmap_mode="r"` and `allow_pickle=False`, using axis 0 as the slice axis. Each plane contributes nine evenly spaced slices including endpoints, or all slices for shorter volumes. Each slice is independently clipped and normalized using its 1st/99th percentiles; a constant slice becomes zero. Sampled slices must be finite, numeric, and at least 2×2.

Ten slice statistics are computed: intensity mean, standard deviation, 10th/50th/90th percentiles, 32-bin entropy, gradient magnitude mean/standard deviation/90th percentile, and fraction of pixels with gradient magnitude above 0.1. Means and standard deviations across sampled slices yield 20 features per plane. Concatenating axial, coronal, and sagittal features gives **60 features per examination**. Normalization uses only the current slice; it learns no population parameters.

For each target, a separate `StandardScaler` → `LogisticRegression` pipeline uses fixed `C=1.0`, `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=2000`, and `random_state=0`. A `DummyClassifier(strategy="prior")` provides a comparison. **Only training examinations are passed to every `.fit()` call**, including scaling, class weights, and dummy priors. Validation is used once for evaluation, without hyperparameter search, calibration, or threshold tuning. All binary metrics use a fixed probability threshold of **0.5**. Nonconvergence stops execution.

Derived artifacts are written under ignored `outputs/v0.4/`:

- `manifest.csv`: labels, splits, and plane availability.
- `features.csv`: examination IDs, splits, and 60 numeric features.
- `report.json`: class distributions, logistic/dummy metrics, recovery records, source label SHA-256 hashes, timings, configuration, feature names, software versions, and limitations.

No MRI arrays are copied into the repository. The CLI rejects output locations overlapping source directories. Feature extraction time covers image reads and feature calculation; training time covers all three scaler/logistic fits and three dummy fits, excluding evaluation and artifact writing.

Metrics include ROC-AUC, average precision, accuracy, precision, recall, F1, and confusion matrices in **`[[TN, FP], [FN, TP]]`** order. Undefined ROC-AUC/AP values are JSON `null` with explanatory notes. Precision/recall/F1 use `zero_division=0`, also reported in notes when applicable.

## Local V0.4 results

The 2026-09-19 run used 1,250 examinations: **1,130 train / 120 validation**, with all three planes available for every exam (3,750 external image files). There were no missing or unlabeled exams and no split overlap. Six first-row recoveries were explicitly recorded; no source labels were rewritten.

| Target | Train negative / positive | Train prevalence | Validation negative / positive | Validation prevalence |
|---|---|---:|---|---:|
| Abnormal | 217 / 913 | 80.80% | 25 / 95 | 79.17% |
| ACL | 922 / 208 | 18.41% | 66 / 54 | 45.00% |
| Meniscus | 733 / 397 | 35.13% | 68 / 52 | 43.33% |

Validation metrics at the fixed 0.5 threshold (AP = average precision):

| Target | Model | ROC-AUC | AP | Accuracy | Precision | Recall | F1 | Confusion matrix |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Abnormal | Logistic | 0.8552 | 0.9365 | 0.8167 | 0.9294 | 0.8316 | 0.8778 | `[[19,6],[16,79]]` |
| Abnormal | Dummy | 0.5000 | 0.7917 | 0.7917 | 0.7917 | 1.0000 | 0.8837 | `[[0,25],[0,95]]` |
| ACL | Logistic | 0.8171 | 0.7877 | 0.7417 | 0.7091 | 0.7222 | 0.7156 | `[[50,16],[15,39]]` |
| ACL | Dummy | 0.5000 | 0.4500 | 0.5500 | 0.0000 | 0.0000 | 0.0000 | `[[66,0],[54,0]]` |
| Meniscus | Logistic | 0.7240 | 0.6239 | 0.6667 | 0.5909 | 0.7500 | 0.6610 | `[[41,27],[13,39]]` |
| Meniscus | Dummy | 0.5000 | 0.4333 | 0.5667 | 0.0000 | 0.0000 | 0.0000 | `[[68,0],[52,0]]` |

The ACL and meniscus dummy models predict no positives; their precision is set to zero with an explicit note in the report. Logistic regression improves ranking metrics and accuracy for all three targets, but the always-positive abnormality dummy has slightly higher F1. No settings were adjusted after observing these results.

Feature extraction took **179.205 seconds**; training the three scaled logistic models and three dummy models took **0.106 seconds** on this machine. These are single-run wall-clock measurements, not benchmark guarantees. The run used Python 3.12.10, NumPy 2.5.3, and scikit-learn 1.9.1. Exact metrics and provenance are in the locally generated, ignored `outputs/v0.4/report.json`.

## Limitations

- Handcrafted slice summaries do not localize tears or provide diagnostic evidence; sampled slices may miss focal findings.
- Per-slice normalization removes absolute intensity scale. Planes are concatenated, not spatially registered.
- The validation set has only 120 examinations; there are no confidence intervals or independent external test results.
- ACL prevalence differs substantially between training and validation. Fixed balanced weights do not establish calibrated probabilities.
- Examination IDs are separated across splits; this does not verify patient-level independence.
- The UI's threshold segmentation remains separate from the predictive baseline and is not medically validated.

Future work should evaluate stronger features and robustness without repeatedly tuning against this validation split. ROI tools, deep segmentation, volumetric radiomics, and an API remain possible later milestones.
