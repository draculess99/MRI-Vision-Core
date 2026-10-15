# Weak-label pipeline: radiology reports -> masked training targets

Research prototype. Report-derived labels are noisy, are not clinical ground truth, and must not be used for
patient care. Nothing here trains a model, calls an external API, or changes an existing checkpoint.

## Modules

| Module | Role |
|---|---|
| `mri_core/report_labels.py` | Contracts: `LabelState` (positive / negative / uncertain / not_mentioned), `TargetLabel`, `ReportLabels`, `to_training_arrays` (values + loss mask), `ReportLabelExtractor` protocol |
| `mri_core/report_label_language.py` | Conservative language ID; only `en` and `es` are supported, everything else is left unclassified |
| `mri_core/report_label_rules.py` | `RuleBasedExtractor` (version `rules-en-es-0.1.0`) |
| `mri_core/report_label_eval.py` | Gold-label evaluation harness with DEV/EVAL split and a one-look ledger |
| `mri_core/weak_label_manifest.py` | Versioned, deterministic manifest builder and loader |
| `scripts/build_weak_label_manifest.py`, `scripts/evaluate_report_labels.py`, `scripts/record_baseline_inventory.py` | Thin CLIs |

The extractor interface is `extract(report_text, study_uid=None) -> ReportLabels`. A later offline multilingual
model plugs in by implementing the same method; the manifest builder and harness do not change.

## Label semantics

- `positive` / `negative` are the only states that ever produce a training target (mask 1).
- `uncertain` covers hedged, historical or post-surgical mentions, trace/mild severity, degenerative context,
  location-ambiguous mentions, "not visualised", and conflicting mentions. Masked by default.
- `not_mentioned` means nothing usable was said. It is **never** turned into 0. Reports in an unsupported language
  produce `not_mentioned` for every target with `reason=unsupported_language`, `supported=false`.
- Masked values are stored as `null` / `NaN`, not 0. `to_training_arrays(..., uncertain="soft")` is an explicit opt-in.

## Operational target definitions (proposed; the official per-finding definitions could not be retrieved)

ACL / MCL: tear or rupture (sprain, laxity, thickening, "lesion" -> uncertain). Medial / Lateral Meniscus: tear
(degeneration, signal change, extrusion, truncation -> uncertain). Medial / Lateral / PF OA: cartilage loss,
chondromalacia, osteophytes, arthrosis or degenerative joint change *in that compartment* (mild/early -> uncertain;
location-free statements -> uncertain for all three; an explicit "no OA / cartilage normal" is applied to every
compartment it covers). Effusion / Synovitis / Baker's: present unless negated (trace/minimal -> uncertain for
all three; bursal effusions are not joint effusion). Contusion: bone contusion or marrow edema (subchondral or
degenerative context -> uncertain). Fracture: any fracture (old/healed/fixation -> uncertain).

## Extraction rules in brief

Indication/history/technique/comparison text is ignored. Sentences are split into clauses (but/however/while) and
then into pieces only where both neighbours name a target structure, so negation and lesion words cannot cross
targets. Bare list items inherit negation from the previous piece. Per target, specific mentions override generic
ones; positive vs negative, or any hedge mixed with a decided state, becomes `uncertain`.

## Evaluation protocol (what counts as evidence)

- The 58 gold studies are split by hashing the study UID (`SPLIT_SALT`) into a DEV half (31) and an EVAL half (27).
  Change rules while looking only at DEV (`--split dev`).
- EVAL can be scored once per `extractor.version` (recorded in a ledger); a second request is refused unless
  forced, and forced runs are labelled in the ledger. Bump `EXTRACTOR_VERSION` after any rule change.
- The earlier scratchpad prototype was scored on all 58 studies, so the EVAL half is **lightly contaminated** and
  is not an independent test set. Treat all 58-study numbers as exploratory.
- Metrics per target and language group: positive precision/recall, negative reliability and recall, coverage,
  4-state confusion counts, AUROC (only with >= 3 per class), percentile-bootstrap 95% CIs over studies.
  With 27-31 studies the intervals are wide; do not rank extractors on small differences.

## Leakage controls

- A study with *any* structured target is treated as gold and excluded from weak-label training (this removes all 58).
- Kaggle test / sample-submission IDs and any caller-supplied hold-out IDs are excluded; an assertion re-checks the
  final manifest. Fold validation studies must be passed as `--holdout-csv`.
- `train.csv`, DICOMs and checkpoints are only read. The CLI refuses to write over an input file, writes only
  `.jsonl` plus a `.summary.json`, and will not overwrite without `--overwrite`.
- Logs contain counts only (no report text, no full UIDs). Evidence sentences are not stored in manifests unless
  `--include-evidence` is given; otherwise only their SHA-256 is.
- Patient-level overlap cannot be checked: `PatientID` is a unique hash per study and dates are scrubbed.

## Usage

```
python scripts/record_baseline_inventory.py --output outputs/weak-labels/baseline_inventory_pre.json
python scripts/evaluate_report_labels.py --train-csv data/rsna-knee/train.csv --split dev
python scripts/build_weak_label_manifest.py --train-csv data/rsna-knee/train.csv --test-csv data/rsna-knee/test.csv \
       --sample-submission data/rsna-knee/sample_submission.csv --dry-run
```
Re-run the inventory with `--compare <pre.json>` to prove that no checkpoint or benchmark file changed.

## Known limitations

Lexical rules only (no syntactic parsing); English and Spanish only (about half of the reports); severity
thresholds for OA / effusion / synovitis are annotation-dependent; gold studies are enriched for abnormalities, so
rule-derived prevalences on the unlabeled population are lower and not directly comparable; reports and images
disagree, which caps what any report-based label can teach. The extractor has not been validated well enough for
bulk labeling; no production manifest has been written.

## Proposed next stage (design only, nothing implemented)

1. **Kaggle training notebook** reusing `scripts/kaggle_rsna_runner.py` and `train_rsna_resumable`: a new masked-loss
   trainer (`BCEWithLogits` summed over mask=1 targets, `pos_weight` from the training fold), selected by a flag so the
   frozen baseline path is unchanged.
2. **Model**: 2.5D slice encoder (pretrained 2D backbone allowed by the rules, to be confirmed) with an
   attention-MIL pooling head over slices and planes, compared with the existing max-pool CNN as the baseline.
3. **Confidence-controlled weak labels**: train on `positive`/`negative` only; later add per-target, per-language
   weights from measured extractor reliability, and optionally soft targets for `uncertain`. Pseudo-labelling only
   after a teacher clears a validated threshold.
4. **Optional offline multilingual LLM extraction** (open weights, run inside the Kaggle notebook or locally, no external
   API) as a second extractor implementing the same protocol; compare to rules on the same DEV/EVAL protocol, and
   combine only where it measurably helps.
5. **Evaluation**: 5-fold study-level CV with the 58 gold studies never in training; report per-target AUROC with
   CIs and the baseline side by side. Checkpoint/resume and export use the existing portable format.
6. **Decision engine** stays separate: provider-independent interface (Groq, AMD Token Factory, Nebius, offline
   fallback) fed by `StudyInferenceResult`; no provider integration is part of this phase.
