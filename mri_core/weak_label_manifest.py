"""Versioned, deterministic weak-label manifests built from radiology reports.

A manifest lists, per eligible study, the report-derived state of each of the 12 targets plus a loss mask
(1 = trainable positive/negative, 0 = masked uncertain / not_mentioned / unsupported language).

Eligibility (study must pass every rule):
  * all 12 structured targets are missing in train.csv  -> studies with ANY gold target are excluded, so
    the 58 gold studies can never become weak-label training data or leak into validation;
  * not present in test.csv / sample_submission.csv (Kaggle test studies);
  * not in any caller-supplied holdout set (e.g. fold validation studies);
  * has report text.

Nothing here modifies train.csv or any DICOM; outputs go only to an explicit path and never overwrite
without `overwrite=True`. Output is byte-stable for identical inputs and extractor version (rows sorted,
sorted JSON keys, no timestamps). Evidence text is not stored unless asked; only its SHA-256 is.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .report_labels import DECIDED_STATES, LabelState, ReportLabelExtractor, ReportLabels, to_training_arrays
from .rsna_knee_dataset import TARGET_COLUMNS

SCHEMA = "weak-label-manifest/1"


class ManifestError(ValueError):
    pass


class LeakageError(ManifestError):
    pass


def redact(uid: str) -> str:
    """Short form safe for logs."""
    return "…" + str(uid)[-6:]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_eligible(train: pd.DataFrame, *, test_ids: Iterable[str] = (), holdout_ids: Iterable[str] = ()):
    """Return (eligible_frame, exclusion_counts, gold_ids)."""
    targets = list(TARGET_COLUMNS)
    uid = train["StudyInstanceUID"].astype(str)
    any_label = train[targets].notna().any(axis=1)
    gold_ids = set(uid[any_label])
    test_set, holdout_set = set(map(str, test_ids)), set(map(str, holdout_ids))
    in_test = uid.isin(test_set)
    in_holdout = uid.isin(holdout_set)
    no_report = train["Report"].isna() | (train["Report"].astype(str).str.strip() == "")
    keep = ~any_label & ~in_test & ~in_holdout & ~no_report
    counts = {"train_rows": int(len(train)), "excluded_has_structured_labels": int(any_label.sum()),
              "excluded_in_test": int((in_test & ~any_label).sum()),
              "excluded_in_holdout": int((in_holdout & ~any_label & ~in_test).sum()),
              "excluded_no_report": int((no_report & ~any_label & ~in_test & ~in_holdout).sum()),
              "eligible": int(keep.sum())}
    return train[keep].sort_values("StudyInstanceUID").reset_index(drop=True), counts, gold_ids


def assert_no_leakage(manifest_ids: Iterable[str], *, gold_ids: Iterable[str], test_ids: Iterable[str],
                      holdout_ids: Iterable[str] = ()) -> None:
    ids = set(map(str, manifest_ids))
    for name, other in (("gold", gold_ids), ("test", test_ids), ("holdout", holdout_ids)):
        overlap = ids & set(map(str, other))
        if overlap:
            raise LeakageError(f"{len(overlap)} manifest studies overlap with the {name} set")


def _row(labels: ReportLabels, include_evidence: bool, active: frozenset) -> dict:
    values, mask = to_training_arrays(labels)
    targets = {}
    for i, label in enumerate(labels.labels):
        entry = label.to_dict(include_evidence)
        inactive = label.target not in active
        entry["mask"] = 0 if inactive else int(mask[i])
        entry["value"] = None if entry["mask"] == 0 else float(values[i])
        entry["tier_masked"] = bool(inactive)  # the extracted state is kept for provenance but never trains
        targets[label.target] = entry
    return {"study_uid": labels.study_uid, "language": labels.language, "supported": labels.supported,
            "method": labels.method, "extractor_version": labels.extractor_version, "targets": targets}


def build_manifest(train: pd.DataFrame, extractor: ReportLabelExtractor, *, test_ids: Iterable[str] = (),
                   holdout_ids: Iterable[str] = (), include_evidence: bool = False, min_decided: int = 1,
                   include_unsupported: bool = False, train_csv_sha256: Optional[str] = None,
                   active_targets: Optional[Sequence[str]] = None):
    """Return (rows, summary). Pure function: no files are read or written.

    active_targets: targets that may train (default all 12). Others are masked with tier_masked=True, and
    do not count toward min_decided.
    """
    if min_decided < 0:
        raise ManifestError("min_decided must be >= 0")
    active = frozenset(TARGET_COLUMNS if active_targets is None else active_targets)
    unknown = active - set(TARGET_COLUMNS)
    if unknown or not active:
        raise ManifestError(f"active_targets must be a non-empty subset of the 12 targets (got unknown: {sorted(unknown)})")
    test_ids, holdout_ids = list(map(str, test_ids)), list(map(str, holdout_ids))
    eligible, exclusions, gold_ids = select_eligible(train, test_ids=test_ids, holdout_ids=holdout_ids)
    rows, state_counts = [], {t: Counter() for t in TARGET_COLUMNS}
    languages, unsupported, below_min = Counter(), 0, 0
    for record in eligible.itertuples():
        labels = extractor.extract(str(record.Report), str(record.StudyInstanceUID))
        languages[labels.language] += 1
        if not labels.supported:
            unsupported += 1
            if not include_unsupported:
                continue
        elif sum(l.state in DECIDED_STATES for l in labels.labels if l.target in active) < min_decided:
            below_min += 1
            continue
        for label in labels.labels:
            state_counts[label.target][label.state.value] += 1
        rows.append(_row(labels, include_evidence, active))
    rows.sort(key=lambda r: r["study_uid"])
    assert_no_leakage([r["study_uid"] for r in rows], gold_ids=gold_ids, test_ids=test_ids, holdout_ids=holdout_ids)
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    trainable = {t: {"positive": state_counts[t]["positive"], "negative": state_counts[t]["negative"]}
                 for t in TARGET_COLUMNS if t in active}
    for t, counts in trainable.items():
        decided = counts["positive"] + counts["negative"]
        counts["positive_fraction"] = round(counts["positive"] / decided, 4) if decided else None
        counts["coverage_of_included"] = round(decided / len(rows), 4) if rows else None
    summary = {
        "schema": SCHEMA, "extractor_version": extractor.version,
        "extractor_source_sha256": getattr(extractor, "source_sha256", None), "train_csv_sha256": train_csv_sha256,
        "config": {"include_evidence": include_evidence, "min_decided": min_decided,
                   "include_unsupported": include_unsupported, "n_test_ids": len(test_ids),
                   "n_holdout_ids": len(holdout_ids), "active_targets": [t for t in TARGET_COLUMNS if t in active]},
        "counts": {**exclusions, "unsupported_language": unsupported, "below_min_decided": below_min,
                   "included": len(rows)},
        "leakage_checks": {"gold_studies_excluded": len(gold_ids), "overlap_with_gold": 0,
                           "overlap_with_test": 0, "overlap_with_holdout": 0},
        "languages_eligible": dict(sorted(languages.items())),
        "state_counts": {t: {s.value: state_counts[t][s.value] for s in LabelState} for t in TARGET_COLUMNS},
        "trainable": trainable,
        "manifest_sha256": sha256_bytes(body.encode("utf-8")),
    }
    return rows, summary


def _check_output_path(path: Path, protected: Iterable[Path]) -> None:
    resolved = path.resolve()
    for other in protected:
        if other is not None and resolved == Path(other).resolve():
            raise ManifestError(f"Refusing to write over an input file: {other}")
    if resolved.suffix.lower() in (".csv", ".dcm", ".pt"):
        raise ManifestError("Manifest output must be a .jsonl file, not a data or checkpoint file")


def write_manifest(rows: list, summary: dict, output: Path, *, overwrite: bool = False,
                   protected: Iterable[Path] = ()) -> tuple[Path, Path]:
    output = Path(output)
    _check_output_path(output, protected)
    summary_path = output.with_name(output.name + ".summary.json")
    for target in (output, summary_path):
        if target.exists() and not overwrite:
            raise ManifestError(f"{target} exists; pass overwrite=True to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    if sha256_bytes(body.encode("utf-8")) != summary["manifest_sha256"]:
        raise ManifestError("rows do not match the summary hash")
    output.write_bytes(body.encode("utf-8"))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return output, summary_path


def load_training_arrays(path: Path):
    """Return (study_ids, values[N,12] with NaN where masked, masks[N,12]); verifies hash and mask consistency."""
    path = Path(path)
    raw = path.read_bytes()
    summary_path = path.with_name(path.name + ".summary.json")
    if summary_path.is_file():
        expected = json.loads(summary_path.read_text(encoding="utf-8"))["manifest_sha256"]
        if sha256_bytes(raw) != expected:
            raise ManifestError("manifest content does not match its recorded SHA-256")
    ids, values, masks = [], [], []
    for line in raw.decode("utf-8").splitlines():
        row = json.loads(line)
        value_row, mask_row = [], []
        for target in TARGET_COLUMNS:
            entry = row["targets"][target]
            if entry["mask"] == 1 and entry["value"] not in (0.0, 1.0):
                raise ManifestError("trainable target without a hard 0/1 value")
            if entry["mask"] == 1 and entry["state"] not in ("positive", "negative"):
                raise ManifestError("uncertain / not_mentioned target is marked trainable")
            value_row.append(np.nan if entry["mask"] == 0 else entry["value"])
            mask_row.append(entry["mask"])
        ids.append(row["study_uid"])
        values.append(value_row)
        masks.append(mask_row)
    return ids, np.array(values, dtype=np.float32), np.array(masks, dtype=np.float32)
