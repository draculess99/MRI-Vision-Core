"""Label-mask override for Phase 17: B0 (original weak-label mask) versus C1 (higher-confidence candidate mask).

The override only REMOVES trainable entries (mask 1 -> 0). It never adds an entry, never changes a label value and never turns a missing,
uncertain or masked label into a negative. It is bound to the manifest it was built from (SHA-256) and to the study IDs it covers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .rsna_knee_dataset import TARGET_COLUMNS

SCHEMA = "phase17-candidate-mask/1"
MODES = ("B0", "C1")


class MaskOverrideError(ValueError):
    pass


def fingerprint(studies_by_target: Mapping[str, Sequence[str]], values_by_target: Mapping[str, Sequence[float]]) -> str:
    """Order-independent SHA-256 of (study, target, value) candidate entries."""
    h = hashlib.sha256()
    entries = sorted((uid, t, int(v)) for t in studies_by_target for uid, v in zip(studies_by_target[t], values_by_target[t]))
    for uid, t, v in entries:
        h.update(f"{uid}:{t}:{v}\n".encode())
    return h.hexdigest()


def build_override(candidate: np.ndarray, values: np.ndarray, ids: Sequence[str], *, manifest_sha256: str, selection_sha256: str,
                   targets: Sequence[str]) -> dict:
    """candidate/values: arrays [n, 12] aligned with ids and TARGET_COLUMNS (as produced by weak_label_quality.candidate_labels)."""
    studies, vals = {}, {}
    for t in targets:
        j = TARGET_COLUMNS.index(t)
        rows = np.flatnonzero(candidate[:, j])
        studies[t] = [ids[i] for i in rows]
        vals[t] = [int(values[i, j]) for i in rows]
    return {"schema": SCHEMA, "manifest_sha256": manifest_sha256, "selection_sha256": selection_sha256,
            "targets": {t: {"studies": studies[t], "values": vals[t]} for t in targets}, "fingerprint": fingerprint(studies, vals)}


def load_override(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise MaskOverrideError(f"unsupported override schema {data.get('schema')!r}")
    recomputed = fingerprint({t: d["studies"] for t, d in data["targets"].items()}, {t: d["values"] for t, d in data["targets"].items()})
    if recomputed != data["fingerprint"]:
        raise MaskOverrideError("override file fingerprint does not match its contents (edited or corrupted)")
    return data


def apply_override(mode: str, ids: Sequence[str], values: np.ndarray, masks: np.ndarray, *, manifest_sha256: str,
                   override: Mapping | None, expected_fingerprint: str | None = None) -> np.ndarray:
    """Return the training mask for `mode`. B0 returns a copy of the original mask. C1 requires an override bound to this manifest and
    returns original_mask * candidate; values are never touched (the caller keeps using the original `values`)."""
    if mode not in MODES:
        raise MaskOverrideError(f"unknown label mask mode {mode!r}")
    original = np.array(masks, dtype=np.float32, copy=True)
    if mode == "B0":
        return original
    if override is None:
        raise MaskOverrideError("mode C1 needs a label-mask override file")
    if override["manifest_sha256"] != manifest_sha256:
        raise MaskOverrideError("override was built for a different manifest")
    if expected_fingerprint is not None and override["fingerprint"] != expected_fingerprint:
        raise MaskOverrideError("override fingerprint differs from the frozen candidate fingerprint")
    index = {u: i for i, u in enumerate(ids)}
    new = np.zeros_like(original)
    for target, entry in override["targets"].items():
        j = TARGET_COLUMNS.index(target)
        for uid, value in zip(entry["studies"], entry["values"]):
            if uid not in index:
                continue  # candidate outside the studies used in this run (e.g. a different subset): not trainable here
            i = index[uid]
            if original[i, j] != 1:
                raise MaskOverrideError("a candidate label is masked in the original manifest (override and manifest disagree)")
            if int(values[i, j]) != int(value):
                raise MaskOverrideError("a candidate label value differs from the manifest value (override and manifest disagree)")
            new[i, j] = 1.0
    out = original * new
    # targets that are not covered by the override keep NO training signal under C1 (they are not candidates)
    if (out > original).any():
        raise MaskOverrideError("override attempted to enable a masked entry")
    return out
