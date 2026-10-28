"""Data access for the weak-label experiment: manifest loading, study-level splits, leakage guards,
series pre-flight checks and a mask-carrying DICOM dataset.

The dataset reuses RSNAKneeDicomDataset's series selection / slice sampling / normalisation unchanged, so every arm
(and the preserved baseline) sees identical pixels. It only swaps the labels for manifest values and adds a
per-target training mask.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .rsna_checkpoint import studies_fingerprint
from .rsna_knee_dataset import PLANE_NAMES, TARGET_COLUMNS, RSNAKneeMetadata, select_series
from .rsna_knee_train import RSNAKneeDicomDataset
from .weak_label_manifest import LeakageError, load_training_arrays


CACHE_VERSION = 1  # bump whenever preprocessing changes so stale cache entries are never reused


@dataclass(frozen=True)
class WeakLabelManifest:
    ids: tuple
    values: np.ndarray  # [N, 12], NaN where masked
    masks: np.ndarray  # [N, 12], 1 = trainable
    languages: tuple
    summary: dict

    @property
    def sha256(self) -> str:
        return self.summary["manifest_sha256"]

    def fingerprint(self) -> str:
        """Identity of the exact label content + study set; stored in every checkpoint."""
        return hashlib.sha256(f"{self.sha256}:{studies_fingerprint(self.ids)}".encode()).hexdigest()


def load_weak_manifest(path: Path) -> WeakLabelManifest:
    path = Path(path)
    ids, values, masks = load_training_arrays(path)  # verifies the SHA-256 recorded next to the file
    summary_path = path.with_name(path.name + ".summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Manifest summary missing next to {path}; refusing to train on an unverified manifest")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    languages = tuple(json.loads(line)["language"] for line in path.read_text(encoding="utf-8").splitlines())
    return WeakLabelManifest(tuple(ids), values, masks, languages, summary)


def hash_split(ids: Sequence[str], val_fraction: float, seed: int):
    """Deterministic study-level split by hashing UIDs; independent of row order. Returns (train_idx, val_idx)."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    bucket = [int(hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for uid in ids]
    val = [i for i, b in enumerate(bucket) if b < val_fraction]
    train = [i for i, b in enumerate(bucket) if b >= val_fraction]
    if not train or not val:
        raise ValueError("Split produced an empty train or validation set")
    return train, val


def stability_split_preflight(ids: Sequence[str], registered_val_ids: Iterable[str], val_fraction: float, seed: int, *, min_train: int = 8, min_val: int = 4) -> dict:
    """Non-empty partition check for the bounded stability rehearsal, run before any GPU work.

    The rehearsal studies are taken from the registered TRAINING partition, i.e. the studies whose seed hash is >= the registered validation fraction (0.2). Re-splitting them with the
    same fraction and seed therefore puts every study in train (the Phase 19C v1 failure). A larger fraction turns the hash band [0.2, fraction) of those training studies into an
    INTERNAL validation set; it never contains a registered validation study. Returns the counts, or raises ValueError."""
    ids = [str(u) for u in ids]
    registered = set(map(str, registered_val_ids))
    if registered & set(ids):
        raise ValueError(f"{len(registered & set(ids))} stability studies belong to the registered validation partition")
    try:
        train, val = hash_split(ids, val_fraction, seed)
    except ValueError as error:
        raise ValueError(f"stability split is empty at val_fraction={val_fraction} for {len(ids)} studies: {error}") from error
    if len(train) < min_train or len(val) < min_val:
        raise ValueError(f"stability split too small at val_fraction={val_fraction}: {len(train)} train / {len(val)} validation (need >= {min_train} / {min_val})")
    return {"studies": len(ids), "internal_train": len(train), "internal_validation": len(val), "val_fraction": val_fraction, "seed": seed,
            "internal_validation_disjoint_from_registered_validation": True}


def assert_no_leakage(train_ids: Iterable[str], val_ids: Iterable[str], *, gold_ids: Iterable[str],
                      test_ids: Iterable[str] = ()) -> None:
    train_ids, val_ids = set(map(str, train_ids)), set(map(str, val_ids))
    for name, other in (("gold", set(map(str, gold_ids))), ("test", set(map(str, test_ids)))):
        if (train_ids | val_ids) & other:
            raise LeakageError(f"weak-label training/validation studies overlap the {name} set")
    if train_ids & val_ids:
        raise LeakageError("training and validation studies overlap")


def preflight_series(metadata: RSNAKneeMetadata, ids: Iterable[str], planes: Sequence[str] = PLANE_NAMES) -> dict:
    """Read-only: which studies lack a usable series directory for a required plane. No pixels are decoded."""
    problems = {}
    for uid in map(str, ids):
        rows = metadata.series_for(uid)
        reasons = []
        for plane in planes:
            candidates = rows[rows["Anatomical_Plane"].astype(str).str.lower() == plane.lower()]
            if candidates.empty:
                reasons.append(f"no {plane} series listed")
                continue
            directory = metadata.dicom_series_dir(uid, select_series(candidates))
            if not directory.is_dir():
                reasons.append(f"{plane} series directory missing")
            elif not any(directory.iterdir()):
                reasons.append(f"{plane} series directory empty")
        if reasons:
            problems[uid] = reasons
    return problems


class WeakLabelDicomDataset(RSNAKneeDicomDataset):
    """Yields images/mask exactly like RSNAKneeDicomDataset plus labels, target_mask and valid.

    on_error="raise" (default) propagates a missing/corrupt series. on_error="skip" returns an all-masked zero
    sample (valid=False) so one bad study cannot kill a long run; the trainer counts these and aborts above a limit.
    """

    def __init__(self, metadata: RSNAKneeMetadata, ids: Sequence[str], values: np.ndarray, masks: np.ndarray,
                 config: dict, *, on_error: str = "raise", cache_dir: Optional[Path] = None):
        if on_error not in ("raise", "skip"):
            raise ValueError("on_error must be 'raise' or 'skip'")
        if not (len(ids) == len(values) == len(masks)):
            raise ValueError("ids, values and masks must align")
        frame = pd.DataFrame({"StudyInstanceUID": list(map(str, ids))})
        for target in TARGET_COLUMNS:
            frame[target] = 0.0
        super().__init__(metadata, frame, config, "train")
        self.values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
        self.masks = np.asarray(masks, dtype=np.float32)
        self.on_error = on_error
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.cache_hits = self.cache_misses = 0  # counted in the process that reads (not aggregated across DataLoader workers)
        self._shape_key = f"{config['image_size']}-{config['slices_per_series']}-{','.join(config.get('planes', PLANE_NAMES))}"

    def _cache_path(self, uid: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{hashlib.sha1(f'v{CACHE_VERSION}:{uid}:{self._shape_key}'.encode()).hexdigest()}.pt"

    def _cache_meta(self, uid: str) -> dict:
        """Identity stored inside every entry: a file copied under the wrong name or built with other settings is rejected."""
        return {"uid": uid, "settings": self._shape_key, "version": CACHE_VERSION}

    def _valid_cache_entry(self, images: torch.Tensor, mask: torch.Tensor) -> bool:
        size, slices = int(self.config["image_size"]), int(self.config["slices_per_series"])
        planes = len(self.config.get("planes", PLANE_NAMES))
        return (tuple(images.shape) == (planes, slices, 1, size, size) and tuple(mask.shape) == (planes, slices)
                and mask.dtype == torch.bool and bool(mask.any(dim=1).all()) and bool(torch.isfinite(images).all()))

    def store_entry(self, uid: str, images: torch.Tensor, mask: torch.Tensor) -> None:
        """Atomically write one float16 cache entry with its identity metadata (shared by decoding and bulk cache building)."""
        path = self._cache_path(uid)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(f".{os.getpid()}.tmp")  # per-process name; rename is atomic
        torch.save({"images": images.half(), "mask": mask, "meta": self._cache_meta(uid)}, temp)
        temp.replace(path)

    def has_valid_entry(self, uid: str) -> bool:
        path = self._cache_path(uid)
        if path is None or not path.is_file():
            return False
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
            return bool(blob.get("meta") == self._cache_meta(uid) and self._valid_cache_entry(blob["images"].float(), blob["mask"]))
        except Exception:
            return False

    def _decode(self, index):
        """Decode (or load from the float16 cache). The cached value is what every epoch and every arm sees, so
        results do not depend on whether a study happened to be cached already."""
        uid = str(self.studies.iloc[index]["StudyInstanceUID"])
        path = self._cache_path(uid)
        if path is not None and path.is_file():
            try:
                blob = torch.load(path, map_location="cpu", weights_only=True)
                images, mask = blob["images"].float(), blob["mask"]
                if blob.get("meta") == self._cache_meta(uid) and self._valid_cache_entry(images, mask):
                    self.cache_hits += 1
                    return {"images": images, "mask": mask, "study_uid": uid}
            except Exception:
                pass  # unreadable or stale entry: fall through, re-decode and atomically rewrite it
        sample = super().__getitem__(index)
        self.cache_misses += 1
        if path is not None:
            self.store_entry(uid, sample["images"], sample["mask"])
            sample = {"images": sample["images"].half().float(), "mask": sample["mask"], "study_uid": uid}
        return sample

    def __getitem__(self, index):
        try:
            sample = self._decode(index)
            valid = True
        except Exception:
            if self.on_error == "raise":
                raise
            size, slices = int(self.config["image_size"]), int(self.config["slices_per_series"])
            planes = len(self.config.get("planes", PLANE_NAMES))
            mask = torch.zeros(planes, slices, dtype=torch.bool)
            mask[:, 0] = True  # models require one valid slice per plane; the all-zero target mask removes its loss
            sample = {"images": torch.zeros(planes, slices, 1, size, size), "mask": mask,
                      "study_uid": str(self.studies.iloc[index]["StudyInstanceUID"])}
            valid = False
        sample["labels"] = torch.from_numpy(self.values[index])
        sample["target_mask"] = torch.from_numpy(self.masks[index]) * (1.0 if valid else 0.0)
        sample["valid"] = torch.tensor(valid)
        return sample


def select_smoke_subset(ids: Sequence[str], values: np.ndarray, masks: np.ndarray, n: int, *, min_per_class: int = 8,
                        targets: Sequence[str] = ("ACL", "Medial Meniscus", "Lateral Meniscus"), seed: int = 0) -> dict:
    """Deterministic, study-level subset of at most `n` studies for a smoke test.

    Every candidate must already have >= 1 decided (mask 1) target among `targets`. Studies are visited in hash order;
    a first pass takes studies that fill an unmet per-target/per-class quota (min_per_class positives and negatives where
    feasible), a second pass fills up to n in the same hash order. Returns indices (sorted) plus a quota report.
    """
    cols = [TARGET_COLUMNS.index(t) for t in targets]
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(f"smoke:{seed}:{ids[i]}".encode()).hexdigest())
    candidates = [i for i in order if masks[i, cols].sum() > 0]
    quota = {(t, c): min_per_class for t in cols for c in (0.0, 1.0)}
    chosen, taken = [], set()
    for i in candidates:
        if len(chosen) >= n or not any(v > 0 for v in quota.values()):
            break
        helps = [(t, float(values[i, t])) for t in cols if masks[i, t] == 1 and quota[(t, float(values[i, t]))] > 0]
        if helps:
            chosen.append(i)
            taken.add(i)
            for key in helps:
                quota[key] -= 1
    for i in candidates:
        if len(chosen) >= n:
            break
        if i not in taken:
            chosen.append(i)
            taken.add(i)
    chosen.sort()
    balance = {}
    for t in cols:
        on = masks[chosen, t] == 1
        balance[TARGET_COLUMNS[t]] = {"positive": int(((values[chosen, t] == 1) & on).sum()),
                                      "negative": int(((values[chosen, t] == 0) & on).sum())}
    return {"indices": chosen, "candidates": len(candidates),
            "unmet_quota": {f"{TARGET_COLUMNS[t]}:{'positive' if c == 1.0 else 'negative'}": v
                            for (t, c), v in quota.items() if v > 0},
            "class_balance": balance}
