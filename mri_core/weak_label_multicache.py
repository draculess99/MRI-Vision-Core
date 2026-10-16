"""Build several (image size x slices-per-plane) cache variants from ONE DICOM decode per study.

The production dataset (RSNAKneeDicomDataset) re-reads the DICOMs for every configuration; for a resolution/slice ablation that would
repeat the most expensive step once per variant. This module reads each plane's series once and derives every variant with exactly the
same operations as the production code (same series choice, same np.linspace slice sampling, same per-slice 1-99 percentile
normalisation, same bilinear resize, float16 storage). Tests assert bitwise equality with the production dataset for each variant, so
this is a faster route to identical tensors, not a different preprocessing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .rsna_knee_dataset import PLANE_NAMES, load_series_volume, select_series
from .rsna_knee_train import _normalize
from .weak_label_data import WeakLabelDicomDataset


def parse_variants(text: str) -> list[tuple[int, int]]:
    """"128x9,192x15" -> [(128, 9), (192, 15)]."""
    out = []
    for item in text.split(","):
        size, slices = item.strip().lower().split("x")
        out.append((int(size), int(slices)))
    if not out or len(set(out)) != len(out):
        raise ValueError("variants must be a non-empty list without duplicates")
    return out


def variants_for_study(metadata, study_uid: str, planes: Sequence[str], variants: Sequence[tuple[int, int]], split: str = "train") -> dict:
    """{(size, slices): (images [P,S,1,H,W] float32, mask [P,S] bool)} decoded with a single read of each plane's series."""
    max_slices = max(s for _, s in variants)
    out = {v: (torch.zeros(len(planes), v[1], 1, v[0], v[0], dtype=torch.float32), torch.zeros(len(planes), v[1], dtype=torch.bool))
           for v in variants}
    rows = metadata.series_for(study_uid, split)
    for plane_index, plane in enumerate(planes):
        candidates = rows[rows["Anatomical_Plane"].astype(str).str.lower() == plane.lower()]
        if candidates.empty:
            raise ValueError(f"StudyInstanceUID={study_uid}: no {plane} series listed in the {split} metadata")
        volume = load_series_volume(metadata, study_uid, select_series(candidates), split)
        normalised: dict[int, torch.Tensor] = {}
        for size, slices in variants:
            chosen = np.linspace(0, volume.num_slices - 1, min(slices, volume.num_slices), dtype=int)
            images, mask = out[(size, slices)]
            for slice_index, source in enumerate(chosen):
                source = int(source)
                if source not in normalised:  # percentile normalisation is per slice, so it is shared across variants
                    normalised[source] = torch.from_numpy(_normalize(volume.get_slice(source))).unsqueeze(0).unsqueeze(0)
                images[plane_index, slice_index] = F.interpolate(normalised[source], size=(size, size), mode="bilinear",
                                                                 align_corners=False)[0]
                mask[plane_index, slice_index] = True
    return out


class MultiVariantWarmDataset(Dataset):
    """Iterating this dataset (with DataLoader workers) fills the cache for every variant; unreadable studies report ok=False."""

    def __init__(self, metadata, ids: Sequence[str], variants: Sequence[tuple[int, int]], config: dict, cache_dir: Path):
        self.metadata, self.ids, self.variants = metadata, list(map(str, ids)), list(variants)
        self.planes = tuple(config.get("planes", PLANE_NAMES))
        zeros = np.zeros((len(self.ids), 12), dtype=np.float32)
        self.variant_datasets = {
            (size, slices): WeakLabelDicomDataset(metadata, self.ids, zeros, zeros,
                                                  {**config, "image_size": size, "slices_per_series": slices},
                                                  on_error="raise", cache_dir=cache_dir)
            for size, slices in self.variants}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        uid = self.ids[index]
        started = time.perf_counter()
        if all(ds.has_valid_entry(uid) for ds in self.variant_datasets.values()):
            return {"study_uid": uid, "ok": True, "decoded": False, "seconds": time.perf_counter() - started}
        try:
            built = variants_for_study(self.metadata, uid, self.planes, self.variants)
        except Exception:
            return {"study_uid": uid, "ok": False, "decoded": False, "seconds": time.perf_counter() - started}
        for variant, (images, mask) in built.items():
            self.variant_datasets[variant].store_entry(uid, images, mask)
        return {"study_uid": uid, "ok": True, "decoded": True, "seconds": time.perf_counter() - started}
