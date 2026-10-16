"""Read-only external evaluation of preserved C0-family checkpoints on MRNet (Stanford) knee MRI.

MRNet labels (abnormal / ACL tear / meniscus tear) are clinician-derived and external to RSNA; the C0 family never trained on MRNet. This module only runs
inference: it never trains, never modifies a checkpoint, never writes into the dataset folder. MRNet has ONE meniscus label, so the meniscus score is the
larger of the Medial and Lateral Meniscus probabilities (fixed in advance). Preprocessing mirrors the RSNA pipeline: per-slice 1-99 percentile normalisation,
evenly spaced slices (always including first and last), bilinear resize, plane order Axial / Coronal / Sagittal. MRNet's sequences differ from RSNA's, so a
low score can reflect domain shift; results are transfer performance on MRNet only, never RSNA accuracy or clinical accuracy.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .labels import load_labels
from .manifest import build_manifest
from .rsna_knee_dataset import TARGET_COLUMNS
from .rsna_knee_train import _normalize
from .weak_label_confound import bootstrap
from .weak_label_models import build_model

PLANES = ("axial", "coronal", "sagittal")
ENDPOINTS = ("acl", "meniscus")
UNTRAINED_SEEDS = tuple(range(9001, 9011))
J_ACL, J_MED, J_LAT = (TARGET_COLUMNS.index(t) for t in ("ACL", "Medial Meniscus", "Lateral Meniscus"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference(mrnet_root: Path, *, recover_first_row: bool = True) -> tuple[list, dict]:
    """Rows [{exam_id, split, abnormal, acl, meniscus, *_available}] plus provenance (label hashes, recoveries, missing images)."""
    root = Path(mrnet_root)
    labels = load_labels(root / "labels", recover_first_row=recover_first_row)
    image_root = next(p for p in root.iterdir() if p.is_dir() and p.name != "labels")
    manifest = build_manifest(labels, image_root)
    if manifest.missing_images:
        raise ValueError(f"{len(manifest.missing_images)} labelled exams lack an image plane")
    provenance = {"label_sha256": {p.name: sha256_file(p) for p in sorted((root / "labels").glob("*.csv"))}, "recoveries": list(labels.recoveries),
                  "exams": len(manifest.rows), "unlabeled_images": len(manifest.unlabeled_images)}
    rows = [dict(r, paths={p: manifest.paths[(r["exam_id"], p)] for p in PLANES}) for r in manifest.rows]
    return rows, provenance


def exam_tensor(volumes: Mapping[str, np.ndarray], size: int, slices: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[3 planes, slices, 1, size, size] float32 and a [3, slices] validity mask, using the RSNA preprocessing."""
    images = torch.zeros(len(PLANES), slices, 1, size, size, dtype=torch.float32)
    mask = torch.zeros(len(PLANES), slices, dtype=torch.bool)
    for p, plane in enumerate(PLANES):
        volume = volumes[plane]
        chosen = np.linspace(0, volume.shape[0] - 1, min(slices, volume.shape[0]), dtype=int)
        for k, source in enumerate(chosen):
            normalised = torch.from_numpy(_normalize(volume[int(source)])).unsqueeze(0).unsqueeze(0)
            images[p, k] = F.interpolate(normalised, size=(size, size), mode="bilinear", align_corners=False)[0]
            mask[p, k] = True
    return images, mask


class MRNetDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping], size: int, slices: int):
        self.rows, self.size, self.slices = list(rows), size, slices

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        volumes = {p: np.load(row["paths"][p], mmap_mode="r") for p in PLANES}
        images, mask = exam_tensor({p: np.asarray(v) for p, v in volumes.items()}, self.size, self.slices)
        feats = np.array([*(volumes[p].shape[0] for p in PLANES), *(float(np.mean(volumes[p])) for p in PLANES)], dtype=np.float32)
        return {"images": images, "mask": mask, "meta": torch.from_numpy(feats), "index": i}


def predict(model: torch.nn.Module, dataset: MRNetDataset, device: torch.device, batch_size: int = 8, num_workers: int = 0) -> dict:
    """Probabilities for all 12 outputs + metadata features, with timing. Eval mode, no gradients."""
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs, meta, order, forward_seconds = [], [], [], 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            x, m = batch["images"].to(device), batch["mask"].to(device)
            t = time.perf_counter()
            out = model(x, m).float().sigmoid().cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - t
            probs.append(out)
            meta.append(batch["meta"].numpy())
            order.extend(batch["index"].tolist())
    total = time.perf_counter() - started
    assert order == list(range(len(dataset)))
    n = len(dataset)
    return {"probs": np.vstack(probs), "meta": np.vstack(meta), "forward_seconds": forward_seconds, "total_seconds": total,
            "exams_per_second_forward": n / max(forward_seconds, 1e-9), "images_per_second_forward": n * dataset.slices * 3 / max(forward_seconds, 1e-9)}


def endpoint_scores(probs: np.ndarray) -> np.ndarray:
    """[n, 2] scores for (acl, meniscus): ACL output and the larger of the Medial / Lateral Meniscus outputs."""
    return np.stack([probs[:, J_ACL], np.maximum(probs[:, J_MED], probs[:, J_LAT])], axis=1)


def load_checkpoint_model(path: Path) -> tuple[torch.nn.Module, dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(state["config"]["arm"], state["config"])
    model.load_state_dict(state["model"])
    return model, state["config"]


def untrained_model(config: dict, seed: int) -> torch.nn.Module:
    from .rsna_knee_train import seed_everything
    seed_everything(seed)
    return build_model(config["arm"], config)


def auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    return float(roc_auc_score(y, s)) if len(y) and 0 < y.sum() < len(y) else None


def metadata_oof_scores(meta: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold probabilities of a logistic regression and a small boosted tree on the six acquisition features (slice counts, mean intensity per plane)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    out = np.full((len(y), 2), np.nan)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(meta, y):
        scaler = StandardScaler().fit(meta[tr])
        out[te, 0] = LogisticRegression(C=0.5, max_iter=2000).fit(scaler.transform(meta[tr]), y[tr]).predict_proba(scaler.transform(meta[te]))[:, 1]
        out[te, 1] = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, random_state=0).fit(meta[tr], y[tr]).predict_proba(meta[te])[:, 1]
    return out


def evaluate(labels: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray], untrained: Mapping[int, np.ndarray], metadata: Mapping[str, np.ndarray],
             differences: Sequence[tuple[str, str]], *, rounds: int = 1000, seed: int = 0) -> dict:
    """AUC (ACL, meniscus, mean of the two) per arm with bootstrap CIs and paired differences. Arms: checkpoints, metadata (logreg/boosted OOF), untrained mean."""
    n = len(next(iter(labels.values())))
    arms = {k: v for k, v in scores.items()}
    arms["META_logreg"] = np.stack([metadata[e][:, 0] for e in ENDPOINTS], axis=1)
    arms["META_boosted"] = np.stack([metadata[e][:, 1] for e in ENDPOINTS], axis=1)

    def endpoint_auc(arr, k, idx):
        return auc(labels[ENDPOINTS[k]][idx], arr[idx, k])

    metrics = {}
    for name, arr in arms.items():
        for k, e in enumerate(ENDPOINTS):
            metrics[f"{name}|{e}"] = (lambda a, kk: lambda idx: endpoint_auc(a, kk, idx))(arr, k)
        metrics[f"{name}|mean"] = (lambda a: lambda idx: _mean([endpoint_auc(a, 0, idx), endpoint_auc(a, 1, idx)]))(arr)
    for k, e in enumerate(ENDPOINTS):
        metrics[f"UNTRAINED|{e}"] = (lambda kk: lambda idx: _mean([endpoint_auc(u, kk, idx) for u in untrained.values()]))(k)
    metrics["UNTRAINED|mean"] = lambda idx: _mean([_mean([endpoint_auc(u, 0, idx), endpoint_auc(u, 1, idx)]) for u in untrained.values()])
    pairs = [(f"{a}|{t}", f"{b}|{t}") for a, b in differences for t in (*ENDPOINTS, "mean")]
    boot = bootstrap(metrics, n, rounds=rounds, seed=seed, differences=pairs)
    per_seed = {s: {e: auc(labels[e], u[:, k]) for k, e in enumerate(ENDPOINTS)} for s, u in untrained.items()}
    return {"reference": "MRNet clinician-derived labels (external); transfer performance only", "exams": n,
            "class_counts": {e: {"positive": int(labels[e].sum()), "negative": int(len(labels[e]) - labels[e].sum())} for e in ENDPOINTS},
            "auc": boot["point"], "ci95": boot["ci95"], "paired_difference": boot["paired_difference"], "untrained_per_seed": per_seed, "rounds": rounds}


def _mean(values):
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None
