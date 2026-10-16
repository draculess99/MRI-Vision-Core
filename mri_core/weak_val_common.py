"""A common weak-validation yardstick for comparing models trained on DIFFERENT label versions.

Weak-validation loss is computed against the labels of the manifest a model was trained with, so it cannot rank label versions.
Here every model is scored on the same studies against labels that two manifests AGREE on (both decided, same value) and that are in
both validation splits. This still measures agreement with automated labels, not with experts: it is a fair yardstick between runs,
not evidence of clinical accuracy. Gold studies are never touched.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .rsna_knee_dataset import TARGET_COLUMNS
from .weak_label_data import WeakLabelManifest, hash_split

TIER1 = ("ACL", "Medial Meniscus", "Lateral Meniscus")


def common_yardstick(manifests: Mapping[str, WeakLabelManifest], *, val_fraction: float, seed: int,
                     targets: Sequence[str] = TIER1) -> dict:
    """Return {"ids": [...], "labels": {target: {uid: 0/1}}} for validation studies shared by all manifests, with per-target
    labels on which every manifest is decided and identical."""
    val_sets = []
    for m in manifests.values():
        _, val_idx = hash_split(list(m.ids), val_fraction, seed)
        val_sets.append({m.ids[i] for i in val_idx})
    shared = set.intersection(*val_sets)
    index = {name: {uid: i for i, uid in enumerate(m.ids)} for name, m in manifests.items()}
    labels = {}
    for target in targets:
        j = TARGET_COLUMNS.index(target)
        agreed = {}
        for uid in sorted(shared):
            values = []
            for name, m in manifests.items():
                i = index[name][uid]
                values.append(float(m.values[i, j]) if m.masks[i, j] == 1 else None)
            if None not in values and len(set(values)) == 1:
                agreed[uid] = int(values[0])
        labels[target] = agreed
    return {"ids": sorted(shared), "labels": labels, "n_shared_validation_studies": len(shared)}


def score_on_yardstick(probabilities: Mapping[str, np.ndarray], uids: Sequence[str], yardstick: dict, *, rounds: int = 1000,
                       seed: int = 0, targets: Sequence[str] = TIER1) -> dict:
    """probabilities[target] aligned with `uids`; returns per-target AUC with bootstrap CI over studies and the macro mean."""
    from sklearn.metrics import roc_auc_score
    position = {u: i for i, u in enumerate(uids)}
    rng = np.random.default_rng(seed)
    out = {"per_target": {}, "n_shared_validation_studies": yardstick["n_shared_validation_studies"]}
    aucs = []
    for target in targets:
        agreed = {u: y for u, y in yardstick["labels"][target].items() if u in position}
        ids = sorted(agreed)
        y = np.array([agreed[u] for u in ids])
        p = np.array([probabilities[target][position[u]] for u in ids])
        entry = {"n": len(ids), "positives": int(y.sum()), "negatives": int(len(y) - y.sum()), "auc": None, "auc_ci95": None}
        if entry["positives"] >= 3 and entry["negatives"] >= 3:
            entry["auc"] = float(roc_auc_score(y, p))
            draws = []
            for _ in range(rounds):
                idx = rng.integers(0, len(y), len(y))
                if 0 < y[idx].sum() < len(idx):
                    draws.append(roc_auc_score(y[idx], p[idx]))
            entry["auc_ci95"] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if len(draws) >= 50 else None
            aucs.append(entry["auc"])
        out["per_target"][target] = entry
    out["macro_auc"] = float(np.mean(aucs)) if len(aucs) == len(targets) else None
    return out
