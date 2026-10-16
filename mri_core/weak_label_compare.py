"""Phase 17A comparison harness: B0, C1, metadata-only, shuffled-label and untrained-network arms scored against one reference label set.

The reference is typed. Weak-label references (the manifest's own labels) are DESCRIPTIVE: no claim can be made from them. Only an adjudicated
reference that is disjoint from every training, validation, hold-out, gold and test study can license a claim (`claim_allowed`).
Medial Meniscus is the primary target; ACL and Lateral Meniscus are secondary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .weak_label_confound import ACTIVE, ACTIVE_IDX, bootstrap, target_auc

PRIMARY_TARGET = "Medial Meniscus"
SECONDARY_TARGETS = ("ACL", "Lateral Meniscus")
ARMS = ("B0", "C1", "M0", "S", "N")  # original CNN, candidate-mask CNN, metadata-only, shuffled-label control, untrained control


@dataclass(frozen=True)
class ReferenceLabels:
    kind: str  # "weak" or "adjudicated"
    ids: tuple
    values: np.ndarray  # [n, 12]; NaN where undecided
    masks: np.ndarray  # [n, 12]; 1 where the reference decides
    source: str = ""


def claim_allowed(reference: ReferenceLabels, protected: Mapping[str, Iterable[str]]) -> tuple[bool, list]:
    """A claim needs an adjudicated reference with no overlap with any protected set (training, validation, hold-out, gold, test)."""
    reasons = []
    if reference.kind != "adjudicated":
        reasons.append("the reference is weak (automated) labels: results are descriptive only, not evidence of accuracy")
    ids = set(reference.ids)
    for name, other in protected.items():
        n = len(ids & set(other))
        if n:
            reasons.append(f"{n} reference studies overlap the {name} set")
    return (not reasons), reasons


def _check_alignment(reference: ReferenceLabels, scores: Mapping[str, np.ndarray]) -> None:
    n = len(reference.ids)
    for name, arr in scores.items():
        if arr.shape != (n, len(ACTIVE)):
            raise ValueError(f"scores for {name} have shape {arr.shape}, expected {(n, len(ACTIVE))} aligned with the reference studies")


def score_table(reference: ReferenceLabels, scores: Mapping[str, np.ndarray], *, rounds: int = 1000, seed: int = 0,
                differences: Sequence[tuple[str, str]] = (), protected: Optional[Mapping[str, Iterable[str]]] = None) -> dict:
    """Per-target and macro AUC with bootstrap CIs for every arm, plus paired differences, against one reference."""
    _check_alignment(reference, scores)
    metrics = {}
    for name, arr in scores.items():
        for k, target in enumerate(ACTIVE):
            metrics[f"{name}|{target}"] = (lambda a, kk: lambda idx: target_auc(reference.values[idx], reference.masks[idx], a[idx], kk))(arr, k)
        metrics[f"{name}|macro"] = (lambda a: lambda idx: _macro(reference, a, idx))(arr)
    pairs = []
    for a, b in differences:
        for target in (*ACTIVE, "macro"):
            pairs.append((f"{a}|{target}", f"{b}|{target}"))
    boot = bootstrap(metrics, len(reference.ids), rounds=rounds, seed=seed, differences=pairs)
    allowed, reasons = claim_allowed(reference, protected or {})
    return {"reference_kind": reference.kind, "reference_source": reference.source, "studies": len(reference.ids),
            "primary_target": PRIMARY_TARGET, "secondary_targets": list(SECONDARY_TARGETS),
            "auc": boot["point"], "ci95": boot["ci95"], "paired_difference": boot["paired_difference"],
            "claim_allowed": allowed, "claim_blockers": reasons}


def _macro(reference: ReferenceLabels, scores: np.ndarray, idx: np.ndarray) -> Optional[float]:
    vals = [target_auc(reference.values[idx], reference.masks[idx], scores[idx], k) for k in range(len(ACTIVE))]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def across_seeds(tables: Sequence[Mapping], differences: Sequence[tuple[str, str]], target: str = PRIMARY_TARGET) -> dict:
    """Descriptive summary over independent seeds: mean and sd of each arm's AUC and how often each paired difference is positive. No verdict."""
    out = {"target": target, "seeds": len(tables), "arms": {}, "paired": {}}
    names = sorted({k.split("|")[0] for t in tables for k in t["auc"]})
    for name in names:
        vals = [t["auc"].get(f"{name}|{target}") for t in tables]
        vals = [v for v in vals if v is not None]
        out["arms"][name] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "n": len(vals)} if vals else None
    for a, b in differences:
        d = [t["paired_difference"].get(f"{a}|{target}-{b}|{target}", {}).get("point") for t in tables]
        d = [x for x in d if x is not None]
        out["paired"][f"{a}-{b}"] = {"mean": float(np.mean(d)), "positive_in": int(sum(x > 0 for x in d)), "of": len(d)} if d else None
    out["note"] = "descriptive aggregation of per-seed results; not a hypothesis test and not a claim"
    return out
