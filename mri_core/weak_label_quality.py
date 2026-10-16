"""Label-quality audit and documented high-confidence filtering for the weak report labels (text-free: works on the manifest fields only).

Nothing here changes a label. The candidate subset is a *mask* over the original labels (never a rewrite); states "not_mentioned" and "uncertain" are
never converted to negatives; model performance, the Phase 16 hold-out, the gold studies and the competition test labels are never used to define it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .rsna_knee_dataset import TARGET_COLUMNS

DECIDED = ("positive", "negative")
# Explicit, documented confidence tiers by the rule that produced the label (see docs/phase17_label_quality_audit.md).
HIGH_REASONS = {"positive": {None}, "negative": {"negated_lesion", "normal_structure", "normal_cartilage"}}
LOW_REASONS = {"positive": {"inherited_side"}, "negative": {"list_inherited", "generic_location", "normal_phrase"}}
SHARED_EVIDENCE_LIMIT = 4  # one sentence/list that supports >= this many targets is a global statement, not target-specific evidence
PROFILES = ("0.1.0", "0.2.0", "0.2.2")


class AuditScopeError(ValueError):
    pass


def load_manifest_rows(path: Path) -> dict:
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["study_uid"]] = row
    return rows


def assert_audit_scope(ids: Iterable[str], allowed: Iterable[str], forbidden: Mapping[str, Iterable[str]]) -> None:
    """The audit may only look at the registered training selection; hold-out, gold and test studies are refused."""
    ids, allowed = set(ids), set(allowed)
    if ids - allowed:
        raise AuditScopeError(f"{len(ids - allowed)} studies are outside the audit selection")
    for name, other in forbidden.items():
        if ids & set(other):
            raise AuditScopeError(f"audit studies overlap the protected {name} set")


def shared_evidence_counts(row: Mapping) -> dict:
    """Per target: how many targets of the same study point at the same evidence sentence/list."""
    shas = Counter(d["evidence_sha256"] for d in row["targets"].values() if d.get("evidence_sha256"))
    return {t: shas.get(d.get("evidence_sha256"), 0) if d.get("evidence_sha256") else 0 for t, d in row["targets"].items()}


def label_confidence(entry: Mapping, shared: int) -> str:
    """'undecided' (not a training label), or 'high' / 'medium' / 'low' for a decided label."""
    state = entry["state"]
    if state not in DECIDED:
        return "undecided"
    if not entry.get("evidence_sha256"):
        return "low"  # decided but without recorded evidence = unsupported
    reason = entry.get("reason")
    if shared >= SHARED_EVIDENCE_LIMIT or reason in LOW_REASONS[state]:
        return "low"
    return "high" if reason in HIGH_REASONS[state] else "medium"


def stability(states_by_profile: Sequence[Optional[str]]) -> str:
    """'stable' when every available profile gives the same decided state; 'flipped' when two profiles give opposite decided states;
    'unstable' when a profile calls it uncertain/not_mentioned or it is missing from a profile that should have it."""
    seen = [s for s in states_by_profile if s is not None]
    if not seen:
        return "missing"
    decided = [s for s in seen if s in DECIDED]
    if len(decided) == len(states_by_profile) and len(set(decided)) == 1:
        return "stable"
    if len(set(decided)) > 1:
        return "flipped"
    return "unstable"


def candidate_labels(rows_by_profile: Mapping[str, Mapping], ids: Sequence[str], *, base_profile: str = "0.1.0",
                     tier_targets: Optional[Sequence[str]] = None, require_stable: bool = True,
                     ignore_tier_mask: bool = False) -> dict:
    """High-confidence candidate mask over the ORIGINAL (base-profile) labels. Returns arrays aligned with `ids` and TARGET_COLUMNS:
    values/mask are the original ones (NaN/0 where undecided), `candidate` is the subset flag, `confidence` and `stability` per entry."""
    base = rows_by_profile[base_profile]
    n, k = len(ids), len(TARGET_COLUMNS)
    values, mask = np.full((n, k), np.nan), np.zeros((n, k))
    candidate = np.zeros((n, k), bool)
    confidence = np.full((n, k), "undecided", dtype=object)
    stab = np.full((n, k), "missing", dtype=object)
    for i, uid in enumerate(ids):
        row = base[uid]
        shared = shared_evidence_counts(row)
        for j, target in enumerate(TARGET_COLUMNS):
            entry = row["targets"][target]
            if entry["state"] in DECIDED:  # tier-masked targets carry no value in the manifest; derive it from the state (audit only)
                values[i, j], mask[i, j] = (1.0 if entry["state"] == "positive" else 0.0), 1.0
            confidence[i, j] = label_confidence(entry, shared[target])
            states = [rows_by_profile[p][uid]["targets"][target]["state"] if uid in rows_by_profile[p] else None for p in rows_by_profile]
            stab[i, j] = stability(states)
            trainable = (tier_targets is None or target in tier_targets) and (ignore_tier_mask or not entry.get("tier_masked", False))
            candidate[i, j] = bool(mask[i, j] == 1 and confidence[i, j] == "high" and trainable and (stab[i, j] == "stable" or not require_stable))
    return {"values": values, "mask": mask, "candidate": candidate, "confidence": confidence, "stability": stab}


def never_negative_from_missing(rows: Mapping[str, Mapping], result: Mapping, ids: Sequence[str]) -> bool:
    """True when no label with state not_mentioned/uncertain became a decided training value, in the audit arrays."""
    for i, uid in enumerate(ids):
        for j, target in enumerate(TARGET_COLUMNS):
            if rows[uid]["targets"][target]["state"] not in DECIDED and (result["mask"][i, j] != 0 or result["candidate"][i, j]):
                return False
    return True


def candidate_fingerprint(ids: Sequence[str], result: Mapping) -> str:
    h = hashlib.sha256()
    for i, uid in enumerate(ids):
        for j in np.flatnonzero(result["candidate"][i]):
            h.update(f"{uid}:{TARGET_COLUMNS[j]}:{int(result['values'][i, j])}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- statistics
def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def positives_needed(auc: float, positive_rate: float, half_width: float = 0.05, limit: int = 100000) -> tuple[int, int]:
    """Smallest adjudicated sample (n_pos, n_total) for which a 95% CI around `auc` has the requested half-width (Hanley-McNeil)."""
    for total in range(40, limit, 10):
        n_pos = max(int(round(total * positive_rate)), 2)
        if 1.96 * hanley_mcneil_se(auc, n_pos, total - n_pos) <= half_width:
            return n_pos, total
    return -1, -1


def auc_of_score(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(score)
    y, score = y[ok], score[ok]
    return float(roc_auc_score(y, score)) if len(y) and 0 < y.sum() < len(y) else None


def two_by_two_p(a_pos: int, a_n: int, b_pos: int, b_n: int) -> Optional[float]:
    """Chi-square p-value for equal positive rates in two groups; None when a group is empty."""
    from scipy.stats import chi2_contingency
    if a_n == 0 or b_n == 0:
        return None
    table = np.array([[a_pos, a_n - a_pos], [b_pos, b_n - b_pos]])
    if (table.sum(axis=0) == 0).any():
        return None
    return float(chi2_contingency(table, correction=False)[1])
