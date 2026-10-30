"""Pre-registered submission gate for the Phase 18 leaderboard attempt (computed locally from retrieved artifacts).

Candidate = weak-label C0 trained on all trainable report-derived targets (gold and test studies never in training), scored on the 58 gold studies it never saw.
Baseline = the exact baseline family evaluated out-of-fold on the same 58 studies (each study scored by the fold checkpoint that did not train on it).
Gate (all must hold): G1 paired bootstrap 95% CI lower bound of (candidate - baseline) macro AUC > 0; G2 each seed alone beats the baseline macro AUC;
G3 candidate macro AUC >= 0.58 (so that a public score above 0.536 is plausible: baseline OOF 0.554 vs public 0.536); G4 integrity (no gold/test in training,
resume check passed, schema valid, finite values, submissions remaining).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .rsna_knee_dataset import SUBMISSION_COLUMNS, TARGET_COLUMNS

MIN_CANDIDATE_MACRO = 0.58


def macro_auc(y: np.ndarray, p: np.ndarray, idx: Optional[np.ndarray] = None) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    idx = np.arange(len(y)) if idx is None else idx
    vals = []
    for j in range(y.shape[1]):
        yy = y[idx, j]
        if 0 < yy.sum() < len(yy):
            vals.append(roc_auc_score(yy, p[idx, j]))
    return float(np.mean(vals)) if len(vals) == y.shape[1] else None  # a resample missing a class in any target is skipped


def per_target_auc(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    return {t: (float(roc_auc_score(y[:, j], p[:, j])) if 0 < y[:, j].sum() < len(y) else None) for j, t in enumerate(TARGET_COLUMNS)}


def paired_bootstrap(y: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, *, rounds: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(rounds):
        idx = rng.integers(0, len(y), len(y))
        a, b = macro_auc(y, candidate, idx), macro_auc(y, baseline, idx)
        if a is not None and b is not None:
            diffs.append(a - b)
    point = macro_auc(y, candidate) - macro_auc(y, baseline)
    return {"point": float(point), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], "valid_resamples": len(diffs),
            "fraction_resamples_positive": float(np.mean(np.array(diffs) > 0))}


def evaluate_gate(y: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, seed_predictions: Mapping[str, np.ndarray], integrity: Mapping[str, bool],
                  *, submissions_remaining: int, rounds: int = 2000) -> dict:
    base_macro, cand_macro = macro_auc(y, baseline), macro_auc(y, candidate)
    boot = paired_bootstrap(y, candidate, baseline, rounds=rounds)
    seeds = {name: macro_auc(y, p) for name, p in seed_predictions.items()}
    checks = {"G1_paired_ci_lower_bound_above_zero": boot["ci95"][0] > 0,
              "G2_every_seed_beats_baseline": bool(seeds) and all(v is not None and v > base_macro for v in seeds.values()),
              "G3_candidate_macro_at_least_0.58": cand_macro is not None and cand_macro >= MIN_CANDIDATE_MACRO,
              "G4_integrity": bool(integrity) and all(integrity.values()),
              "G5_submissions_remaining": submissions_remaining > 0}
    return {"baseline_macro_auc": base_macro, "candidate_macro_auc": cand_macro, "per_seed_macro_auc": seeds, "paired_difference": boot,
            "per_target_baseline": per_target_auc(y, baseline), "per_target_candidate": per_target_auc(y, candidate), "integrity": dict(integrity),
            "checks": checks, "gate_passed": all(checks.values())}


def validate_submission_frame(frame, template) -> list:
    """Problems with a submission frame (empty list = valid): columns/order, IDs/order, row count, finite values in [0, 1], no missing."""
    problems = []
    if tuple(frame.columns) != SUBMISSION_COLUMNS:
        problems.append("columns differ from the competition schema or are out of order")
    if len(frame) != len(template):
        problems.append(f"row count {len(frame)} != template {len(template)}")
    elif list(frame["StudyInstanceUID"].astype(str)) != list(template["StudyInstanceUID"].astype(str)):
        problems.append("study IDs or their order differ from the template")
    if tuple(frame.columns) == SUBMISSION_COLUMNS:
        values = frame[list(TARGET_COLUMNS)]
        if values.isna().any().any():
            problems.append("missing values")
        else:
            arr = values.to_numpy(dtype=float)
            if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
                problems.append("values outside [0, 1] or non-finite")
    return problems
