"""Training-health diagnostics for the weak-label pipeline: constant-prediction baselines, shuffled-label control, leakage report.

Everything here is computed from the same masked, pos_weighted loss and the same masked AUC as training, so the reference numbers are
directly comparable with a run's validation loss. Nothing touches gold labels or test data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .rsna_knee_dataset import TARGET_COLUMNS
from .weak_label_train import compute_pos_weight, masked_bce


def shuffled_rows(n: int, seed: int) -> np.ndarray:
    """A deterministic permutation of the training rows that moves every row (no fixed points when n > 1)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    for _ in range(100):
        if n < 2 or not (perm == np.arange(n)).any():
            break
        perm = rng.permutation(n)
    return perm


def shuffle_labels(values: np.ndarray, masks: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permute (values, masks) jointly across studies: keeps prevalence and the mask pattern, destroys the study-label link."""
    perm = shuffled_rows(len(values), seed)
    return values[perm].copy(), masks[perm].copy(), perm


def _loss_at(probs_per_target: np.ndarray, values: np.ndarray, masks: np.ndarray, pos_weight: torch.Tensor) -> float:
    p = np.clip(probs_per_target, 1e-6, 1 - 1e-6)
    logits = torch.from_numpy(np.log(p / (1 - p))).float().expand(len(values), -1).contiguous()
    return float(masked_bce(logits, torch.from_numpy(np.nan_to_num(values, nan=0.0)).float(), torch.from_numpy(masks).float(),
                            pos_weight, torch.ones(len(TARGET_COLUMNS))))


def constant_baselines(train_values: np.ndarray, train_masks: np.ndarray, val_values: np.ndarray, val_masks: np.ndarray,
                       max_pos_weight: float = 10.0) -> dict:
    """Validation loss (the training objective, training pos_weight) of predictors that ignore the images.

    * p05: every probability 0.5.
    * train_prevalence: every probability equals the training positive rate of that target.
    * weighted_optimum: the constant that minimises the pos_weighted training loss (the best any image-blind model can do on train).
    ROC-AUC of any constant predictor is 0.5 by definition."""
    pos_weight = compute_pos_weight(train_values, train_masks, max_pos_weight)
    on = train_masks == 1
    pos = ((train_values == 1) & on).sum(axis=0).astype(np.float64)
    neg = ((train_values == 0) & on).sum(axis=0).astype(np.float64)
    prevalence = np.where(pos + neg > 0, pos / np.maximum(pos + neg, 1), 0.5)
    pw = pos_weight.numpy().astype(np.float64)
    weighted = np.where(pos + neg > 0, pw * prevalence / (pw * prevalence + (1 - prevalence)), 0.5)
    result = {"pos_weight": {t: float(w) for t, w in zip(TARGET_COLUMNS, pw)},
              "train_prevalence": {t: float(p) for t, p, n in zip(TARGET_COLUMNS, prevalence, pos + neg) if n > 0},
              "constant_auc": 0.5}
    for name, probs in (("p05", np.full(len(TARGET_COLUMNS), 0.5)), ("train_prevalence", prevalence), ("weighted_optimum", weighted)):
        result[f"val_loss_{name}"] = _loss_at(probs, val_values, val_masks, pos_weight)
        result[f"train_loss_{name}"] = _loss_at(probs, train_values, train_masks, pos_weight)
    return result


def leakage_report(train_ids: Sequence[str], val_ids: Sequence[str], gold_ids: Sequence[str] = (), test_ids: Sequence[str] = (),
                   allow_val_equals_train: bool = False) -> dict:
    train, val = list(map(str, train_ids)), list(map(str, val_ids))
    overlap = sorted(set(train) & set(val))
    report = {"train": len(train), "validation": len(val), "train_duplicates": len(train) - len(set(train)),
              "validation_duplicates": len(val) - len(set(val)), "train_validation_overlap": len(overlap),
              "gold_in_train_or_validation": len((set(train) | set(val)) & set(map(str, gold_ids))),
              "test_in_train_or_validation": len((set(train) | set(val)) & set(map(str, test_ids))),
              "val_equals_train_by_design": bool(allow_val_equals_train)}
    report["clean"] = bool(report["train_duplicates"] == 0 and report["validation_duplicates"] == 0
                           and report["gold_in_train_or_validation"] == 0 and report["test_in_train_or_validation"] == 0
                           and (report["train_validation_overlap"] == 0 or allow_val_equals_train))
    return report
