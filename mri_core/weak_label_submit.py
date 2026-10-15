"""Offline submission inference that keeps the original baseline for every target the new experiment did not improve.

Inference is a separate workflow from training: it only reads checkpoints from local paths, makes no network calls,
and always emits all 12 competition targets in the exact sample_submission column order.

  * Baseline probabilities: mean of the sigmoid outputs of one or more preserved RSNAKneeCNN checkpoints.
  * New probabilities: one checkpoint from the weak-label experiment (any arm).
  * `new_targets` take the new model's probability (optionally blended with the baseline); all other targets keep
    the baseline value untouched.
  * A test study whose MRI cannot be read gets the per-target mean of the other studies' predictions (neutral for
    ranking) and is reported; too many such studies aborts instead of writing a degraded file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .rsna_knee_dataset import PLANE_NAMES, SUBMISSION_COLUMNS, TARGET_COLUMNS, RSNAKneeMetadata
from .rsna_knee_train import RSNAKneeDicomDataset
from .weak_label_models import build_model
from .weak_label_train import predict_probabilities


class SubmissionError(ValueError):
    pass


class SafeTestDataset(RSNAKneeDicomDataset):
    """Test-split dataset that never raises on an unreadable study; failed UIDs are recorded in `failed`."""

    def __init__(self, metadata, config):
        super().__init__(metadata, metadata.test, config, split="test")
        self.failed: list[str] = []

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            size, slices = int(self.config["image_size"]), int(self.config["slices_per_series"])
            planes = len(self.config.get("planes", PLANE_NAMES))
            mask = torch.zeros(planes, slices, dtype=torch.bool)
            mask[:, 0] = True
            uid = str(self.studies.iloc[index]["StudyInstanceUID"])
            self.failed.append(uid)
            return {"images": torch.zeros(planes, slices, 1, size, size), "mask": mask,
                    "labels": torch.zeros(len(TARGET_COLUMNS)), "study_uid": uid}


def load_checkpoint_model(path: Path, device: torch.device):
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if list(state.get("target_columns", [])) != list(TARGET_COLUMNS):
        raise SubmissionError(f"{path}: checkpoint target order does not match the competition schema")
    config = state["config"]
    model = build_model(config.get("arm", "A"), {**config, "pretrained_weights": None}).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), config


def predict_test(path: Path, metadata: RSNAKneeMetadata, device: torch.device, batch_size: int = 8):
    """Return ({uid: probs[12]}, failed_uids) for one checkpoint."""
    model, config = load_checkpoint_model(path, device)
    # some older checkpoints store None for dataset keys; fall back to the documented baseline values
    config = {**{"image_size": 128, "slices_per_series": 9}, **{k: v for k, v in config.items() if v is not None}}
    dataset = SafeTestDataset(metadata, config)
    uids, probs = predict_probabilities(model, dataset, device, batch_size, num_workers=0)
    failed = set(dataset.failed)
    return {u: p for u, p in zip(uids, probs)}, failed


def combine(baseline: np.ndarray, new: Optional[np.ndarray], new_targets: Sequence[str], weight: float = 1.0) -> np.ndarray:
    """Replace (or blend into) only the chosen target columns; every other column is the baseline, untouched."""
    if not 0.0 <= weight <= 1.0:
        raise SubmissionError("blend weight must be in [0, 1]")
    out = baseline.copy()
    if new is None or not new_targets:
        return out
    unknown = set(new_targets) - set(TARGET_COLUMNS)
    if unknown:
        raise SubmissionError(f"unknown targets: {sorted(unknown)}")
    for target in new_targets:
        j = TARGET_COLUMNS.index(target)
        out[:, j] = weight * new[:, j] + (1.0 - weight) * baseline[:, j]
    return out


def validate_submission(frame: pd.DataFrame, template: pd.DataFrame) -> None:
    if tuple(frame.columns) != SUBMISSION_COLUMNS or tuple(template.columns) != SUBMISSION_COLUMNS:
        raise SubmissionError("submission columns must match the sample submission exactly and in order")
    if list(frame["StudyInstanceUID"].astype(str)) != list(template["StudyInstanceUID"].astype(str)):
        raise SubmissionError("submission study IDs/order differ from the sample submission")
    values = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
        raise SubmissionError("probabilities must be finite and within [0, 1]")


APPROVED_NEW_TARGETS = ("ACL", "Medial Meniscus", "Lateral Meniscus")  # Tier 1; widen only with explicit review
log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_submission(metadata: RSNAKneeMetadata, baseline_checkpoints: Sequence[Path], new_checkpoint: Optional[Path],
                   new_targets: Sequence[str], output: Path, *, device: str = "cpu", weight: float = 1.0,
                   max_failed_fraction: float = 0.05, overwrite: bool = False,
                   approved_targets: Sequence[str] = APPROVED_NEW_TARGETS) -> dict:
    """Write submission.csv plus <output>.audit.json (provenance, per-target source, every fallback study)."""
    output = Path(output)
    audit_path = output.with_name(output.name + ".audit.json")
    if output.exists() and not overwrite:
        raise SubmissionError(f"{output} exists; pass overwrite=True")
    if not baseline_checkpoints:
        raise SubmissionError("at least one baseline checkpoint is required")
    not_approved = sorted(set(new_targets) - set(approved_targets))
    if not_approved:
        raise SubmissionError(f"the new model may only replace approved targets; not approved: {not_approved}")
    started = time.perf_counter()
    torch_device = torch.device(device)
    template = metadata.sample_submission
    order = list(template["StudyInstanceUID"].astype(str))
    failed_by_checkpoint: dict = {}
    baseline_runs = []
    for path in baseline_checkpoints:
        preds, failed = predict_test(Path(path), metadata, torch_device)
        baseline_runs.append(preds)
        failed_by_checkpoint[str(path)] = sorted(failed)
    new_preds = None
    if new_checkpoint is not None:
        new_preds, failed = predict_test(Path(new_checkpoint), metadata, torch_device)
        failed_by_checkpoint[str(new_checkpoint)] = sorted(failed)
    failed_all = set().union(*failed_by_checkpoint.values()) if failed_by_checkpoint else set()
    missing = [u for u in order if any(u not in run for run in baseline_runs)]
    if missing:
        raise SubmissionError(f"{len(missing)} sample-submission studies have no test metadata/predictions")
    baseline = np.stack([np.mean([run[u] for run in baseline_runs], axis=0) for u in order])
    new = None if new_preds is None else np.stack([new_preds[u] for u in order])
    merged = combine(baseline, new, new_targets, weight)
    fallbacks = []
    if failed_all:
        if len(failed_all) / len(order) > max_failed_fraction:
            raise SubmissionError(f"{len(failed_all)}/{len(order)} test studies were unreadable; aborting "
                                  f"(limit {max_failed_fraction:.0%})")
        good = np.array([u not in failed_all for u in order])
        fill = merged[good].mean(axis=0) if good.any() else np.full(len(TARGET_COLUMNS), 0.5)
        merged[~good] = fill
        fallbacks = [{"study": u, "fill": "mean_of_readable_studies" if good.any() else "constant_0.5",
                      "unreadable_in": [c for c, f in failed_by_checkpoint.items() if u in f]}
                     for u in order if u in failed_all]
        log.warning("%d test studies were unreadable and received fallback values (see %s)", len(fallbacks), audit_path.name)
    frame = pd.DataFrame({"StudyInstanceUID": template["StudyInstanceUID"].astype("string")})
    for j, target in enumerate(TARGET_COLUMNS):
        frame[target] = merged[:, j].astype(float)
    validate_submission(frame, template)
    seconds = time.perf_counter() - started
    audit = {"output": output.name, "studies": len(order), "seconds": round(seconds, 2),
             "seconds_per_study": round(seconds / max(len(order), 1), 4),
             "target_source": {t: ("new_model" if new is not None and t in new_targets else "baseline") for t in TARGET_COLUMNS},
             "blend_weight": weight, "fallback_count": len(fallbacks), "fallback_studies": fallbacks,
             "checkpoints": {str(p): _sha256(Path(p)) for p in [*baseline_checkpoints, *([new_checkpoint] if new_checkpoint else [])]}}
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {"output": str(output), "audit": str(audit_path), "studies": len(order), "failed_studies": len(failed_all),
            "fallback_studies": [f["study"] for f in fallbacks], "seconds_per_study": audit["seconds_per_study"],
            "targets_from_new_model": [t for t in TARGET_COLUMNS if audit["target_source"][t] == "new_model"],
            "baseline_checkpoints": [str(p) for p in baseline_checkpoints],
            "new_checkpoint": str(new_checkpoint) if new_checkpoint else None}
