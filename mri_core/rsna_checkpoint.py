"""Resumable training checkpoints for RSNA knee training.

Extends the proven MRNet CNN checkpoint pattern (mri_core/cnn_train.py: save_checkpoint,
load_checkpoint) with optimizer, scheduler, AMP-scaler, RNG, dataset-fingerprint, and
exact study-ID state so an interrupted RSNA training run can resume bit-for-bit instead
of restarting. The older RSNA inference-only checkpoint format (model/config/target_columns/
epoch/validation, written by rsna_knee_train.train_rsna) is left untouched: run_inference()
and run_ensemble_inference() keep reading it exactly as before.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .rsna_knee_dataset import TARGET_COLUMNS

CHECKPOINT_SCHEMA_VERSION = 1

REQUIRED_RESUMABLE_KEYS = (
    "schema_version", "model", "optimizer", "config", "target_columns",
    "epoch", "global_step", "torch_rng", "cuda_rng", "python_rng", "numpy_rng",
    "dataset_fingerprint", "train_study_ids", "valid_study_ids",
)


class CheckpointError(Exception):
    """Base class for resumable-checkpoint failures."""


class CheckpointIncompatibleError(CheckpointError):
    """Checkpoint's target columns, model shape, or config disagree with the current run."""


class CheckpointMalformedError(CheckpointError):
    """Checkpoint file could not be read or is missing required fields."""


@dataclass
class ResumableState:
    """Everything needed to continue a training run exactly where it left off."""

    epoch: int
    global_step: int
    metrics: dict
    best_metric: Optional[float]
    is_best: bool
    config: dict
    target_columns: list
    train_study_ids: list
    valid_study_ids: list
    dataset_fingerprint: str
    optimizer_state: dict
    scheduler_state: Optional[dict] = None
    scaler_state: Optional[dict] = None
    loader_rng_state: Optional[torch.Tensor] = None
    extra: dict = field(default_factory=dict)


def studies_fingerprint(study_ids) -> str:
    """Order-independent hash of a set of study UIDs, for detecting split/data drift."""
    import hashlib
    canonical = "\n".join(sorted(str(s) for s in study_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def labeled_frame_fingerprint(frame) -> str:
    """Hash of study UID + its 12 target values, order-independent (sorted by UID)."""
    import hashlib
    import json as _json

    records = sorted(
        (
            str(row["StudyInstanceUID"]),
            tuple(float(row[t]) for t in TARGET_COLUMNS),
        )
        for _, row in frame.iterrows()
    )
    canonical = _json.dumps(records, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _rng_states():
    numpy_state = np.random.get_state()
    return {
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
    }


def _restore_rng_states(checkpoint):
    torch.set_rng_state(checkpoint["torch_rng"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng"):
        states = checkpoint["cuda_rng"]
        devices = torch.cuda.device_count()
        if len(states) != devices:  # resumed on a different GPU count: restore what exists, warn about the rest
            import warnings
            warnings.warn(f"checkpoint has CUDA RNG state for {len(states)} device(s) but {devices} are visible; "
                          "restoring the overlapping devices only", RuntimeWarning, stacklevel=2)
        for index, state in enumerate(states[:devices]):
            torch.cuda.set_rng_state(state, index)
    random.setstate(_destringify_python_rng(checkpoint["python_rng"]))
    numpy_state = checkpoint["numpy_rng"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))


def _destringify_python_rng(state):
    # torch.save/load round-trips tuples/lists faithfully; python's random.setstate
    # requires the inner sequence to be a tuple, not a list, so normalize defensively.
    if isinstance(state, list):
        return tuple(_destringify_python_rng(item) if isinstance(item, (list, tuple)) else item for item in state)
    if isinstance(state, tuple):
        return tuple(_destringify_python_rng(item) if isinstance(item, (list, tuple)) else item for item in state)
    return state


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    config: dict,
    target_columns,
    epoch: int,
    global_step: int,
    train_study_ids,
    valid_study_ids,
    dataset_fingerprint: str,
    metrics: dict,
    best_metric: Optional[float] = None,
    is_best: bool = False,
    scheduler=None,
    scaler=None,
    loader_generator: Optional[torch.Generator] = None,
    best_suffix: str = "_best",
):
    """Atomically write a fully resumable checkpoint (temp file + rename).

    When is_best is True, also writes a second copy at `<path stem><best_suffix><suffix>`
    so the best-validation checkpoint is never overwritten by a later, worse epoch.
    """
    path = Path(path)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": dict(config),
        "target_columns": list(target_columns),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "train_study_ids": sorted(str(s) for s in train_study_ids),
        "valid_study_ids": sorted(str(s) for s in valid_study_ids),
        "dataset_fingerprint": dataset_fingerprint,
        "metrics": metrics,
        "best_metric": best_metric,
        "is_best": bool(is_best),
        "loader_rng": loader_generator.get_state() if loader_generator is not None else None,
        **_rng_states(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

    if is_best:
        best_path = path.with_name(path.stem + best_suffix + path.suffix)
        best_temporary = best_path.with_suffix(best_path.suffix + ".tmp")
        torch.save(payload, best_temporary)
        best_temporary.replace(best_path)


def load_checkpoint(
    path: Path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    expected_target_columns=None,
    expected_dataset_fingerprint=None,
    restore_rng: bool = True,
    device="cpu",
) -> ResumableState:
    """Load a resumable checkpoint, validating compatibility before touching any state.

    Rejects (raises CheckpointIncompatibleError) rather than silently loading when:
      - target_columns differ from expected_target_columns (if given)
      - dataset_fingerprint differs from expected_dataset_fingerprint (if given)
      - the model's state_dict shapes do not match the checkpoint's (strict=True)
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointMalformedError(f"No checkpoint file at {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointMalformedError(f"Failed to load checkpoint {path}: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise CheckpointMalformedError(f"Checkpoint {path} did not contain a dict")
    missing = [key for key in REQUIRED_RESUMABLE_KEYS if key not in checkpoint]
    if missing:
        raise CheckpointMalformedError(f"Checkpoint {path} is missing required fields: {missing}")
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointIncompatibleError(
            f"Checkpoint {path} schema_version={checkpoint['schema_version']} "
            f"!= supported {CHECKPOINT_SCHEMA_VERSION}"
        )

    if expected_target_columns is not None and list(checkpoint["target_columns"]) != list(expected_target_columns):
        raise CheckpointIncompatibleError(
            f"Checkpoint {path} target_columns {checkpoint['target_columns']!r} "
            f"do not match expected {list(expected_target_columns)!r}"
        )
    if expected_dataset_fingerprint is not None and checkpoint["dataset_fingerprint"] != expected_dataset_fingerprint:
        raise CheckpointIncompatibleError(
            f"Checkpoint {path} dataset_fingerprint {checkpoint['dataset_fingerprint']!r} "
            f"does not match current dataset {expected_dataset_fingerprint!r} "
            "(labels, splits, or local DICOM set may have changed since this checkpoint was written)"
        )

    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except (RuntimeError, ValueError) as exc:
        raise CheckpointIncompatibleError(
            f"Checkpoint {path} model weights are not compatible with the given model: {exc}"
        ) from exc
    model.to(device)

    if optimizer is not None:
        if checkpoint.get("optimizer") is None:
            raise CheckpointIncompatibleError(f"Checkpoint {path} has no optimizer state to restore")
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        if checkpoint.get("scheduler") is None:
            raise CheckpointIncompatibleError(f"Checkpoint {path} has no scheduler state to restore")
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None:
        if checkpoint.get("scaler") is None:
            raise CheckpointIncompatibleError(f"Checkpoint {path} has no AMP scaler state to restore")
        scaler.load_state_dict(checkpoint["scaler"])

    if restore_rng:
        _restore_rng_states(checkpoint)

    loader_rng_state = checkpoint.get("loader_rng")

    return ResumableState(
        epoch=int(checkpoint["epoch"]),
        global_step=int(checkpoint["global_step"]),
        metrics=checkpoint.get("metrics", {}),
        best_metric=checkpoint.get("best_metric"),
        is_best=bool(checkpoint.get("is_best", False)),
        config=checkpoint["config"],
        target_columns=list(checkpoint["target_columns"]),
        train_study_ids=list(checkpoint["train_study_ids"]),
        valid_study_ids=list(checkpoint["valid_study_ids"]),
        dataset_fingerprint=checkpoint["dataset_fingerprint"],
        optimizer_state=checkpoint.get("optimizer"),
        scheduler_state=checkpoint.get("scheduler"),
        scaler_state=checkpoint.get("scaler"),
        loader_rng_state=loader_rng_state,
    )
