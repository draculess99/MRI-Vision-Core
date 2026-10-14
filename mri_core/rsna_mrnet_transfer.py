"""Transfer a shared MRNetCNN image encoder into a fresh 12-target RSNAKneeCNN.

MRNetCNN (3 targets: abnormal/acl/meniscus) and RSNAKneeCNN (12 RSNA targets) are both
thin wrappers around the same encoder architecture in mri_core/cnn_model.py: identical
"encoder.*" Conv2d/GroupNorm stack, differing only in the final "head.*" Linear layer's
output width. This module copies only the encoder weights across; the head is always a
freshly initialized 12-target layer. It never falls back to loading mismatched tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .rsna_knee_model import RSNAKneeCNN

ENCODER_PREFIX = "encoder."
HEAD_PREFIX = "head."


class TransferError(Exception):
    """Base class for MRNet -> RSNA encoder transfer failures."""


class UntrustedCheckpointError(TransferError):
    """The MRNet checkpoint file could not be loaded or is not a recognized state dict."""


@dataclass
class TransferReport:
    """Exact accounting of every tensor considered during the transfer."""

    source_path: str
    copied: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    incompatible: list = field(default_factory=list)
    head_reinitialized: bool = True
    target_num_targets: int = 12

    @property
    def ok(self) -> bool:
        return not self.missing and not self.incompatible

    def summary(self) -> str:
        return (
            f"copied={len(self.copied)} missing={len(self.missing)} "
            f"incompatible={len(self.incompatible)} head_reinitialized={self.head_reinitialized}"
        )


def _extract_state_dict(checkpoint_path: Path):
    """Accept either a raw state_dict or a full training checkpoint dict with a "model" key."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise UntrustedCheckpointError(f"Failed to load {checkpoint_path}: {exc}") from exc

    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint
    raise UntrustedCheckpointError(
        f"{checkpoint_path} is not a recognized state dict or training checkpoint with a 'model' key"
    )


def transfer_mrnet_encoder_to_rsna(
    mrnet_checkpoint_path: Path | str,
    *,
    num_planes: int = 3,
    dropout: float = 0.2,
    device: str = "cpu",
) -> tuple[RSNAKneeCNN, TransferReport]:
    """Build a fresh RSNAKneeCNN and copy only compatible encoder.* tensors from an MRNet checkpoint.

    Returns (model, report). The head is always freshly initialized for 12 RSNA targets;
    head.* tensors from the MRNet source are never inspected or copied, regardless of shape.
    """
    mrnet_checkpoint_path = Path(mrnet_checkpoint_path)
    source_state = _extract_state_dict(mrnet_checkpoint_path)

    model = RSNAKneeCNN(num_planes=num_planes, dropout=dropout)
    target_state = model.state_dict()

    report = TransferReport(source_path=str(mrnet_checkpoint_path), target_num_targets=model.num_targets)

    source_encoder_keys = {k for k in source_state if k.startswith(ENCODER_PREFIX)}
    target_encoder_keys = {k for k in target_state if k.startswith(ENCODER_PREFIX)}

    new_state = dict(target_state)  # head.* stays at its freshly initialized values

    for key in sorted(target_encoder_keys):
        if key not in source_encoder_keys:
            report.missing.append(key)
            continue
        source_tensor = source_state[key]
        target_tensor = target_state[key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape) or source_tensor.dtype != target_tensor.dtype:
            report.incompatible.append(
                (key, f"source={tuple(source_tensor.shape)}/{source_tensor.dtype} "
                      f"target={tuple(target_tensor.shape)}/{target_tensor.dtype}")
            )
            continue
        new_state[key] = source_tensor.clone()
        report.copied.append(key)

    # Anything in the source encoder that the target architecture doesn't have is noted,
    # never applied (prevents silently reusing tensors that don't belong to this model).
    extra_in_source = sorted(source_encoder_keys - target_encoder_keys)
    for key in extra_in_source:
        report.incompatible.append((key, "present in source but not in target architecture"))

    # Explicitly never touch head.* from the source, no matter its shape.
    for key in target_state:
        if key.startswith(HEAD_PREFIX) and key in source_state:
            pass  # deliberately ignored; head is always freshly initialized

    model.load_state_dict(new_state, strict=True)
    model.to(device)
    report.head_reinitialized = True
    return model, report
