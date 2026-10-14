"""Bridge for the externally-trained Kaggle `kneeabdet-baseline-v1` checkpoints.

These five `foldN-knee-mil-best.ckpt` files are real trained weights, but for a different
architecture than the shared `mri_core.rsna_knee_model.RSNAKneeCNN` (MRNet-style encoder):
a `timm` `efficientnet_b0` backbone with gated attention-MIL pooling over DICOM slices,
trained via PyTorch Lightning. `mri_core`'s own checkpoint loaders (`rsna_knee_model`,
`rsna_checkpoint`) cannot load them -- the state_dict keys and shapes do not match.

This module is a minimal, inference-only port of that architecture (model class, DICOM
slice reading, slice-stack preprocessing) so the real weights can be run on a real local
study. It does not touch `mri_core`'s own model/checkpoint code and is not used by
`hackathon/amd/run_inference.py`, which stays on the native RSNAKneeCNN path.

Preprocessing must match the original training notebook exactly or the sigmoid outputs
are meaningless: 24 slices sampled evenly by index, each resized to 224x224, 1st/99th
percentile intensity normalization per study, then single-channel ImageNet mean/std
standardization (0.449 / 0.226 -- the three ImageNet channel stats averaged for in_chans=1).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

TARGET_COLUMNS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)
IMAGE_SIZE = 224
N_SLICES = 24
IMAGENET_MEAN = 0.449
IMAGENET_STD = 0.226
BACKBONE_NAME = "efficientnet_b0"


class BridgeError(Exception):
    """Checkpoint or DICOM input is not compatible with this bridge."""


class AttentionMILPooling(nn.Module):
    """Gated attention pooling over the slice axis (Ilse et al. attention-MIL)."""

    def __init__(self, in_features: int, hidden_features: int = 128) -> None:
        super().__init__()
        self.attn_v = nn.Sequential(nn.Linear(in_features, hidden_features), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(in_features, hidden_features), nn.Sigmoid())
        self.attn_w = nn.Linear(hidden_features, 1)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn_w(self.attn_v(feats) * self.attn_u(feats))
        weights = torch.softmax(scores, dim=1)
        return (feats * weights).sum(dim=1), weights.squeeze(-1)


class KneeMILClassifier(nn.Module):
    """Inference-only port of the notebook's `KneeMILClassifier` (no Lightning dependency)."""

    def __init__(self, model_name: str = BACKBONE_NAME, num_labels: int = 12) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=1)
        self.pooling = AttentionMILPooling(self.backbone.num_features)
        self.head = nn.Linear(self.backbone.num_features, num_labels)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        batch, slices, channels, height, width = volume.shape
        feats = self.backbone(volume.view(batch * slices, channels, height, width)).view(batch, slices, -1)
        pooled, _ = self.pooling(feats)
        return self.head(pooled)


def load_bridge_checkpoint(path: Path, *, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load one `foldN-knee-mil-best.ckpt` Lightning checkpoint into the plain inference model.

    Raises BridgeError if the file is not a recognizable Lightning checkpoint for this
    architecture (missing keys, wrong shapes, wrong backbone).
    """
    path = Path(path)
    if not path.is_file():
        raise BridgeError(f"No checkpoint file at {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise BridgeError(f"Failed to load checkpoint {path}: {exc}") from exc

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise BridgeError(f"Checkpoint {path} is not a PyTorch Lightning checkpoint (no 'state_dict')")

    hparams = checkpoint.get("hyper_parameters", {})
    model_name = hparams.get("model_name", BACKBONE_NAME)
    num_labels = hparams.get("num_labels", len(TARGET_COLUMNS))
    if num_labels != len(TARGET_COLUMNS):
        raise BridgeError(
            f"Checkpoint {path} has num_labels={num_labels}, expected {len(TARGET_COLUMNS)}"
        )

    model = KneeMILClassifier(model_name=model_name, num_labels=num_labels)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except (RuntimeError, ValueError) as exc:
        raise BridgeError(f"Checkpoint {path} weights are not compatible with {model_name} MIL bridge: {exc}") from exc
    model.to(device).eval()

    provenance = {
        "architecture": f"{model_name} backbone + gated attention-MIL pooling + linear head (12 targets)",
        "source": "Kaggle kneeabdet-baseline-v1 (bridged, not native mri_core.RSNAKneeCNN)",
        "lightning_epoch": checkpoint.get("epoch"),
        "lightning_global_step": checkpoint.get("global_step"),
        "pytorch_lightning_version": checkpoint.get("pytorch-lightning_version"),
    }
    return model, provenance


def sorted_slice_paths(dir_series: Path) -> list[str]:
    """Order a series' `.dcm` files by InstanceNumber, falling back to filename order."""
    import pydicom

    paths = sorted(glob.glob(os.path.join(str(dir_series), "*.dcm")))
    if not paths:
        return []
    numbers = []
    for path in paths:
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True)
            value = getattr(header, "InstanceNumber", None)
            numbers.append(float(value) if value is not None else np.nan)
        except (OSError, ValueError, TypeError, pydicom.errors.InvalidDicomError):
            numbers.append(np.nan)
    if np.isnan(numbers).any():
        return paths
    return [path for _, path in sorted(zip(numbers, paths))]


def read_dicom_slice(path: str) -> Optional[np.ndarray]:
    """Decode one DICOM file to a 2D float32 array, or None if it cannot be read."""
    import pydicom

    try:
        arr = pydicom.dcmread(path).pixel_array.astype(np.float32)
    except (
        OSError, ValueError, TypeError, RuntimeError, NotImplementedError,
        AttributeError, pydicom.errors.InvalidDicomError,
    ):
        return None
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    return arr if arr.ndim == 2 else None


def read_dicom_volume(dir_series: Path, *, n_slices: int = N_SLICES, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """Read a DICOM series into a normalized (n_slices, 1, image_size, image_size) tensor.

    Raises BridgeError if no slice in the series could be decoded (rather than silently
    returning an all-zero volume, which would produce a meaningless but plausible-looking
    prediction).
    """
    paths = sorted_slice_paths(dir_series)
    if not paths:
        raise BridgeError(f"No .dcm files found under {dir_series}")

    indices = np.linspace(0, len(paths) - 1, n_slices).round().astype(int)
    cache: dict[int, torch.Tensor] = {}
    slices = []
    any_decoded = False
    for i in indices:
        if i not in cache:
            arr = read_dicom_slice(paths[i])
            if arr is None:
                tensor = torch.zeros(1, 1, image_size, image_size)
            else:
                any_decoded = True
                tensor = torch.nn.functional.interpolate(
                    torch.from_numpy(arr)[None, None], size=(image_size, image_size),
                    mode="bilinear", align_corners=False,
                )
            cache[i] = tensor
        slices.append(cache[i])
    if not any_decoded:
        raise BridgeError(f"No slice under {dir_series} could be decoded from {len(paths)} files")

    volume = torch.cat(slices, dim=0).float()
    lo, hi = np.percentile(volume.numpy(), [1.0, 99.0])
    volume = ((volume - lo) / max(float(hi - lo), 1e-6)).clamp(0.0, 1.0)
    volume = (volume - IMAGENET_MEAN) / IMAGENET_STD
    return volume


def run_study_inference(model: nn.Module, series_dir: Path) -> dict:
    """Run the bridged model on one DICOM series directory; returns probabilities + shape info."""
    volume = read_dicom_volume(series_dir)
    batch = volume.unsqueeze(0)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).squeeze(0)
    return {
        "probabilities": {col: float(probs[i]) for i, col in enumerate(TARGET_COLUMNS)},
        "n_slices": volume.shape[0],
        "image_size": volume.shape[-1],
    }
