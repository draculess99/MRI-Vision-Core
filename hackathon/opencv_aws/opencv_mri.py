from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class OpenCVMRIResult:
    original: np.ndarray
    enhanced: np.ndarray
    edges: np.ndarray
    overlay: np.ndarray


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize an MRI slice to uint8 [0, 255].
    """
    arr = np.asarray(image, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale MRI slice, got shape {arr.shape}")

    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.where(finite, arr, 0.0)

    low = float(arr.min())
    high = float(arr.max())

    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = (arr - low) / (high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE contrast enhancement to an MRI slice.
    """
    gray = normalize_to_uint8(image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


def detect_edges(image: np.ndarray) -> np.ndarray:
    """
    Produce a structural edge map using Gaussian smoothing + Canny.
    """
    enhanced = enhance_contrast(image)

    blurred = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred,
        threshold1=40,
        threshold2=120,
    )

    return edges


def create_overlay(
    image: np.ndarray,
    edges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Overlay detected structural edges on top of the MRI slice.
    """
    enhanced = enhance_contrast(image)

    if edges is None:
        edges = detect_edges(image)

    base = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2BGR,
    )

    overlay = base.copy()

    # Mark structural edges.
    overlay[edges > 0] = (0, 255, 0)

    return overlay


def process_mri_slice(image: np.ndarray) -> OpenCVMRIResult:
    """
    Run the complete OpenCV MRI preprocessing pipeline.
    """
    original = normalize_to_uint8(image)
    enhanced = enhance_contrast(image)
    edges = detect_edges(image)
    overlay = create_overlay(image, edges)

    return OpenCVMRIResult(
        original=original,
        enhanced=enhanced,
        edges=edges,
        overlay=overlay,
    )


def save_result(
    result: OpenCVMRIResult,
    output_dir: Path | str,
    prefix: str = "mri",
) -> dict[str, Path]:
    """
    Save OpenCV processing artifacts for demos and validation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "original": output_dir / f"{prefix}_original.png",
        "enhanced": output_dir / f"{prefix}_enhanced.png",
        "edges": output_dir / f"{prefix}_edges.png",
        "overlay": output_dir / f"{prefix}_overlay.png",
    }

    cv2.imwrite(str(paths["original"]), result.original)
    cv2.imwrite(str(paths["enhanced"]), result.enhanced)
    cv2.imwrite(str(paths["edges"]), result.edges)
    cv2.imwrite(str(paths["overlay"]), result.overlay)

    return paths