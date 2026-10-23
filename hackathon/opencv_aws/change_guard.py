"""
ChangeGuard: Deterministic baseline comparison for MRI processing.

Non-diagnostic research signal. Compares current processed image against
a baseline using fixed thresholds to detect significant intensity changes.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ComparisonResult:
    """Result of baseline vs current image comparison."""

    status: str  # "STABLE", "REVIEW", or "INVALID"
    foreground_delta: Optional[float] = None  # percentage points
    mean_intensity_delta: Optional[float] = None  # counts
    reason: Optional[str] = None

    def to_dict(self):
        """Convert to dict for serialization."""
        return {
            "status": self.status,
            "foreground_delta": self.foreground_delta,
            "mean_intensity_delta": self.mean_intensity_delta,
            "reason": self.reason,
        }


class ChangeGuard:
    """Deterministic baseline comparison using fixed thresholds."""

    # Review thresholds (research signal, non-diagnostic)
    FOREGROUND_DELTA_THRESHOLD = 15.0  # percentage points
    MEAN_INTENSITY_DELTA_THRESHOLD = 20.0  # counts

    @staticmethod
    def compare(
        baseline: Optional[np.ndarray],
        current: np.ndarray,
    ) -> ComparisonResult:
        """
        Compare current image against baseline using deterministic thresholds.

        Args:
            baseline: Previous processed image (uint8). None = no baseline.
            current: Current processed image (uint8)

        Returns:
            ComparisonResult with status (STABLE/REVIEW/INVALID),
            deltas, and reason.

        Non-diagnostic. Research signal only.
        """
        if baseline is None:
            return ComparisonResult(
                status="INVALID",
                reason="No baseline image available for comparison",
            )

        # Validate shapes match
        if baseline.shape != current.shape:
            return ComparisonResult(
                status="INVALID",
                reason=f"Shape mismatch: baseline {baseline.shape} vs current {current.shape}",
            )

        # Validate dtypes
        if baseline.dtype != np.uint8:
            baseline = baseline.astype(np.uint8)
        if current.dtype != np.uint8:
            current = current.astype(np.uint8)

        try:
            # Calculate mean-intensity delta
            baseline_mean = float(baseline.mean())
            current_mean = float(current.mean())
            mean_intensity_delta = abs(current_mean - baseline_mean)

            # Calculate foreground delta (percentage of pixels where intensity changed by ≥10)
            intensity_change = np.abs(baseline.astype(float) - current.astype(float))
            foreground_pixels = (intensity_change >= 10).sum()
            total_pixels = baseline.size
            foreground_delta = (foreground_pixels / total_pixels) * 100.0

            # Determine status based on thresholds
            if (
                foreground_delta >= ChangeGuard.FOREGROUND_DELTA_THRESHOLD
                or mean_intensity_delta >= ChangeGuard.MEAN_INTENSITY_DELTA_THRESHOLD
            ):
                status = "REVIEW"
                reason = "Thresholds exceeded: baseline changed significantly"
            else:
                status = "STABLE"
                reason = None

            return ComparisonResult(
                status=status,
                foreground_delta=foreground_delta,
                mean_intensity_delta=mean_intensity_delta,
                reason=reason,
            )

        except Exception as e:
            return ComparisonResult(
                status="INVALID",
                reason=f"Comparison error: {str(e)}",
            )
