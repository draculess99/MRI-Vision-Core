"""Local-only image-derived evidence adapter for MRI copilot.

Accepts user-authorized PNG/JPG/JPEG image pairs uploaded locally.
Processes them in-memory using the existing mri_core pipeline.
Generates deterministic MRIEvidence based on technical measurements.

Privacy: local-only, no persistence, no cloud calls, no claims of diagnosis.
Configuration: LOCAL_IMAGE_EVIDENCE_ENABLED environment variable (default: false).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import MutableMapping, Optional

# Environment variable for enabling local image mode
ENV_LOCAL_IMAGE_EVIDENCE_ENABLED = "LOCAL_IMAGE_EVIDENCE_ENABLED"

from mri_core.loader import load_image
from mri_core.pipeline import process_mri_image
from hackathon.amazon_alexa.tools import MRIEvidence, ReviewStatus


# Technical thresholds for deterministic classification
# These are measurement-based rules, not clinical interpretations.
FOREGROUND_CHANGE_THRESHOLD = 15.0  # percentage points
INTENSITY_CHANGE_THRESHOLD = 20.0   # counts
MIN_FOREGROUND_PERCENT = 10.0       # sufficient technical coverage
MIN_QUALITY_SCORE = 0.50            # technical sufficiency heuristic


def is_local_image_evidence_enabled(env: Optional[dict[str, str]] = None) -> bool:
    """Check if local image evidence mode is enabled.

    Args:
        env: Environment dict (defaults to os.environ)

    Returns:
        True only if LOCAL_IMAGE_EVIDENCE_ENABLED is explicitly 'true'
    """
    env = os.environ if env is None else env
    value = env.get(ENV_LOCAL_IMAGE_EVIDENCE_ENABLED, "false").strip().lower()
    return value == "true"


def clear_local_image_evidence_state(state: MutableMapping) -> None:
    """Remove the in-memory local evidence and dependent cached UI state.

    Uploaded image bytes are intentionally not copied into this state. This helper
    exists so the clear-session control is deterministic and directly testable.
    """
    for key in (
        "local_image_evidence",
        "llm_explanation",
        "llm_explanation_case_id",
        "last_voice_result",
        "voice_speech",
    ):
        state.pop(key, None)


class LocalImageEvidenceAdapter:
    """Adapter to generate deterministic evidence from local image pairs."""

    @staticmethod
    def generate_evidence_from_bytes(
        baseline_bytes: bytes,
        current_bytes: bytes,
    ) -> MRIEvidence:
        """Generate deterministic evidence from baseline/current image bytes.

        Args:
            baseline_bytes: PNG/JPG/JPEG bytes for baseline image
            current_bytes: PNG/JPG/JPEG bytes for current image

        Returns:
            Frozen MRIEvidence with deterministic technical assessment

        Raises:
            ValueError: If images cannot be loaded or processed
        """
        # Load images from bytes (in-memory, no disk I/O)
        baseline_image = load_image(baseline_bytes)
        current_image = load_image(current_bytes)

        # Process both images through the existing OpenCV pipeline
        baseline_result = process_mri_image(baseline_image)
        current_result = process_mri_image(current_image)

        baseline_features = baseline_result["features"]
        current_features = current_result["features"]

        # Extract measurements
        baseline_fg = baseline_features["foreground_percentage"]
        current_fg = current_features["foreground_percentage"]
        baseline_intensity = baseline_features["mean_intensity"]
        current_intensity = current_features["mean_intensity"]

        # Calculate changes
        fg_change = abs(current_fg - baseline_fg)
        intensity_change = abs(current_intensity - baseline_intensity)

        # Determine technical quality (heuristic, not clinical)
        # Quality based on foreground coverage and measurement stability
        baseline_quality = LocalImageEvidenceAdapter._calculate_technical_quality(
            baseline_features
        )
        current_quality = LocalImageEvidenceAdapter._calculate_technical_quality(
            current_features
        )
        quality_score = min(baseline_quality, current_quality)

        # Determine status based on technical thresholds
        status, change_description = LocalImageEvidenceAdapter._classify_evidence(
            baseline_fg,
            current_fg,
            baseline_intensity,
            current_intensity,
            fg_change,
            intensity_change,
            quality_score,
        )

        # Generate timestamps
        now = datetime.now(timezone.utc)
        baseline_date = now.strftime("%Y-%m-%d")
        current_date = now.strftime("%Y-%m-%d")

        return MRIEvidence(
            case_id="local-image-pair",
            baseline_date=baseline_date,
            current_date=current_date,
            quality_score=quality_score,
            foreground_percentage=current_fg,
            mean_intensity=current_intensity,
            status=status,
            change_description=change_description,
        )

    @staticmethod
    def _calculate_technical_quality(features: dict) -> float:
        """Calculate a technical-sufficiency heuristic, not clinical quality.

        Args:
            features: Feature dict from mri_core.process_mri_image

        Returns:
            Quality score (0.0-1.0) based on measurement confidence
        """
        fg_percent = features.get("foreground_percentage", 0.0)

        # Score based on foreground coverage
        # Good coverage = high confidence in measurement
        if fg_percent >= 40:
            return 0.95
        elif fg_percent >= 25:
            return 0.85
        elif fg_percent >= MIN_FOREGROUND_PERCENT:
            return 0.70
        else:
            return 0.50

    @staticmethod
    def _classify_evidence(
        baseline_fg: float,
        current_fg: float,
        baseline_intensity: float,
        current_intensity: float,
        fg_change: float,
        intensity_change: float,
        quality_score: float,
    ) -> tuple[ReviewStatus, str]:
        """Classify evidence based on technical thresholds.

        Args:
            baseline_fg: Baseline foreground percentage
            current_fg: Current foreground percentage
            baseline_intensity: Baseline mean intensity
            current_intensity: Current mean intensity
            fg_change: Absolute foreground change
            intensity_change: Absolute intensity change
            quality_score: Technical sufficiency score

        Returns:
            (ReviewStatus, change_description)
        """
        # Low technical quality → UNCERTAIN
        if quality_score < MIN_QUALITY_SCORE:
            return (
                ReviewStatus.UNCERTAIN,
                f"Technical sufficiency is low (quality: {quality_score:.2f}); "
                "measurements may not be reliable.",
            )

        # Significant changes → NEEDS_REVIEW
        if fg_change >= FOREGROUND_CHANGE_THRESHOLD or intensity_change >= INTENSITY_CHANGE_THRESHOLD:
            changes = []
            if fg_change >= FOREGROUND_CHANGE_THRESHOLD:
                changes.append(f"foreground {fg_change:.1f}% (baseline {baseline_fg:.1f}% → current {current_fg:.1f}%)")
            if intensity_change >= INTENSITY_CHANGE_THRESHOLD:
                changes.append(f"mean intensity {intensity_change:.1f} (baseline {baseline_intensity:.1f} → current {current_intensity:.1f})")

            return (
                ReviewStatus.NEEDS_REVIEW,
                "Notable measured changes: " + ", ".join(changes) + ".",
            )

        # No significant changes → CONSISTENT
        return (
            ReviewStatus.CONSISTENT,
            f"Measured values are consistent: foreground {current_fg:.1f}%, "
            f"mean intensity {current_intensity:.1f}.",
        )
