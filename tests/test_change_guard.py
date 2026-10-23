"""
Tests for ChangeGuard deterministic baseline comparison.

All tests are offline, non-network, deterministic.
"""

import numpy as np
import pytest

from hackathon.opencv_aws.change_guard import ChangeGuard, ComparisonResult


class TestChangeGuardComparison:
    """Tests for ChangeGuard.compare() baseline comparison."""

    def test_identical_images_stable(self):
        """Identical baseline and current images return STABLE."""
        baseline = np.ones((64, 64), dtype=np.uint8) * 100
        current = np.ones((64, 64), dtype=np.uint8) * 100

        result = ChangeGuard.compare(baseline, current)

        assert result.status == "STABLE"
        assert result.foreground_delta == 0.0
        assert result.mean_intensity_delta == 0.0

    def test_no_baseline_invalid(self):
        """Missing baseline returns INVALID status."""
        current = np.ones((64, 64), dtype=np.uint8) * 100

        result = ChangeGuard.compare(None, current)

        assert result.status == "INVALID"
        assert "No baseline" in result.reason

    def test_shape_mismatch_invalid(self):
        """Mismatched shapes return INVALID status."""
        baseline = np.ones((64, 64), dtype=np.uint8) * 100
        current = np.ones((32, 32), dtype=np.uint8) * 100

        result = ChangeGuard.compare(baseline, current)

        assert result.status == "INVALID"
        assert "Shape mismatch" in result.reason

    def test_small_delta_stable(self):
        """Small changes below thresholds return STABLE."""
        baseline = np.ones((64, 64), dtype=np.uint8) * 100
        # Change only a few pixels by small amounts
        current = baseline.copy()
        current[0:5, 0:5] = 105  # 25 pixels changed by 5 counts

        result = ChangeGuard.compare(baseline, current)

        assert result.status == "STABLE"
        # 25 pixels out of 4096 = 0.61%, less than 15% threshold
        assert result.foreground_delta < 15.0
        # Mean change is 25 * 5 / 4096 ≈ 0.03, less than 20 threshold
        assert result.mean_intensity_delta < 20.0

    def test_large_mean_intensity_review(self):
        """Large mean intensity change triggers REVIEW."""
        baseline = np.ones((64, 64), dtype=np.uint8) * 100
        current = np.ones((64, 64), dtype=np.uint8) * 130  # 30 count difference

        result = ChangeGuard.compare(baseline, current)

        assert result.status == "REVIEW"
        assert result.mean_intensity_delta >= ChangeGuard.MEAN_INTENSITY_DELTA_THRESHOLD

    def test_large_foreground_delta_review(self):
        """Large foreground change (≥10 counts) triggers REVIEW."""
        baseline = np.ones((64, 64), dtype=np.uint8) * 100
        current = baseline.copy()
        # Change 20% of pixels by ≥10 counts (exceeds 15% threshold)
        num_pixels_to_change = int(64 * 64 * 0.20)
        current.flat[:num_pixels_to_change] = 115

        result = ChangeGuard.compare(baseline, current)

        assert result.status == "REVIEW"
        assert result.foreground_delta >= ChangeGuard.FOREGROUND_DELTA_THRESHOLD

    def test_thresholds_exact_boundary(self):
        """Test behavior at exact threshold boundaries."""
        baseline = np.ones((100, 100), dtype=np.uint8) * 100
        current = baseline.copy()

        # Set exactly 15% of pixels to change by ≥10 (at foreground threshold)
        num_pixels = int(100 * 100 * 0.15)
        current.flat[:num_pixels] = 115

        result = ChangeGuard.compare(baseline, current)

        # At or above threshold should trigger REVIEW
        assert result.foreground_delta >= ChangeGuard.FOREGROUND_DELTA_THRESHOLD

    def test_deterministic_reproducibility(self):
        """Same images always produce same comparison result."""
        baseline = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        current = baseline.copy()

        result1 = ChangeGuard.compare(baseline, current)
        result2 = ChangeGuard.compare(baseline, current)

        assert result1.status == result2.status
        assert result1.foreground_delta == result2.foreground_delta
        assert result1.mean_intensity_delta == result2.mean_intensity_delta

    def test_dtype_conversion(self):
        """Non-uint8 arrays are converted safely."""
        baseline = np.ones((32, 32), dtype=np.float32) * 100
        current = np.ones((32, 32), dtype=np.int32) * 105

        result = ChangeGuard.compare(baseline, current)

        assert result.status == "STABLE"
        assert result.mean_intensity_delta < 20.0

    def test_max_possible_delta(self):
        """Maximum possible delta (all pixels change 255 → 0)."""
        baseline = np.ones((8, 8), dtype=np.uint8) * 255
        current = np.zeros((8, 8), dtype=np.uint8)

        result = ChangeGuard.compare(baseline, current)

        assert result.status == "REVIEW"
        assert result.foreground_delta == 100.0
        assert result.mean_intensity_delta == 255.0


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_result_serialization(self):
        """ComparisonResult converts to dict correctly."""
        result = ComparisonResult(
            status="STABLE",
            foreground_delta=5.5,
            mean_intensity_delta=10.0,
        )

        d = result.to_dict()

        assert d["status"] == "STABLE"
        assert d["foreground_delta"] == 5.5
        assert d["mean_intensity_delta"] == 10.0

    def test_invalid_result_serialization(self):
        """INVALID result serializes with reason."""
        result = ComparisonResult(
            status="INVALID",
            reason="Test reason",
        )

        d = result.to_dict()

        assert d["status"] == "INVALID"
        assert d["reason"] == "Test reason"
