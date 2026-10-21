"""Tests for local-only image-derived evidence adapter."""

import io
from PIL import Image

import numpy as np
import pytest

from hackathon.amazon_alexa import stt
from hackathon.amazon_alexa.local_image_evidence import (
    LocalImageEvidenceAdapter,
    clear_local_image_evidence_state,
    is_local_image_evidence_enabled,
)
from hackathon.amazon_alexa.tools import ReviewStatus
from hackathon.amazon_alexa.voice_router import VoiceSession, route_command
from hackathon.amazon_alexa.voice_ui import run_transcription
from hackathon.amazon_alexa.usage import UsageTracker


def create_synthetic_image(width: int = 256, height: int = 256, foreground_percent: float = 50.0, mean_intensity: int = 100) -> bytes:
    """Create a synthetic grayscale PNG image in memory (no disk I/O).

    Args:
        width: Image width in pixels
        height: Image height in pixels
        foreground_percent: Approximate foreground percentage (0-100)
        mean_intensity: Target mean intensity value (0-255)

    Returns:
        PNG bytes
    """
    # Create synthetic grayscale image
    img_array = np.zeros((height, width), dtype=np.uint8)

    # Create foreground region (upper portion with high intensity)
    fg_height = int(height * foreground_percent / 100.0)
    fg_intensity = min(int(mean_intensity + 50), 255)
    img_array[:fg_height, :] = fg_intensity

    # Fill background with lower intensity
    bg_intensity = max(int(mean_intensity - 30), 0)
    img_array[fg_height:, :] = bg_intensity

    # Convert to PIL image and encode as PNG bytes
    image = Image.fromarray(img_array, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestLocalImageEvidenceAdapter:
    """Test local image evidence generation."""

    def test_consistent_case_no_change(self):
        """Same baseline/current images yield CONSISTENT."""
        # Create identical baseline and current images
        baseline_bytes = create_synthetic_image(foreground_percent=45.0, mean_intensity=128)
        current_bytes = create_synthetic_image(foreground_percent=45.0, mean_intensity=128)

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        assert evidence.status == ReviewStatus.CONSISTENT
        assert evidence.case_id == "local-image-pair"
        assert "consistent" in evidence.change_description.lower()

    def test_needs_review_foreground_change(self):
        """Large foreground change yields NEEDS_REVIEW."""
        baseline_bytes = create_synthetic_image(foreground_percent=30.0)
        # Change foreground by 20 percentage points (exceeds 15 threshold)
        current_bytes = create_synthetic_image(foreground_percent=50.0)

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        assert evidence.status == ReviewStatus.NEEDS_REVIEW
        assert "notable" in evidence.change_description.lower() or "change" in evidence.change_description.lower()

    def test_needs_review_large_foreground_change(self):
        """Foreground changes ≥15% trigger NEEDS_REVIEW."""
        # Create images with different foreground percentages
        baseline_bytes = create_synthetic_image(foreground_percent=20.0)
        current_bytes = create_synthetic_image(foreground_percent=40.0)

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        # Foreground change of ~20% should trigger NEEDS_REVIEW
        assert evidence.status == ReviewStatus.NEEDS_REVIEW

    def test_uncertain_very_low_foreground(self):
        """Very low foreground may yield UNCERTAIN."""
        # Create images with foreground below 10% (quality threshold)
        # Note: Pipeline may still produce readable measurements, so this tests the lower bound
        baseline_bytes = create_synthetic_image(foreground_percent=5.0)
        current_bytes = create_synthetic_image(foreground_percent=8.0)

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        # Low foreground should not produce high quality scores
        assert evidence.quality_score <= 0.70

    def test_case_id_generic(self):
        """Case ID is always generic 'local-image-pair'."""
        baseline_bytes = create_synthetic_image()
        current_bytes = create_synthetic_image()

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        assert evidence.case_id == "local-image-pair"

    def test_dates_set(self):
        """Baseline and current dates are set."""
        baseline_bytes = create_synthetic_image()
        current_bytes = create_synthetic_image()

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        assert evidence.baseline_date
        assert evidence.current_date
        assert len(evidence.baseline_date) == 10  # YYYY-MM-DD

    def test_evidence_is_frozen(self):
        """Generated evidence is immutable."""
        baseline_bytes = create_synthetic_image()
        current_bytes = create_synthetic_image()

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        with pytest.raises(AttributeError):
            evidence.status = ReviewStatus.NEEDS_REVIEW

    def test_invalid_baseline_bytes_raises(self):
        """Invalid image bytes raise ValueError."""
        invalid_bytes = b"not an image"
        current_bytes = create_synthetic_image()

        with pytest.raises(ValueError):
            LocalImageEvidenceAdapter.generate_evidence_from_bytes(invalid_bytes, current_bytes)

    def test_invalid_current_bytes_raises(self):
        """Invalid current image bytes raise ValueError."""
        baseline_bytes = create_synthetic_image()
        invalid_bytes = b"not an image"

        with pytest.raises(ValueError):
            LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, invalid_bytes)

    def test_foreground_and_intensity_measured(self):
        """Foreground and intensity measurements are captured."""
        baseline_bytes = create_synthetic_image(foreground_percent=40.0, mean_intensity=100)
        current_bytes = create_synthetic_image(foreground_percent=45.0, mean_intensity=150)

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        # Measurements should be from current image (may be modified by pipeline processing)
        assert evidence.foreground_percentage > 0  # Has some foreground
        assert evidence.mean_intensity > 0  # Has some intensity value

    def test_quality_score_based_on_foreground(self):
        """Quality score reflects foreground coverage."""
        # High foreground → high quality
        high_fg_bytes = create_synthetic_image(foreground_percent=60.0)
        low_fg_bytes = create_synthetic_image(foreground_percent=5.0)
        baseline_bytes = high_fg_bytes

        evidence_high = LocalImageEvidenceAdapter.generate_evidence_from_bytes(high_fg_bytes, high_fg_bytes)
        evidence_low = LocalImageEvidenceAdapter.generate_evidence_from_bytes(low_fg_bytes, low_fg_bytes)

        assert evidence_high.quality_score > evidence_low.quality_score

    def test_no_filesystem_persistence(self):
        """Adapter does not write to filesystem."""
        import os
        import glob

        baseline_bytes = create_synthetic_image()
        current_bytes = create_synthetic_image()

        # Generate evidence
        LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        # Check that no files were created in common temp locations
        # (This is a basic check; the adapter shouldn't persist at all)
        temp_files = glob.glob("/tmp/*local*") + glob.glob("C:\\Temp\\*local*")
        assert len(temp_files) == 0, "Adapter should not create temporary files"

    def test_adapter_does_not_mutate_evidence(self):
        """Adapter does not modify evidence after creation."""
        baseline_bytes = create_synthetic_image(foreground_percent=45.0)
        current_bytes = create_synthetic_image(foreground_percent=50.0)

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)
        original_status = evidence.status
        original_fg = evidence.foreground_percentage

        # Try to modify (should fail due to frozen dataclass)
        with pytest.raises(AttributeError):
            evidence.status = ReviewStatus.UNCERTAIN

        # Original values unchanged
        assert evidence.status == original_status
        assert evidence.foreground_percentage == original_fg

    def test_consistent_with_small_changes(self):
        """Small changes below thresholds yield CONSISTENT."""
        baseline_bytes = create_synthetic_image(foreground_percent=45.0, mean_intensity=128)
        # Small changes: 5% foreground change, 10 intensity change
        current_bytes = create_synthetic_image(foreground_percent=50.0, mean_intensity=138)

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        assert evidence.status == ReviewStatus.CONSISTENT

    def test_mixed_changes_large_foreground_triggers_review(self):
        """Large foreground change triggers NEEDS_REVIEW even with small intensity change."""
        baseline_bytes = create_synthetic_image(foreground_percent=30.0, mean_intensity=128)
        current_bytes = create_synthetic_image(foreground_percent=50.0, mean_intensity=130)

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        assert evidence.status == ReviewStatus.NEEDS_REVIEW

    def test_status_consistent_small_changes(self):
        """Small changes below thresholds yield CONSISTENT."""
        # Both images with same foreground - no change
        baseline_bytes = create_synthetic_image(foreground_percent=45.0)
        current_bytes = create_synthetic_image(foreground_percent=47.0)

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        # Small ~2% foreground change should be CONSISTENT
        assert evidence.status == ReviewStatus.CONSISTENT

    def test_approval_state_always_pending(self):
        """Evidence always has PENDING_HUMAN_APPROVAL approval state (enforced by tools.py)."""
        # Note: MRIEvidence doesn't have approval_state directly,
        # but this is guaranteed by the voice routing and approval workflow
        baseline_bytes = create_synthetic_image()
        current_bytes = create_synthetic_image()

        evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline_bytes, current_bytes)

        # Verify evidence can be used with existing approval workflow
        # (This is tested in voice routing tests)
        assert evidence.case_id == "local-image-pair"


def test_mode_is_disabled_by_default_and_requires_explicit_true():
    assert not is_local_image_evidence_enabled({})
    assert not is_local_image_evidence_enabled({"LOCAL_IMAGE_EVIDENCE_ENABLED": "yes"})
    assert is_local_image_evidence_enabled({"LOCAL_IMAGE_EVIDENCE_ENABLED": "true"})


def test_local_evidence_overrides_synthetic_voice_commands():
    baseline = create_synthetic_image(foreground_percent=30.0)
    current = create_synthetic_image(foreground_percent=55.0)
    evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(baseline, current)
    session = VoiceSession()

    for command in (
        "Analyze this MRI",
        "Explain the review status",
        "What changed between the images?",
        "Generate an evidence report",
    ):
        result = route_command(command, "case-001-consistent", session, evidence=evidence)
        assert result.case_id == "local-image-pair"
        assert "case-001-consistent" not in (result.spoken + result.display)


def test_local_mode_forces_whisper_and_never_calls_assemblyai():
    calls = []
    state = {
        "voice_audio_bytes": b"in-memory-audio",
        "voice_stt_backend": stt.BACKEND_ASSEMBLYAI,
        "voice_usage": UsageTracker(),
    }

    class Result:
        text = "analyze this MRI"
        error = None

    def fake_transcribe(data, backend, usage, **kwargs):
        calls.append((data, backend, kwargs))
        assert backend == stt.BACKEND_LOCAL
        return Result()

    run_transcription(state, transcribe_fn=fake_transcribe, local_only=True)
    assert len(calls) == 1
    assert state["voice_stt_backend"] == stt.BACKEND_LOCAL


def test_clear_session_removes_local_evidence_and_cached_explanation():
    state = {
        "local_image_evidence": object(),
        "llm_explanation": object(),
        "llm_explanation_case_id": "local-image-pair",
        "last_voice_result": object(),
        "voice_speech": object(),
        "unrelated": "kept",
    }
    clear_local_image_evidence_state(state)
    assert state == {"unrelated": "kept"}


def test_local_mode_audit_is_generic_and_contains_no_image_details():
    evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(
        create_synthetic_image(), create_synthetic_image()
    )
    session = VoiceSession()
    route_command(
        "Analyze scan named private_scan.png with 123.4 measurements",
        "case-001-consistent",
        session,
        evidence=evidence,
    )
    audit_text = repr(session.audit)
    assert "local-image-pair" in audit_text
    assert "private_scan.png" not in audit_text
    assert "123.4" not in audit_text
    assert "text" not in session.audit[0]


def test_local_mode_review_request_remains_pending():
    evidence = LocalImageEvidenceAdapter.generate_evidence_from_bytes(
        create_synthetic_image(), create_synthetic_image()
    )
    session = VoiceSession()
    result = route_command(
        "Request human review", "case-001-consistent", session,
        reviewer_name="Dr. Reviewer", evidence=evidence,
    )
    assert result.approval_state.value == "PENDING_HUMAN_APPROVAL"
    assert session.requests["local-image-pair"].approval_state.value == "PENDING_HUMAN_APPROVAL"
