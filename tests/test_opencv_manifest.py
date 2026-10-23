"""Tests for OpenCV MRI evidence manifest."""

import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

from hackathon.opencv_aws.evidence_manifest import (
    EvidenceManifest,
    MAX_IMAGE_BYTES,
    compute_image_hash,
    create_manifest,
    lambda_handler,
)


@pytest.fixture
def sample_mri_image():
    """Create a deterministic sample MRI image."""
    # 64x64 grayscale with known pattern for hash reproducibility
    img = np.zeros((64, 64), dtype=np.float32)
    img[16:48, 16:48] = 128.0  # Central square
    img[24:40, 24:40] = 200.0  # Inner square
    return img


def test_evidence_manifest_creation(sample_mri_image):
    """Verify manifest captures all required fields."""
    manifest = create_manifest(sample_mri_image, source_filename="test.mri")

    assert manifest.source_filename == "test.mri"
    assert manifest.input_width == 64
    assert manifest.input_height == 64
    assert manifest.input_dtype == "float32"
    assert manifest.input_shape == (64, 64)
    assert manifest.mode == "local_deterministic"
    assert "Non-diagnostic" in manifest.safety_notice
    assert len(manifest.processing_steps) == 4


def test_manifest_deterministic_hash(sample_mri_image):
    """Verify hash is deterministic for same input."""
    hash1 = compute_image_hash(sample_mri_image)
    hash2 = compute_image_hash(sample_mri_image)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 is 256 bits = 64 hex chars


def test_manifest_different_hash_for_different_image(sample_mri_image):
    """Verify different images produce different hashes."""
    hash1 = compute_image_hash(sample_mri_image)

    # Create slightly different image
    different_img = sample_mri_image.copy()
    different_img[0, 0] = 255.0
    hash2 = compute_image_hash(different_img)

    assert hash1 != hash2


def test_manifest_json_serialization(sample_mri_image):
    """Verify manifest can be serialized to JSON."""
    manifest = create_manifest(sample_mri_image)

    # Should serialize without error
    json_str = manifest.to_json_str()
    assert isinstance(json_str, str)

    # Should deserialize back to dict
    data = json.loads(json_str)
    assert data["mode"] == "local_deterministic"
    assert data["input_width"] == 64
    assert data["input_height"] == 64
    assert len(data["processing_steps"]) == 4


def test_manifest_processing_steps_completeness(sample_mri_image):
    """Verify all processing steps are recorded."""
    manifest = create_manifest(sample_mri_image)

    step_names = {s["step"] for s in manifest.processing_steps}
    expected = {
        "normalize_to_uint8",
        "enhance_contrast",
        "detect_edges",
        "create_overlay",
    }
    assert step_names == expected

    # All steps must be marked as deterministic
    for step in manifest.processing_steps:
        assert step["deterministic"] is True


def test_manifest_artifact_names(sample_mri_image):
    """Verify generated artifact names are recorded."""
    artifact_names = ["test_original", "test_enhanced"]
    manifest = create_manifest(
        sample_mri_image,
        artifact_names=artifact_names,
    )

    assert manifest.generated_artifacts == artifact_names


def test_lambda_handler_with_file(sample_mri_image):
    """Verify Lambda handler works with a file."""
    with TemporaryDirectory() as tmpdir:
        # Save test image
        image_path = Path(tmpdir) / "test.png"
        cv2.imwrite(str(image_path), sample_mri_image)

        # Call handler
        event = {
            "image_path": str(image_path),
            "source_filename": "test_image.mri",
        }
        response = lambda_handler(event)

        assert response["statusCode"] == 200
        assert "manifest" in response
        manifest_data = response["manifest"]
        assert manifest_data["source_filename"] == "test_image.mri"
        assert manifest_data["input_width"] == 64
        assert manifest_data["input_height"] == 64


def test_lambda_handler_missing_image_path():
    """Verify handler returns error for missing image_path."""
    event = {}
    response = lambda_handler(event)

    assert response["statusCode"] == 400
    assert "error" in response
    assert "image_path" in response["error"]


def test_lambda_handler_invalid_image_path():
    """Verify handler returns error for invalid image path."""
    event = {
        "image_path": "/nonexistent/path/image.png",
    }
    response = lambda_handler(event)

    assert response["statusCode"] == 400
    assert "error" in response
    assert "Failed to load image" in response["error"]


def test_manifest_to_dict(sample_mri_image):
    """Verify to_dict() conversion is correct."""
    manifest = create_manifest(sample_mri_image)
    data = manifest.to_dict()

    # All required fields should be present
    assert "mode" in data
    assert "safety_notice" in data
    assert "source_filename" in data
    assert "source_hash_sha256" in data
    assert "input_width" in data
    assert "input_height" in data
    assert "input_dtype" in data
    assert "input_shape" in data
    assert "opencv_version" in data
    assert "processing_steps" in data
    assert "generated_artifacts" in data

    # input_shape should be converted to list
    assert isinstance(data["input_shape"], list)
    assert data["input_shape"] == [64, 64]


def test_manifest_opencv_version_recorded(sample_mri_image):
    """Verify OpenCV version is recorded."""
    manifest = create_manifest(sample_mri_image)

    assert manifest.opencv_version
    assert len(manifest.opencv_version) > 0
    # Should look like a version string (e.g., "4.5.0")
    assert manifest.opencv_version[0].isdigit()


def test_manifest_hash_in_output(sample_mri_image):
    """Verify SHA-256 hash is included in output."""
    manifest = create_manifest(sample_mri_image)

    assert manifest.source_hash_sha256
    assert len(manifest.source_hash_sha256) == 64

    # Verify it matches direct computation
    expected_hash = compute_image_hash(sample_mri_image)
    assert manifest.source_hash_sha256 == expected_hash
