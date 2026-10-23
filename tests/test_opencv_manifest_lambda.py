"""Tests for Lambda handler base64-encoded image events."""

import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

from hackathon.opencv_aws.evidence_manifest import (
    MAX_IMAGE_BYTES,
    lambda_handler,
)


@pytest.fixture
def sample_mri_image():
    """Create a deterministic sample MRI image."""
    img = np.zeros((64, 64), dtype=np.float32)
    img[16:48, 16:48] = 128.0
    img[24:40, 24:40] = 200.0
    return img


def test_lambda_handler_with_base64_image(sample_mri_image):
    """Verify Lambda handler works with base64-encoded image data."""
    # Encode image as PNG bytes then base64
    success, png_bytes = cv2.imencode(".png", sample_mri_image.astype(np.uint8))
    assert success

    image_data_b64 = base64.b64encode(png_bytes.tobytes()).decode('utf-8')

    # Call handler with base64 event
    event = {
        "image_data": image_data_b64,
        "source_filename": "base64-test.mri",
    }
    response = lambda_handler(event)

    assert response["statusCode"] == 200
    assert "manifest" in response
    manifest_data = response["manifest"]
    assert manifest_data["source_filename"] == "base64-test.mri"
    assert manifest_data["input_width"] == 64
    assert manifest_data["input_height"] == 64


def test_lambda_handler_malformed_base64():
    """Verify handler returns error for malformed base64 data."""
    event = {
        "image_data": "not-valid-base64!!!@@@",
        "source_filename": "bad.mri",
    }
    response = lambda_handler(event)

    assert response["statusCode"] == 400
    assert "error" in response
    assert "Invalid base64" in response["error"]


def test_lambda_handler_oversized_image():
    """Verify handler rejects images exceeding 4 MB size limit."""
    # Create oversized image data (exceeding MAX_IMAGE_BYTES = 4 MB)
    oversized_data = b'\x00' * (MAX_IMAGE_BYTES + 1000)
    image_data_b64 = base64.b64encode(oversized_data).decode('utf-8')

    event = {
        "image_data": image_data_b64,
        "source_filename": "huge.mri",
    }
    response = lambda_handler(event)

    assert response["statusCode"] == 400
    assert "error" in response
    assert "exceeds" in response["error"].lower()


def test_lambda_handler_invalid_image_format():
    """Verify handler rejects invalid image data."""
    # Valid base64 but not decodable as image
    invalid_image_b64 = base64.b64encode(b"not an image format").decode('utf-8')

    event = {
        "image_data": invalid_image_b64,
        "source_filename": "invalid.mri",
    }
    response = lambda_handler(event)

    assert response["statusCode"] == 400
    assert "error" in response
    assert "Failed to decode" in response["error"]


def test_lambda_handler_missing_both_inputs():
    """Verify handler returns error when both image_path and image_data are missing."""
    event = {
        "source_filename": "missing.mri",
    }
    response = lambda_handler(event)

    assert response["statusCode"] == 400
    assert "error" in response
    assert "image_path" in response["error"] or "image_data" in response["error"]


def test_lambda_handler_prefers_file_path_over_base64(sample_mri_image):
    """Verify handler prefers file path if both are provided."""
    with TemporaryDirectory() as tmpdir:
        # Save test image
        image_path = Path(tmpdir) / "test.png"
        cv2.imwrite(str(image_path), sample_mri_image)

        # Create base64 with different dimensions (should be ignored)
        different_img = np.ones((32, 32), dtype=np.float32) * 50.0
        success, png_bytes = cv2.imencode(".png", different_img.astype(np.uint8))
        image_data_b64 = base64.b64encode(png_bytes.tobytes()).decode('utf-8')

        # Call handler with both path and data
        event = {
            "image_path": str(image_path),
            "image_data": image_data_b64,
            "source_filename": "test_image.mri",
        }
        response = lambda_handler(event)

        assert response["statusCode"] == 200
        manifest_data = response["manifest"]
        # Should use file path version (64x64)
        assert manifest_data["input_width"] == 64
        assert manifest_data["input_height"] == 64


def test_lambda_handler_with_sample_event():
    """Verify handler works with the local sample-event.json file."""
    # Load the sample event from the deployed location
    sample_event_path = Path(__file__).parent.parent / "hackathon" / "opencv_aws" / "sample-event.json"

    with open(sample_event_path) as f:
        event = json.load(f)

    # Invoke the handler
    response = lambda_handler(event)

    # Verify success response
    assert response["statusCode"] == 200, f"Expected statusCode 200, got {response.get('statusCode')}"
    assert "manifest" in response, "Response missing 'manifest' field"

    manifest = response["manifest"]

    # Verify required manifest fields
    assert manifest["mode"] == "local_deterministic"
    assert manifest["source_hash_sha256"], "Missing source_hash_sha256"
    assert len(manifest["source_hash_sha256"]) == 64, "Invalid SHA-256 hash length"
    assert manifest["opencv_version"].startswith("5."), f"Expected OpenCV 5.x, got {manifest['opencv_version']}"
    assert "Non-diagnostic" in manifest["safety_notice"]
    assert len(manifest["processing_steps"]) == 4
    assert manifest["source_filename"] == "sample-mri-slice.png"
