"""
Tests for OpenCV adapter for execution router.

All tests use local processing; no AWS invocations.
"""

import base64
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

from hackathon.opencv_aws.opencv_adapter import OpenCVProcessor


@pytest.fixture
def sample_image():
    """Create a deterministic sample MRI image."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 128
    img[24:40, 24:40] = 200
    return img


class TestOpenCVProcessor:
    """Tests for OpenCV processor."""

    def test_process_with_file_path(self, sample_image):
        """Processor accepts file path input."""
        with TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            cv2.imwrite(str(img_path), sample_image)

            payload = {
                "image_path": str(img_path),
                "source_filename": "test.mri",
            }
            result = OpenCVProcessor.process(payload)

            assert result["statusCode"] == 200
            assert result["mode"] == "local_deterministic"
            assert "manifest" in result
            assert "artifacts" in result

    def test_process_with_base64_image(self, sample_image):
        """Processor accepts base64-encoded image."""
        _, png_bytes = cv2.imencode(".png", sample_image)
        image_data_b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

        payload = {
            "image_data": image_data_b64,
            "source_filename": "test.mri",
        }
        result = OpenCVProcessor.process(payload)

        assert result["statusCode"] == 200
        assert result["mode"] == "local_deterministic"
        assert "manifest" in result
        assert "artifacts" in result

    def test_process_returns_artifacts(self, sample_image):
        """Processor returns all four artifact arrays."""
        _, png_bytes = cv2.imencode(".png", sample_image)
        image_data_b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

        payload = {"image_data": image_data_b64}
        result = OpenCVProcessor.process(payload)

        artifacts = result["artifacts"]
        assert "original" in artifacts
        assert "enhanced" in artifacts
        assert "edges" in artifacts
        assert "overlay" in artifacts

        # All should be numpy arrays
        assert isinstance(artifacts["original"], np.ndarray)
        assert isinstance(artifacts["enhanced"], np.ndarray)
        assert isinstance(artifacts["edges"], np.ndarray)
        assert isinstance(artifacts["overlay"], np.ndarray)

    def test_process_manifest_serializable(self, sample_image):
        """Processor returns JSON-serializable manifest."""
        _, png_bytes = cv2.imencode(".png", sample_image)
        image_data_b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

        payload = {"image_data": image_data_b64, "source_filename": "test.mri"}
        result = OpenCVProcessor.process(payload)

        manifest = result["manifest"]
        assert isinstance(manifest, dict)
        assert manifest["mode"] == "local_deterministic"
        assert "source_hash_sha256" in manifest
        assert len(manifest["source_hash_sha256"]) == 64  # SHA-256 hex length

    def test_missing_image_path_error(self):
        """Missing both image_path and image_data returns error."""
        payload = {"source_filename": "test.mri"}
        result = OpenCVProcessor.process(payload)

        assert result["statusCode"] == 400
        assert "error" in result
        assert "Missing required field" in result["error"]

    def test_invalid_file_path_error(self):
        """Invalid file path returns error."""
        payload = {
            "image_path": "/nonexistent/path/image.png",
            "source_filename": "test.mri",
        }
        result = OpenCVProcessor.process(payload)

        assert result["statusCode"] == 400
        assert "error" in result

    def test_malformed_base64_error(self):
        """Malformed base64 returns error."""
        payload = {
            "image_data": "not-valid-base64!!!@@@",
            "source_filename": "test.mri",
        }
        result = OpenCVProcessor.process(payload)

        assert result["statusCode"] == 400
        assert "error" in result

    def test_invalid_image_data_error(self):
        """Valid base64 but invalid image data returns error."""
        invalid_b64 = base64.b64encode(b"not an image").decode("utf-8")
        payload = {
            "image_data": invalid_b64,
            "source_filename": "test.mri",
        }
        result = OpenCVProcessor.process(payload)

        assert result["statusCode"] == 400
        assert "error" in result
        assert "Failed to decode" in result["error"]

    def test_oversized_image_error(self):
        """Image exceeding size limit returns error."""
        from hackathon.opencv_aws.evidence_manifest import MAX_IMAGE_BYTES

        oversized = b"\x00" * (MAX_IMAGE_BYTES + 1000)
        image_data_b64 = base64.b64encode(oversized).decode("utf-8")

        payload = {
            "image_data": image_data_b64,
            "source_filename": "huge.mri",
        }
        result = OpenCVProcessor.process(payload)

        assert result["statusCode"] == 400
        assert "exceeds" in result["error"].lower()

    def test_file_path_preferred_over_base64(self, sample_image):
        """File path is preferred if both are provided."""
        with TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "file.png"
            cv2.imwrite(str(img_path), sample_image)

            # Create different base64 image
            different_img = np.ones((32, 32), dtype=np.uint8) * 50
            _, png_bytes = cv2.imencode(".png", different_img)
            different_b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

            payload = {
                "image_path": str(img_path),
                "image_data": different_b64,
                "source_filename": "test.mri",
            }
            result = OpenCVProcessor.process(payload)

            assert result["statusCode"] == 200
            # Should use file path (64x64), not base64 (32x32)
            manifest = result["manifest"]
            assert manifest["input_width"] == 64
            assert manifest["input_height"] == 64

    def test_source_filename_preserved(self, sample_image):
        """Source filename is preserved in manifest."""
        _, png_bytes = cv2.imencode(".png", sample_image)
        image_data_b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

        payload = {
            "image_data": image_data_b64,
            "source_filename": "patient-001.mri",
        }
        result = OpenCVProcessor.process(payload)

        assert result["manifest"]["source_filename"] == "patient-001.mri"

    def test_deterministic_reproducibility(self, sample_image):
        """Same image produces identical hashes on repeated calls."""
        _, png_bytes = cv2.imencode(".png", sample_image)
        image_data_b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

        payload = {
            "image_data": image_data_b64,
            "source_filename": "test.mri",
        }

        result1 = OpenCVProcessor.process(payload)
        result2 = OpenCVProcessor.process(payload)

        hash1 = result1["manifest"]["source_hash_sha256"]
        hash2 = result2["manifest"]["source_hash_sha256"]
        assert hash1 == hash2
