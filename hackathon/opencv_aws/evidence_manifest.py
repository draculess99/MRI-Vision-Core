"""
Deterministic evidence manifest layer for OpenCV MRI processing.

Non-diagnostic, research-only. This module captures processing metadata
and generated artifacts as JSON-serializable evidence for reproducibility.
"""

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from hackathon.opencv_aws.opencv_mri import OpenCVMRIResult, process_mri_slice

# Maximum decoded image size: 4 MB
MAX_IMAGE_BYTES = 4 * 1024 * 1024


@dataclass
class EvidenceManifest:
    """Non-diagnostic evidence manifest for processed MRI images."""

    mode: str = "local_deterministic"
    safety_notice: str = (
        "Research and educational prototype. Non-diagnostic. "
        "Not for medical diagnosis or clinical decision-making."
    )

    source_filename: str = ""
    source_hash_sha256: str = ""

    input_width: int = 0
    input_height: int = 0
    input_dtype: str = ""
    input_shape: tuple = ()

    opencv_version: str = ""

    processing_steps: list[dict[str, Any]] = None
    generated_artifacts: list[str] = None

    def __post_init__(self):
        """Initialize list defaults after dataclass construction."""
        if self.processing_steps is None:
            self.processing_steps = []
        if self.generated_artifacts is None:
            self.generated_artifacts = []

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to JSON-serializable dictionary."""
        data = asdict(self)
        data["input_shape"] = list(self.input_shape)
        return data

    def to_json_str(self) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def compute_image_hash(image: np.ndarray) -> str:
    """
    Compute SHA-256 hash of image bytes.
    Ensures deterministic and verifiable image integrity.
    """
    image_bytes = image.astype(np.uint8).tobytes()
    return hashlib.sha256(image_bytes).hexdigest()


def create_manifest(
    image: np.ndarray,
    source_filename: str = "mri_image",
    artifact_names: Optional[list[str]] = None,
) -> EvidenceManifest:
    """
    Create a deterministic evidence manifest for an MRI image.

    Records image metadata, OpenCV version, and processing steps
    without any clinical interpretation.
    """
    if artifact_names is None:
        artifact_names = []

    # Compute hash of input image
    image_hash = compute_image_hash(image)

    # Record input metadata
    height, width = image.shape[:2]
    dtype = str(image.dtype)

    # Record processing steps with parameters
    processing_steps = [
        {
            "step": "normalize_to_uint8",
            "method": "min-max normalization to [0, 255]",
            "deterministic": True,
        },
        {
            "step": "enhance_contrast",
            "method": "CLAHE (Contrast Limited Adaptive Histogram Equalization)",
            "parameters": {
                "clipLimit": 2.0,
                "tileGridSize": [8, 8],
            },
            "deterministic": True,
        },
        {
            "step": "detect_edges",
            "method": "Gaussian blur + Canny edge detection",
            "parameters": {
                "gaussian_kernel": [5, 5],
                "gaussian_sigma": 0,
                "canny_threshold1": 40,
                "canny_threshold2": 120,
            },
            "deterministic": True,
        },
        {
            "step": "create_overlay",
            "method": "Overlay edges on enhanced image",
            "parameters": {
                "edge_color": [0, 255, 0],  # BGR green
            },
            "deterministic": True,
        },
    ]

    manifest = EvidenceManifest(
        source_filename=source_filename,
        source_hash_sha256=image_hash,
        input_width=width,
        input_height=height,
        input_dtype=dtype,
        input_shape=image.shape,
        opencv_version=cv2.__version__,
        processing_steps=processing_steps,
        generated_artifacts=artifact_names,
    )

    return manifest


def _process_manifest(event: dict) -> dict:
    """
    Core manifest processing (shared by direct and HTTP invocations).

    Args:
        event: Dict with image_path or image_data

    Returns:
        {statusCode: int, manifest: dict} or {statusCode: int, error: str}
    """
    try:
        image = None
        source_filename = "unknown.mri"

        # Try file path first
        image_path = event.get("image_path")
        if image_path:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return {
                    "statusCode": 400,
                    "error": f"Failed to load image from {image_path}",
                }
            source_filename = event.get("source_filename", Path(image_path).name)

        # Try base64-encoded image
        image_data = event.get("image_data")
        if image_data and image is None:
            try:
                # Decode base64
                decoded = base64.b64decode(image_data)

                # Check size limit
                if len(decoded) > MAX_IMAGE_BYTES:
                    return {
                        "statusCode": 400,
                        "error": f"Image exceeds {MAX_IMAGE_BYTES} bytes ({len(decoded)} bytes)",
                    }

                # Decode to numpy array
                nparr = np.frombuffer(decoded, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    return {
                        "statusCode": 400,
                        "error": "Failed to decode image from base64 data",
                    }

                source_filename = event.get("source_filename", "uploaded.mri")

            except Exception as e:
                return {
                    "statusCode": 400,
                    "error": f"Invalid base64 image data: {str(e)}",
                }

        # Check if we got an image
        if image is None:
            return {
                "statusCode": 400,
                "error": "Missing required field: provide either 'image_path' or 'image_data'",
            }

        # Run deterministic processing
        result = process_mri_slice(image)

        # Generate artifact names
        artifact_names = [
            "original",
            "enhanced",
            "edges",
            "overlay",
        ]

        # Create manifest
        manifest = create_manifest(
            image,
            source_filename=source_filename,
            artifact_names=artifact_names,
        )

        return {
            "statusCode": 200,
            "manifest": manifest.to_dict(),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "error": str(e),
        }


def lambda_handler(event: dict, context: Any = None) -> dict:
    """
    AWS Lambda handler supporting both direct events and HTTP Function URL events.

    Direct invocation (programmatic):
    {
        "image_path": "...",
        "source_filename": "..."
    }

    HTTP Function URL invocation (POST only):
    {
        "requestContext": {"http": {"method": "POST"}},
        "body": "{...json...}",
        "headers": {"content-type": "application/json"}
    }

    Returns:
    Direct: {statusCode: int, manifest: dict or error: str}
    HTTP: {statusCode: int, body: json, headers: {"Content-Type": "application/json"}}
    """
    # Detect HTTP Function URL event (has requestContext.http)
    request_context = event.get("requestContext", {})
    http_info = request_context.get("http")

    if http_info:
        # HTTP Function URL invocation
        method = http_info.get("method", "")

        # Only allow POST
        if method != "POST":
            return {
                "statusCode": 405,
                "body": json.dumps({"error": "Method not allowed. Use POST."}),
                "headers": {"Content-Type": "application/json"},
            }

        # Parse request body
        body = event.get("body", "")
        if not body:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Request body is empty."}),
                "headers": {"Content-Type": "application/json"},
            }

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid JSON in request body."}),
                "headers": {"Content-Type": "application/json"},
            }

        # Process using core handler
        result = _process_manifest(payload)

        # Return HTTP response
        return {
            "statusCode": result.get("statusCode", 200),
            "body": json.dumps(result),
            "headers": {"Content-Type": "application/json"},
        }

    else:
        # Direct Lambda invocation (backward compatible)
        return _process_manifest(event)
