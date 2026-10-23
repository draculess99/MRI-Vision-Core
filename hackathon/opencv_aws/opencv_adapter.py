"""
OpenCV adapter for execution router.

Wraps existing process_mri_slice pipeline and evidence manifest
generation. Converts payloads to/from execution router format.
"""

import base64
from typing import Any, Dict, Optional

import cv2
import numpy as np

from hackathon.opencv_aws.evidence_manifest import (
    MAX_IMAGE_BYTES,
    create_manifest,
)
from hackathon.opencv_aws.opencv_mri import process_mri_slice


class OpenCVProcessor:
    """Process MRI images using existing deterministic OpenCV pipeline."""

    @staticmethod
    def process(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image from payload.

        Payload format:
        {
            "image_path": "/path/to/image.png",  # Optional
            "image_data": "base64-encoded-bytes",  # Optional
            "source_filename": "name.mri",  # Optional
        }

        Returns:
        {
            "statusCode": 200 or 4xx,
            "mode": "local_deterministic",
            "manifest": {...},
            "artifacts": {
                "original": numpy array,
                "enhanced": numpy array,
                "edges": numpy array,
                "overlay": numpy array,
            },
            "error": "..." (if statusCode != 200),
        }
        """
        image = None
        source_filename = "unknown.mri"

        # Try file path first
        image_path = payload.get("image_path")
        if image_path:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return {
                    "statusCode": 400,
                    "mode": "local_deterministic",
                    "error": f"Failed to load image from {image_path}",
                }
            source_filename = payload.get("source_filename", image_path.split("/")[-1])

        # Try base64-encoded image
        image_data = payload.get("image_data")
        if image_data and image is None:
            try:
                decoded = base64.b64decode(image_data)

                if len(decoded) > MAX_IMAGE_BYTES:
                    return {
                        "statusCode": 400,
                        "mode": "local_deterministic",
                        "error": f"Image exceeds {MAX_IMAGE_BYTES} bytes ({len(decoded)} bytes)",
                    }

                nparr = np.frombuffer(decoded, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    return {
                        "statusCode": 400,
                        "mode": "local_deterministic",
                        "error": "Failed to decode image from base64 data",
                    }

                source_filename = payload.get("source_filename", "uploaded.mri")

            except Exception as e:
                return {
                    "statusCode": 400,
                    "mode": "local_deterministic",
                    "error": f"Invalid base64 image data: {str(e)}",
                }

        if image is None:
            return {
                "statusCode": 400,
                "mode": "local_deterministic",
                "error": "Missing required field: provide either 'image_path' or 'image_data'",
            }

        try:
            result = process_mri_slice(image)
            manifest = create_manifest(
                image,
                source_filename=source_filename,
                artifact_names=["original", "enhanced", "edges", "overlay"],
            )

            return {
                "statusCode": 200,
                "mode": "local_deterministic",
                "manifest": manifest.to_dict(),
                "artifacts": {
                    "original": result.original,
                    "enhanced": result.enhanced,
                    "edges": result.edges,
                    "overlay": result.overlay,
                },
            }
        except Exception as e:
            return {
                "statusCode": 500,
                "mode": "local_deterministic",
                "error": str(e),
            }
