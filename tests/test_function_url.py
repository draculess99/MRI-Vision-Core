"""
Tests for Lambda Function URL HTTP event handling.

Tests both direct Lambda events and HTTP Function URL events.
"""

import base64
import json

import cv2
import numpy as np
import pytest

from hackathon.opencv_aws.evidence_manifest import lambda_handler


@pytest.fixture
def sample_image():
    """Create a test MRI image."""
    img = np.zeros((32, 32), dtype=np.uint8)
    img[8:24, 8:24] = 128
    return img


class TestDirectLambdaEvents:
    """Test existing direct Lambda invocation (backward compatibility)."""

    def test_direct_base64_event(self, sample_image):
        """Direct Lambda event with base64 image still works."""
        _, png_bytes = cv2.imencode(".png", sample_image)
        image_data_b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

        event = {
            "image_data": image_data_b64,
            "source_filename": "direct-test.mri",
        }

        response = lambda_handler(event)

        assert response["statusCode"] == 200
        assert "manifest" in response
        assert response["manifest"]["source_filename"] == "direct-test.mri"

    def test_direct_missing_image_error(self):
        """Direct event without image returns 400."""
        event = {"source_filename": "test.mri"}
        response = lambda_handler(event)

        assert response["statusCode"] == 400
        assert "error" in response


class TestFunctionUrlHttpEvents:
    """Test new HTTP Function URL event handling."""

    def test_http_post_valid_base64(self, sample_image):
        """HTTP POST with valid base64 image succeeds."""
        _, png_bytes = cv2.imencode(".png", sample_image)
        image_data_b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

        payload = {
            "image_data": image_data_b64,
            "source_filename": "http-test.mri",
        }

        event = {
            "requestContext": {"http": {"method": "POST"}},
            "body": json.dumps(payload),
            "headers": {"content-type": "application/json"},
        }

        response = lambda_handler(event)

        assert response["statusCode"] == 200
        assert "body" in response
        assert "headers" in response
        assert response["headers"]["Content-Type"] == "application/json"

        body = json.loads(response["body"])
        assert body["statusCode"] == 200
        assert "manifest" in body
        assert body["manifest"]["source_filename"] == "http-test.mri"

    def test_http_get_rejected(self):
        """HTTP GET is rejected with 405."""
        event = {
            "requestContext": {"http": {"method": "GET"}},
            "body": "",
        }

        response = lambda_handler(event)

        assert response["statusCode"] == 405
        assert "body" in response
        assert "headers" in response
        body = json.loads(response["body"])
        assert "Method not allowed" in body["error"]

    def test_http_put_rejected(self):
        """HTTP PUT is rejected with 405."""
        event = {
            "requestContext": {"http": {"method": "PUT"}},
            "body": "{}",
        }

        response = lambda_handler(event)

        assert response["statusCode"] == 405

    def test_http_empty_body(self):
        """HTTP POST with empty body returns 400."""
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "body": "",
        }

        response = lambda_handler(event)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "empty" in body["error"].lower()

    def test_http_malformed_json(self):
        """HTTP POST with invalid JSON returns 400."""
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "body": "not valid json {{{",
        }

        response = lambda_handler(event)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "Invalid JSON" in body["error"]

    def test_http_missing_image_data(self):
        """HTTP POST without image_data or image_path returns 400."""
        payload = {"source_filename": "test.mri"}

        event = {
            "requestContext": {"http": {"method": "POST"}},
            "body": json.dumps(payload),
        }

        response = lambda_handler(event)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "Missing required field" in body["error"]

    def test_http_invalid_base64(self):
        """HTTP POST with invalid base64 returns 400."""
        payload = {
            "image_data": "not-valid-base64!!!",
            "source_filename": "bad.mri",
        }

        event = {
            "requestContext": {"http": {"method": "POST"}},
            "body": json.dumps(payload),
        }

        response = lambda_handler(event)

        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "error" in body

    def test_http_response_headers(self, sample_image):
        """HTTP response includes Content-Type header."""
        _, png_bytes = cv2.imencode(".png", sample_image)
        image_data_b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

        payload = {
            "image_data": image_data_b64,
            "source_filename": "header-test.mri",
        }

        event = {
            "requestContext": {"http": {"method": "POST"}},
            "body": json.dumps(payload),
        }

        response = lambda_handler(event)

        assert "headers" in response
        assert response["headers"]["Content-Type"] == "application/json"

    def test_http_no_stack_traces(self):
        """HTTP error responses don't expose stack traces."""
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "body": "invalid{json",
        }

        response = lambda_handler(event)

        body = json.loads(response["body"])
        # Should not contain traceback or file paths
        assert "Traceback" not in str(body)
        assert "/tmp/" not in str(body)
