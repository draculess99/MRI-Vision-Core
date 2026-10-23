# OpenCV MRI Evidence Manifest — Lambda Deployment Readiness

## Current Status

**This project is NOT deployed.** No AWS resources have been created or deployed. This document describes how the project *could* be deployed in the future, should AWS resources and deployment be authorized.

## What This Is

- A deterministic OpenCV MRI processing pipeline with evidence manifest generation
- Lambda-container-ready code with Dockerfile and SAM template
- Non-diagnostic, research-only demonstration
- Local-only processing with no external API calls or credential requirements

## What This Is NOT

- Currently deployed to AWS Lambda
- A production system
- Diagnostic or medical software
- Requiring AWS credentials or accounts to test locally

## Future Build & Test Commands (Local Only)

### Test without Docker or AWS

Run all tests locally without any AWS infrastructure:

```bash
python -m pytest -q tests/test_opencv_mri.py tests/test_opencv_manifest.py tests/test_opencv_manifest_lambda.py
```

### Build Docker Image (Requires Docker)

If Docker is installed locally, build the container image:

```bash
docker build -t opencv-mri-evidence-manifest:latest -f hackathon/opencv_aws/Dockerfile .
```

### Deploy to AWS (Future Only)

**This command is NOT run. It is provided only for reference if AWS deployment is authorized in the future:**

```bash
# DO NOT RUN unless explicitly authorized to deploy
# sam build --template-file hackathon/opencv_aws/template.yaml
# sam deploy --guided
```

## Prerequisites (If Deployment Becomes Necessary)

- AWS CLI configured with credentials (future only)
- AWS SAM CLI installed (future only)
- Docker installed (optional, for building images locally)
- Python 3.12+ with pytest (required for local testing)

## Key Design Decisions

1. **No AWS SDK in code** — The handler uses only standard library + OpenCV/NumPy. No boto3, S3, or other AWS services.

2. **Deterministic processing** — All OpenCV operations are deterministic and reproducible. SHA-256 hashes track input integrity.

3. **Base64 image support** — The handler accepts either file paths (for local testing) or base64-encoded images (for container/remote scenarios).

4. **Size limits** — Decoded images are capped at 4 MB to prevent resource exhaustion.

5. **Version matching** — Docker image uses opencv-python-headless 5.0.0.93 and numpy 2.5.2 to match the local validated OpenCV major version (5.0.0), while avoiding X11/GUI dependencies that are not available in the minimal AWS Lambda base image. The headless wheel provides the same deterministic image processing without graphics dependencies.

5. **Non-diagnostic** — Every manifest includes a safety notice. No medical claims or clinical interpretation.

## Event Format Examples

### File Path Event (Local)

```json
{
  "image_path": "/path/to/mri_slice.png",
  "source_filename": "mri_slice.mri"
}
```

### Base64 Image Event (Container)

```json
{
  "image_data": "iVBORw0KGgoAAAANSUhEUgAAAA...(base64)...CYII=",
  "source_filename": "uploaded.mri"
}
```

## Output Format

```json
{
  "statusCode": 200,
  "manifest": {
    "mode": "local_deterministic",
    "safety_notice": "Research and educational prototype. Non-diagnostic. Not for medical diagnosis or clinical decision-making.",
    "source_filename": "mri_slice.mri",
    "source_hash_sha256": "abc123def456...",
    "input_width": 512,
    "input_height": 512,
    "input_dtype": "uint8",
    "input_shape": [512, 512],
    "opencv_version": "4.8.1",
    "processing_steps": [...],
    "generated_artifacts": ["original", "enhanced", "edges", "overlay"]
  }
}
```

## Testing Checklist (Local Only)

- [x] All 37 existing tests pass (OpenCV + manifest)
- [x] Lambda handler accepts file paths
- [x] Lambda handler accepts base64-encoded images
- [x] Handler rejects oversized images (> 50 MB)
- [x] Handler rejects malformed base64
- [x] Handler rejects invalid images
- [x] Deterministic pipeline runs without errors
- [x] Manifests serialize to valid JSON

## Research & Educational Use

This code demonstrates:
- Deterministic image processing pipelines
- Containerization for cloud deployment
- Evidence capture and integrity verification (SHA-256)
- Graceful error handling for invalid inputs
- Non-diagnostic safety practices

## Next Steps (If Authorization Comes)

1. Ensure AWS SAM CLI is installed
2. Configure AWS CLI with appropriate credentials
3. Run `sam build` to build the container image
4. Run `sam deploy --guided` to deploy to AWS Lambda
5. Test with sample events once deployed

**Until explicit authorization, this remains local research code only.**
