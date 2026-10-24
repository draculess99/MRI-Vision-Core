# MRI Vision Core — Deterministic OpenCV Evidence Workflow

A deterministic, research-and-education OpenCV image-processing workflow for exploring MRI slice structure. This is a **non-diagnostic** demonstration and does not provide clinical interpretation.

## Live Demonstrations

**Railway Streamlit Interactive Demo:**
https://mri-vision-core-production-6dfd.up.railway.app

**AWS Lambda API Endpoint (Evidence Manifest):**
https://gpint2fywbzbcy5crjldc2j6oi0pmxyb.lambda-url.us-east-1.on.aws/

## Architecture

### Interactive Visual Demo — Railway Streamlit

```mermaid
flowchart TD
    A["PNG/JPG user input"] --> B["Railway Streamlit demo"]
    B --> C["Execution Router"]
    C --> D["Local deterministic<br/>OpenCV pipeline"]
    D --> E["OpenCV artifacts<br/>Enhanced • Edges • Overlay"]
    D --> F["Evidence manifest<br/>SHA-256 + processing metadata"]
    F --> G["ChangeGuard<br/>baseline comparison"]
    G --> H["Evidence Review Agent<br/>deterministic rules"]
    H --> I["One auditable outcome<br/>FINALIZE_EVIDENCE_BUNDLE<br/>REQUEST_MORE_EVIDENCE<br/>PENDING_HUMAN_APPROVAL"]
```

The interactive Streamlit demo executes locally and does not invoke AWS Lambda, an external API, or an LLM.

### Evidence-Manifest API — AWS Lambda

```mermaid
flowchart TD
    A["Repository sample<br/>JSON event"] --> B["AWS Lambda<br/>Function URL"]
    B --> C["Lambda container<br/>OpenCV 5"]
    C --> D["JSON evidence-manifest<br/>response"]
```

## Included Synthetic Walkthrough Input

The repository includes `synthetic_knee_mri_phantom_1.png` as the reproducible input for the OpenCV walkthrough. It is a synthetic demonstration image, not patient data, and is not intended for clinical use.

[Open the included synthetic walkthrough input](../../sample_data/opencv_aws/synthetic_knee_mri_phantom_1.png)

## How It Works — Live Walkthrough

1. **Load an image**: Download and upload the included synthetic demonstration image, `synthetic_knee_mri_phantom_1.png`, or upload another PNG/JPG supplied by the user.
2. **Run OpenCV processing**: Click "Run OpenCV Processing"
   - Normalizes grayscale to [0–255]
   - Applies CLAHE contrast enhancement (clipLimit=2.0, 8×8 tiles)
   - Detects edges with Gaussian blur + Canny (thresholds 40, 120)
   - Overlays green edges on the enhanced base
3. **Review outputs**: Four visual artifacts in a 2×2 grid
4. **Compare with baseline** (ChangeGuard):
   - Run again to store and compare against the enhanced output
   - Foreground delta: % of pixels changing ≥10 intensity counts
   - Mean intensity delta: absolute difference in pixel means
   - **STABLE** if both below thresholds (≥15%, ≥20 counts)
   - **REVIEW** if thresholds exceeded
5. **Agentic Evidence Review** (automatic):
   - Deterministic agent evaluates manifest and baseline comparison
   - Three possible outcomes:
     - **FINALIZE_EVIDENCE_BUNDLE**: All artifacts present, processing deterministic, baseline stable or first-run
     - **REQUEST_MORE_EVIDENCE**: ChangeGuard triggered REVIEW, or artifact evidence incomplete
     - **PENDING_HUMAN_APPROVAL**: Metadata malformed, dimensions invalid, or unexpected state
   - View expandable JSON audit trace for full decision details
6. **Inspect evidence manifest**: View SHA-256, processing steps, and artifact names

## Agentic Evidence Review

The Agentic Evidence Review is a **deterministic, rule-based evidence-validation system** that evaluates the OpenCV manifest and ChangeGuard baseline comparison to produce an outcome. It is **not an LLM** and **not a diagnostic system**. It consumes the evidence already produced by the pipeline and applies fixed decision rules.

### How It Works

The agent executes a three-step audit trace:

1. **validate_manifest**: Checks that the evidence manifest is complete and well-formed (required fields present, processing steps marked deterministic, all four artifacts generated, image dimensions valid).
2. **assess_change_guard**: Evaluates the ChangeGuard baseline comparison result (STABLE, REVIEW, or INVALID). First-run conditions with no prior baseline are expected; other INVALID results warrant investigation.
3. **choose_next_action**: Applies decision precedence rules to determine the outcome.

### Outcomes

The agent produces exactly one of three outcomes:

- **FINALIZE_EVIDENCE_BUNDLE**: All artifacts present, all processing steps marked deterministic, baseline stable or first-run condition. The evidence bundle is ready for inspection.
- **REQUEST_MORE_EVIDENCE**: Recoverable evidence gaps detected (e.g., ChangeGuard triggered REVIEW, missing artifacts, non-deterministic steps). User can re-upload or refine input.
- **PENDING_HUMAN_APPROVAL**: Unrecoverable manifest errors (e.g., invalid dimensions, unexpected data type, malformed metadata). **This outcome cannot be silently finalized or bypassed.** Manual review is required.

### Implementation

- **Non-diagnostic**: No medical claims. Evidence review documents processing metadata only.
- **Deterministic**: Same input always produces identical output and audit trace.
- **Local execution**: The Streamlit demo runs the agent locally. No Lambda invocation, no external API calls, no LLM.
- **Auditable**: Full decision trace (observations, tool steps, outcome) exported as JSON for inspection.

### Walkthrough

![Agentic Evidence Review overview: deterministic evidence-based finalization decision.](../../docs/images/opencv_aws_demo/09-agentic-evidence-overview.png)

![Agentic Evidence Review audit trace: OpenCV evidence observations, decision, and next action.](../../docs/images/opencv_aws_demo/10-agentic-trace-finalize.png)

## Verification

**Deterministic OpenCV test suite:**
```bash
pytest tests/test_opencv_mri.py -v
```
All processing is pixel-deterministic: identical input always produces identical output, SHA-256, and processing metadata.

**Torch-optional regression test (Railway deployment fix):**
```bash
pytest tests/test_streamlit_app_torch_free.py -v
```
The fresh-process Torch-optional regression test blocks `torch` and verifies that a Torch import does not prevent application startup. It specifically guards against the prior Railway `ModuleNotFoundError: No module named 'torch'`; it does not assert successful startup when other required dependencies are absent.

**Evidence Review Agent tests:**
```bash
pytest tests/test_opencv_review_agents.py -v
```
Deterministic review agent tests in `tests/test_opencv_review_agents.py` verify:
- **FINALIZE_EVIDENCE_BUNDLE**: Valid manifest + stable/first-run comparison
- **REQUEST_MORE_EVIDENCE**: ChangeGuard REVIEW or incomplete artifacts
- **PENDING_HUMAN_APPROVAL**: Malformed metadata, invalid dimensions, unexpected dtype
- Safety notice present in all outcomes
- All outcomes have auditable JSON trace with tool steps
- Deterministic reproducibility: same inputs → byte-for-byte identical trace
- No diagnostic language anywhere in reasoning or next actions

## AWS and Deployment

### Railway Streamlit — Interactive Visual Demo
**Status:** Live and running
**URL:** https://mri-vision-core-production-6dfd.up.railway.app

The Railway Streamlit app provides the interactive visual demonstration of the complete deterministic workflow:
- Local OpenCV processing (no ML model)
- ChangeGuard baseline comparison (deterministic thresholds)
- Agentic evidence review (three deterministic roles, mandatory human gate)

### AWS Lambda — Live Deployed API
**Status:** Deployed and tested
**Function URL:** https://gpint2fywbzbcy5crjldc2j6oi0pmxyb.lambda-url.us-east-1.on.aws/

The AWS Lambda Function URL provides a live, deployed evidence-manifest API that accepts JSON POST requests and returns deterministic evidence manifests.

**Sample Payload:** The Lambda accepts the documented sample JSON event from the repository. On October 23, 2026, a POST request using the repository sample event—not patient data—to the deployed AWS Function URL returned HTTP 200 with an evidence manifest.

**Handler:** `hackathon/opencv_aws/evidence_manifest.py::lambda_handler`

The repository includes:
- SAM template (`template.yaml`) for AWS Lambda deployment
- Lambda-ready Dockerfile (python:3.12 headless) for containerized invocation
- Evidence manifest capture logic with HTTP Function URL support
- Deterministic OpenCV 5 processing in the Lambda container
- Full test coverage with captured invocation evidence

## AWS Deployment Evidence

![AWS SAM deployment and successful direct Lambda invocation.](../../docs/images/opencv_aws_demo/07-aws-lambda-deployment-and-invocation.png)

![Successful POST to the deployed AWS Lambda Function URL, returning HTTP 200 and an evidence manifest.](../../docs/images/opencv_aws_demo/08-aws-function-url-post-success.png)

## Data and License

Any future MIT license applies to project-authored source code only. External datasets, images, and model assets remain governed by their original terms.

## Limitations

This workflow is **not a medical device** and has no clinical purpose. Specifically:
- **No diagnosis**: No diagnostic claims are made or permitted
- **No clinical interpretation**: The outputs are image-processing artifacts, not medical findings
- **No clinical performance claim**: No clinical validation, sensitivity, specificity, or predictive accuracy
- **Evidence is research only**: Deterministic workflow evidence documents processing, not medical truth

Use for education and research only.

---

**MRI Vision Core — A Deterministic, Research-Grade OpenCV Demonstration**
