# MRI Evidence Copilot — Amazon Alexa+ (Simulated)

## Overview

A local-only, non-diagnostic MRI evidence copilot prototype for the Amazon Build, Ship, Shape hackathon. This is a **simulated Alexa+ experience** demonstrating how voice and UI-based tools can expose structured MRI evidence analysis to clinicians without requiring network calls, LLM inference, or external credentials.

## Important Disclaimer

**This is NOT a diagnostic tool.** All outputs are evidence-grounded supporting analyses intended to assist human medical professionals in their review process. No clinical diagnosis or treatment recommendations are made by this system. A qualified radiologist or physician must review and validate all findings.

## Architecture

```
hackathon/amazon_alexa/
├── tools.py              # Deterministic tool layer (three core functions)
├── streamlit_app.py      # Simulated UI/voice experience
└── README.md             # This file

Three Core Tools (fully deterministic, no LLM, no network):
  1. analyze_mri_evidence(case_id)
       → Returns synthetic MRIEvidence record
  2. explain_review_status(evidence)
       → Returns ReviewExplanation with quality and change notes
  3. request_human_approval(case_id, reviewer_name, comment?)
       → Returns ApprovalRequest (always PENDING_HUMAN_APPROVAL)
```

## Design Principles

- **Deterministic**: No randomness, no external API calls, fully reproducible
- **Non-Diagnostic**: Evidence and status only, no clinical conclusions
- **Transparent**: All findings grounded in evidence fields
- **Safe**: Human approval cannot be bypassed; the copilot cannot approve
- **Local-Only**: No credentials, no network access required
- **Synthetic**: All demonstration data is clearly marked as synthetic

## Tool Contracts

### 1. `analyze_mri_evidence(case_id: str) -> MRIEvidence`

Load a demonstration case with synthetic evidence.

**Input:**
- `case_id` — One of:
  - `case-001-consistent` — Baseline matches current image
  - `case-002-change` — Notable change from baseline (needs review)
  - `case-003-uncertain` — Low image quality (insufficient information)

**Returns:**
```python
MRIEvidence(
    case_id: str,
    baseline_date: str,
    current_date: str,
    quality_score: float,      # 0.0–1.0
    foreground_percentage: float,
    mean_intensity: float,
    status: ReviewStatus,      # CONSISTENT | NEEDS_REVIEW | UNCERTAIN
    change_description: str    # Natural language summary
)
```

**Example:**
```python
evidence = analyze_mri_evidence("case-001-consistent")
# → MRIEvidence(
#     case_id='case-001-consistent',
#     status=ReviewStatus.CONSISTENT,
#     quality_score=0.95,
#     ...
#   )
```

### 2. `explain_review_status(evidence: MRIEvidence) -> ReviewExplanation`

Describe the review status in natural language.

**Input:**
- `evidence` — Result from `analyze_mri_evidence()`

**Returns:**
```python
ReviewExplanation(
    status: ReviewStatus,
    summary: str,              # One-sentence summary
    quality_notes: list[str],  # Assessment of image quality
    change_notes: list[str],   # Assessment of evidence changes
    requires_human_approval: bool  # Always True
)
```

**Example:**
```python
explanation = explain_review_status(evidence)
print(explanation.summary)
# → "Current image is consistent with baseline."
print(explanation.quality_notes)
# → ["Image quality is good.", "Foreground fills 45.2% of the image."]
```

### 3. `request_human_approval(case_id: str, reviewer_name: str, comment: str = "") -> ApprovalRequest`

Record that human review is required.

**Input:**
- `case_id` — Case identifier
- `reviewer_name` — Name of human reviewer (required, non-empty)
- `comment` — Optional reviewer comment

**Returns:**
```python
ApprovalRequest(
    case_id: str,
    approval_state: ApprovalState,  # Always PENDING_HUMAN_APPROVAL
    reviewer_name: str,
    comment: Optional[str]
)
```

**Raises:**
- `ValueError` — If `reviewer_name` is empty or whitespace-only

**Safeguard:**
The returned `ApprovalRequest` always has `approval_state == ApprovalState.PENDING_HUMAN_APPROVAL`. The copilot **cannot** transition to `APPROVED_BY_HUMAN`; only a named human reviewer can do that in a live clinical system.

**Example:**
```python
approval = request_human_approval(
    "case-001-consistent",
    "Dr. Smith",
    "checked source images"
)
print(approval.approval_state)
# → ApprovalState.PENDING_HUMAN_APPROVAL
```

## Differences from Other Hackathon Tracks

### vs. OpenCV Track
- **OpenCV track**: Image processing and feature extraction
- **Alexa+ track**: Structured evidence analysis and voice-friendly tool layer
- **This project**: Reuses OpenCV feature outputs (if available) but focuses on evidence interpretation

### vs. Nebius/NVIDIA Track
- **Nebius track**: Multi-agent consensus, optional LLM brief generation, external API integration
- **Alexa+ track**: Single deterministic layer, no LLM, local-only, voice-first
- **This project**: Simpler, fully deterministic, designed for low-latency voice response

## Try the Interactive Demo

Run the deterministic Streamlit demo locally (no credentials required):

```bash
streamlit run hackathon/amazon_alexa/streamlit_app.py
```

The demo shows:
- Three synthetic scenarios with evidence metrics
- Step-by-step demonstration of the three core tools
- Human approval safeguards in action
- Non-diagnostic notices and safety disclaimers

## Demo Walkthrough

This walkthrough demonstrates the copilot workflow using **Case 002: Large-Change Review**.

### Step 1: Analyze MRI Evidence

The copilot loads a demonstration case with deterministic change evidence:

```python
evidence = analyze_mri_evidence("case-002-change")
# status: NEEDS_REVIEW
# change_description: "Foreground changed by 16.9%, mean intensity by 16.8. Review needed."
```

![Step 1: Evidence Analysis](docs/images/amazon_alexa_demo/amazon_alexa_step1_needs_review.png)

The evidence is purely evidence-based: foreground percentage and mean intensity differences trigger a `NEEDS_REVIEW` status. No clinical interpretation is made by the copilot.

### Step 2: Explain Review Status

The copilot describes the findings in natural language without diagnosing:

```python
explanation = explain_review_status(evidence)
# summary: "Current image shows notable differences from baseline."
# quality_notes: ["Image quality is good."]
# change_notes: ["Foreground changed by 16.9%, mean intensity by 16.8. Review needed."]
```

The explanation is non-diagnostic—it describes what the evidence shows, not what it means clinically.

### Step 3: Request Human Approval

A named human reviewer is required. The copilot cannot approve the case:

```python
approval = request_human_approval(
    "case-002-change",
    "Dr. Smith",
    "checked source images"
)
# approval_state: PENDING_HUMAN_APPROVAL (always)
# reviewer_name: "Dr. Smith"
```

![Step 3: Human Approval Request](docs/images/amazon_alexa_demo/amazon_alexa_step3_pending_human_approval.png)

The approval state always remains `PENDING_HUMAN_APPROVAL` on the copilot side. Only the named human reviewer can transition it to `APPROVED_BY_HUMAN` in a live clinical system—the copilot has no approval method.

### Key Safeguards

- **No Diagnosis:** The copilot explains evidence and status only.
- **Non-Bypassable Approval:** Every case requires a named human reviewer.
- **Deterministic:** The same input always produces the same output.
- **Fully Local:** No network calls, no LLM inference.

## Run Tests & Demo

### Streamlit Interactive Demo (Recommended)

Launch the Streamlit application:

```bash
streamlit run hackathon/amazon_alexa/streamlit_app.py
```

### Focused Unit Tests

Run the Alexa+ module tests:

```bash
pytest -q tests/test_amazon_alexa_tools.py tests/test_amazon_alexa_approval.py
```

### Test Coverage

- **test_amazon_alexa_tools.py**:
  - Each of the three tools returns expected types
  - Synthetic cases load correctly
  - Evidence analysis is deterministic
  - Natural language explanations are consistent

- **test_amazon_alexa_approval.py**:
  - Human approval cannot be bypassed
  - Named reviewer is required
  - Approval state is always PENDING until human acts
  - Empty reviewer names are rejected

## Synthetic Demonstration Scenarios

### 1. Consistent Baseline/Current

Baseline and current images are consistent in measured evidence fields.

- **Status:** `CONSISTENT`
- **Quality Score:** 0.95 (good)
- **Change:** No significant changes
- **Approval State:** `PENDING_HUMAN_APPROVAL`

### 2. Large-Change Review

Current image differs substantially from baseline in measured features.

- **Status:** `NEEDS_REVIEW`
- **Quality Score:** 0.92 (acceptable)
- **Change:** Foreground +16.9%, mean intensity +16.8
- **Approval State:** `PENDING_HUMAN_APPROVAL`

### 3. Low-Information Uncertainty

Input has low quality and insufficient information for confident assessment.

- **Status:** `UNCERTAIN`
- **Quality Score:** 0.68 (low)
- **Change:** Image quality is low
- **Approval State:** `PENDING_HUMAN_APPROVAL`

---

## Safety & Architecture Notes

### Human Approval Safeguard

Every case starts in `PENDING_HUMAN_APPROVAL` and requires a named human reviewer to transition to `APPROVED_BY_HUMAN`. This safeguard is **non-bypassable**:

```python
evidence = analyze_mri_evidence("case-002-change")
approval = request_human_approval("case-002-change", "Dr. Smith")
# approval.approval_state == ApprovalState.PENDING_HUMAN_APPROVAL  # Always true
# Only Dr. Smith (in a live system) can call: approval.decide(approve=True, ...)
```

### Deterministic & Reproducible

- No randomness in evidence calculation
- No external API calls or network access
- Same input always produces same output
- Safe for testing, validation, and compliance audits

### Non-Diagnostic by Design

- All outputs are evidence and status only
- No clinical diagnosis or recommendations
- All outputs include non-diagnostic notices
- Requires human review before any clinical use

---

**Note:** This is a demonstration project for the Amazon hackathon. Synthetic workflow-validation metrics only; not clinical accuracy or medical performance. Not for production use without significant additional validation and regulatory review.
