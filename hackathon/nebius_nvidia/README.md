# MRI Evidence Council — Nebius × NVIDIA

## Overview

MRI Evidence Council is a non-diagnostic, evidence-grounded multi-agent MRI review layer built as part of the Nebius × NVIDIA hackathon initiative.

## Architecture

This system reuses the deterministic **MRI Core** and **ChangeGuard** evidence layer, then adds:

- **Specialist Review**: Domain-specific analysis of MRI findings
- **Critic/Verifier Checks**: Automated validation and cross-checks
- **Consensus Mechanism**: Multi-agent agreement on findings
- **Human Approval Gate**: Final human authorization before any conclusions

## Important Disclaimer

**This is NOT a diagnostic tool.** All outputs are evidence-grounded supporting analyses intended to assist human medical professionals in their review process. No clinical diagnosis or treatment recommendations are made by this system. A qualified radiologist or physician must review and validate all findings.

## Design Principles

- Evidence-grounded: All findings traced to deterministic MRI Core processing
- Non-diagnostic: Designed to support, not replace, human clinical judgment
- Transparent: Each step documented and verifiable
- Safe: Human approval gate ensures clinical oversight

## Try the Interactive Demo

Run the deterministic Streamlit demo locally (no credentials required):

```bash
streamlit run hackathon/nebius_nvidia/streamlit_app.py
```

The demo shows:
- Four synthetic scenarios demonstrating the workflow
- Evidence facts, status, and review notes for each scenario
- The named-reviewer approval transition (PENDING → APPROVED)
- Synthetic workflow-validation metrics
- Information about the Nebius Token Factory integration

## Nebius × NVIDIA MRI Evidence Dossier — Demo Walkthrough

These are deterministic, synthetic workflow demonstrations only. They are **non-diagnostic** and do not represent patient cases or clinical conclusions.

### 1. Consistent baseline/current

Current synthetic image matches the baseline in the measured evidence fields.

- **Expected status:** CONSISTENT
- **Quality flags:** None
- **Approval state:** PENDING_HUMAN_APPROVAL — the application cannot approve anything without a named human reviewer

### 2. Large-change review

Current synthetic image differs substantially from its baseline in measured image features.

- **Expected status:** REVIEW
- **Workflow behavior:** Flags the difference for human review; does not interpret it clinically
- **Approval state:** PENDING_HUMAN_APPROVAL — requires named human reviewer

### 3. Low-information uncertainty

Input is deliberately low-information / low-contrast.

- **Expected status:** UNCERTAIN
- **Quality flags:** low_contrast, foreground_nearly_full
- **Change evidence:** No baseline
- **Approval state:** PENDING_HUMAN_APPROVAL — requires named human reviewer

### 4. Named-reviewer approval

Begins as a REVIEW scenario with PENDING_HUMAN_APPROVAL.

- **Demonstration:** Visibly records a named synthetic reviewer transition: PENDING_HUMAN_APPROVAL → APPROVED_BY_HUMAN
- **Safety gate:** Only the named human reviewer can make that transition; the application and optional LLM path cannot bypass the approval safeguard

---

**Note:** Synthetic workflow-validation metrics only; not clinical accuracy or medical performance.

## Synthetic Workflow-Validation Metrics

- **4 synthetic scenarios evaluated**
- **100% expected-status agreement**
- **100% evidence-field completeness**
- **100% human-approval gate enforcement**
- **Mean dossier-build latency:** Shown by the local run

## Nebius Token Factory Integration

Optional Nebius/Nemotron integration is guarded by strict JSON validation and has a deterministic fallback. This demo does not claim a live model call was performed.

## Run Tests & Demo

### Streamlit Interactive Demo (Recommended)

Launch the Streamlit application (no credentials required):

```bash
streamlit run hackathon/nebius_nvidia/streamlit_app.py
```

### Terminal Deterministic Demo

Run the offline deterministic demonstration:

```bash
python -m hackathon.nebius_nvidia.demo
```

### Focused Tests

Run the Nebius module tests:

```bash
pytest -q tests/test_nebius_seed.py tests/test_nebius_dossier.py tests/test_nebius_eval_harness.py tests/test_nebius_demo.py tests/test_nebius_nemotron_review.py
```
