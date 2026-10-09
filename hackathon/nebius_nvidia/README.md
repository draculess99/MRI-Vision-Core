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
