"""Simulated Alexa+ MRI Evidence Copilot.

Non-diagnostic synthetic workflow demonstration. Local-only, deterministic.
No credentials required.

Run: streamlit run hackathon/amazon_alexa/streamlit_app.py
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

from hackathon.amazon_alexa.llm_explanation import (
    MODE_AUTO,
    MODE_DETERMINISTIC,
    _get_explanation_mode,
    generate_enhanced_explanation,
)
from hackathon.amazon_alexa.review_agents import generate_review_support
from hackathon.amazon_alexa.tools import (
    SYNTHETIC_CASES,
    analyze_mri_evidence,
    explain_review_status,
    request_human_approval,
)

APP_TITLE = "MRI Evidence Copilot — Amazon Alexa+ (Simulated)"

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stSelectbox"] [role="group"] {
        min-height: 64px !important;
        height: 64px !important;
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
        background-color: #1a2332 !important;
        border: 2px solid #00d4ff !important;
        border-radius: 8px !important;
        padding: 0 14px !important;
    }
    div[data-testid="stSelectbox"] [role="group"]:focus-within {
        border-color: #00e6ff !important;
        box-shadow: 0 0 8px rgba(0, 212, 255, 0.3) !important;
    }
    div[data-testid="stSelectbox"] input[role="combobox"] {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        height: 100% !important;
        background: transparent !important;
    }
    div[data-testid="stSelectbox"] [role="group"] button {
        width: 1.5rem !important;
        height: 1.5rem !important;
        flex-shrink: 0 !important;
    }
    div[data-testid="stSelectbox"] [role="group"] button svg {
        width: 1.5rem !important;
        height: 1.5rem !important;
    }
    div[data-testid="stSelectbox"] label p {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }
    [role="listbox"] [role="option"] {
        min-height: 52px !important;
        padding: 14px 16px !important;
        font-size: 1.15rem !important;
        display: flex !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)

st.warning(
    """
    **This is a deterministic, non-diagnostic synthetic workflow demonstration.**
    Not for medical diagnosis or clinical use. All results shown are from offline,
    deterministic demonstration scenarios with no external inference. Voice features run locally by default; cloud speech is opt-in.
    """
)

st.markdown(
    """
    ## About This Simulated Alexa+ Experience

    This is a **local-only proof of concept** for the Amazon Build, Ship, Shape hackathon.
    It demonstrates how the three core Alexa+ tools work with synthetic evidence data:
    1. **Analyze MRI Evidence** — Load a demonstration case
    2. **Explain Review Status** — Describe findings in natural language
    3. **Request Human Approval** — Record that human review is required
    """
)

# Case selector
case_labels = {
    "case-001-consistent": "1. Consistent baseline/current (no change)",
    "case-002-change": "2. Large-change review (needs review)",
    "case-003-uncertain": "3. Low-information uncertainty (low quality)",
}

st.markdown("### Select a Demonstration Case")
selected_case = st.selectbox(
    "Choose a case to analyze",
    list(SYNTHETIC_CASES.keys()),
    format_func=lambda k: case_labels.get(k, k),
)

# Step 1: Analyze
st.markdown("---")
st.markdown("### Step 1: Analyze MRI Evidence")
st.markdown(
    "Call `analyze_mri_evidence(case_id)` to load a demonstration case."
)

evidence = analyze_mri_evidence(selected_case)

col1, col2 = st.columns(2)
with col1:
    st.metric("Case ID", evidence.case_id)
    st.metric("Status", evidence.status.value)
    st.metric("Quality Score", f"{evidence.quality_score:.2f}")
with col2:
    st.metric("Foreground %", f"{evidence.foreground_percentage:.1f}%")
    st.metric("Mean Intensity", f"{evidence.mean_intensity:.1f}")
    st.metric("Baseline Date", evidence.baseline_date)

st.metric("Current Date", evidence.current_date)
st.metric("Change Description", evidence.change_description)

# Step 2: Explain
st.markdown("---")
st.markdown("### Step 2: Explain Review Status")
st.markdown(
    "Call `explain_review_status(evidence)` to describe findings in natural language."
)

explanation = explain_review_status(evidence)

st.subheader(explanation.summary)

with st.expander("Quality Assessment"):
    for note in explanation.quality_notes:
        st.markdown(f"• {note}")

with st.expander("Change Evidence"):
    for note in explanation.change_notes:
        st.markdown(f"• {note}")

if explanation.requires_human_approval:
    st.info(
        "⚠️ **Human Approval Required** — "
        "This case requires review by a named human reviewer. "
        "The copilot cannot approve cases."
    )

# Step 2A: Deterministic Review Support
st.markdown("---")
st.markdown("### Step 2A: Deterministic Review Support")
st.markdown(
    "Deterministic synthesis of evidence, critic findings, and reviewer-focused checks."
)

packet = generate_review_support(evidence, explanation)

st.markdown(f"**Evidence Summary:** {packet.evidence_summary}")

with st.expander("Critic Findings"):
    for finding in packet.critic_findings:
        st.markdown(f"• {finding}")

with st.expander("Reviewer Checks & Verification Steps"):
    for check in packet.reviewer_checks:
        st.markdown(f"{check}")

st.info(
    "ℹ️ This packet summarizes deterministic evidence and flagged concerns. "
    "A named human reviewer must evaluate the source images and make the final "
    "decision. The copilot cannot approve cases."
)

# Optional enhanced explanation
st.markdown("---")
st.markdown("#### Optional: Enhanced Explanation")

# Get default mode from environment
env_mode = _get_explanation_mode()
default_index = 1 if env_mode == MODE_AUTO else 0

exp_mode_select = st.selectbox(
    "Explanation mode",
    ["Deterministic only", "Auto (Bedrock → Groq → deterministic fallback)"],
    index=default_index,
    key="llm_explanation_mode",
)

# Determine if auto mode is selected
use_auto_mode = exp_mode_select == "Auto (Bedrock → Groq → deterministic fallback)"

# Show provider availability without exposing configuration
if use_auto_mode:
    st.caption("⚠️ Auto mode enabled — explanation will use configured providers if available.")
else:
    st.caption("Deterministic-only mode — no network calls.")

if st.button("Generate enhanced explanation", key="generate_llm_explanation"):
    with st.spinner("Generating explanation..."):
        # Pass environment and mode to generator
        import os
        mode_to_use = MODE_AUTO if use_auto_mode else MODE_DETERMINISTIC
        os.environ["EXPLANATION_MODE"] = mode_to_use
        explanation = generate_enhanced_explanation(packet)
        # Bind explanation to current case ID to prevent stale display
        st.session_state["llm_explanation"] = explanation
        st.session_state["llm_explanation_case_id"] = evidence.case_id

# Display explanation only if it matches the current case (prevent stale explanations across case changes)
if "llm_explanation" in st.session_state:
    if st.session_state.get("llm_explanation_case_id") == evidence.case_id:
        exp = st.session_state["llm_explanation"]
        source_badge = f"🤖 **Source:** {exp.source}"
        st.markdown(source_badge)
        st.info(exp.text)
    else:
        # Case changed; clear stale explanation
        st.session_state.pop("llm_explanation", None)
        st.session_state.pop("llm_explanation_case_id", None)

# Step 3: Request Approval
st.markdown("---")
st.markdown("### Step 3: Request Human Approval")
st.markdown(
    "Call `request_human_approval(case_id, reviewer_name)` to record that "
    "human review is required. The copilot cannot approve; only a named reviewer can."
)

col1, col2 = st.columns(2)
with col1:
    reviewer = st.text_input("Reviewer Name", placeholder="e.g., Dr. Smith")
with col2:
    comment = st.text_area("Reviewer Comment (optional)", placeholder="e.g., checked source images")

if st.button("Request Approval", type="primary"):
    if not reviewer or not reviewer.strip():
        st.error("Please enter a reviewer name.")
    else:
        try:
            approval = request_human_approval(
                evidence.case_id, reviewer, comment
            )
            st.success(
                f"✓ Approval request recorded for **{approval.reviewer_name}**"
            )
            st.markdown(f"**Approval State:** `{approval.approval_state.value}`")
            st.markdown(
                "ℹ️ Only the named human reviewer can transition this to "
                "`APPROVED_BY_HUMAN` in a live clinical system."
            )
        except ValueError as e:
            st.error(f"Error: {e}")

# Voice assistant (push-to-talk, local by default)
from hackathon.amazon_alexa.voice_ui import render_voice_assistant

render_voice_assistant(evidence.case_id, reviewer, comment)

# Safety note
st.markdown("---")
st.markdown(
    """
    ## Safety & Architecture

    ### Non-Diagnostic
    This copilot returns evidence and status explanations only. It does not make
    clinical diagnoses or treatment recommendations. All outputs are clearly marked
    as non-diagnostic and require human review.

    ### Deterministic & Local
    - Evidence logic makes no network calls and uses no LLM
    - Voice is local by default (AssemblyAI / Amazon Polly are opt-in)
    - No credentials or API keys required
    - All logic is deterministic and reproducible

    ### Human Approval Safeguard
    - Every case starts in `PENDING_HUMAN_APPROVAL`
    - Only a named human reviewer can transition to `APPROVED_BY_HUMAN`
    - The copilot itself cannot approve cases
    - This safeguard is non-bypassable

    ### Synthetic Demonstration Data
    All evidence shown is synthetic and used only for workflow validation.
    This is not real patient data.
    """
)
