"""Deterministic, offline Streamlit demo for MRI Evidence Dossier.

Synthetic workflow validation only. Non-diagnostic. No credentials required.
Run: streamlit run hackathon/nebius_nvidia/streamlit_app.py
"""

import sys
from pathlib import Path

# Bootstrap repository root into sys.path for local launches
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

from hackathon.nebius_nvidia.eval_harness import (
    METRICS_LABEL,
    default_scenarios,
    run_evaluation,
)
from hackathon.nebius_nvidia.evidence_dossier import build_dossier_from_images

APP_TITLE = "MRI Evidence Dossier — Nebius × NVIDIA"

st.set_page_config(page_title="MRI Evidence Dossier", layout="wide")

# Scoped CSS for large, visually prominent scenario selector
st.markdown(
    """
    <style>
    /* Visible bordered box: Streamlit renders this with react-aria-components as
       a div[role="group"] wrapping the input and open-button (no BaseWeb markup). */
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

    /* Selected text input */
    div[data-testid="stSelectbox"] input[role="combobox"] {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        height: 100% !important;
        background: transparent !important;
    }

    /* Open/close button (dropdown arrow) */
    div[data-testid="stSelectbox"] [role="group"] button {
        width: 1.5rem !important;
        height: 1.5rem !important;
        flex-shrink: 0 !important;
    }

    div[data-testid="stSelectbox"] [role="group"] button svg {
        width: 1.5rem !important;
        height: 1.5rem !important;
    }

    /* Widget label */
    div[data-testid="stSelectbox"] label p {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }

    /* Opened dropdown menu options (portaled listbox) */
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
deterministic scenarios with no external inference.
"""
)

# Tabs for different views
tab_scenarios, tab_metrics, tab_integration = st.tabs(
    ["Scenarios", "Metrics", "Nebius Integration"]
)

# ============================================================================
# Scenarios Tab
# ============================================================================
with tab_scenarios:
    st.subheader("Synthetic Evidence Dossier Scenarios")
    st.markdown(
        "Four curated scenarios demonstrate the workflow: "
        "consistent baseline, large change, low information, and named-reviewer approval."
    )

    scenarios = default_scenarios()

    # Human-readable scenario labels
    scenario_labels = {
        0: "1. Consistent baseline/current",
        1: "2. Large-change review",
        2: "3. Low-information uncertainty",
        3: "4. Named-reviewer approval",
    }

    st.markdown("Select a scenario to view its non-diagnostic evidence dossier.")
    selected_idx = st.selectbox(
        "Choose a synthetic workflow scenario",
        range(len(scenarios)),
        format_func=lambda i: scenario_labels.get(i, f"{i+1}. {scenarios[i].name}"),
    )

    sc = scenarios[selected_idx]
    dossier = build_dossier_from_images(sc.image, baseline_image=sc.baseline)
    q = dossier.quality

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Status", dossier.status.value)
        st.metric("Image Size", f"{q.width}×{q.height}")
        st.metric("Mean Intensity", f"{q.mean_intensity:.1f}")
        st.metric("Std Deviation", f"{q.std_deviation:.1f}")
        st.metric("Foreground %", f"{q.foreground_percentage:.1f}%")

    with col2:
        st.metric(
            "Quality Flags", ", ".join(q.flags) if q.flags else "None"
        )
        if dossier.changeguard:
            deltas_text = ", ".join(
                f"{k}={v:+.2f}" for k, v in dossier.changeguard.deltas.items()
            )
            st.metric("Change Evidence", f"{dossier.changeguard.status.value} ({deltas_text})")
        else:
            st.metric("Change Evidence", "No baseline")

    st.subheader("Review & Approval")
    st.markdown(f"**Review Note:** {dossier.notice}")
    st.markdown(
        f"**Approval State:** `{dossier.approval.state.value}` "
        f"(human approval required: yes)"
    )

    if sc.reviewer_decision:
        st.subheader("Named-Reviewer Approval Transition")
        decided = dossier.decide(True, reviewer="Synthetic Reviewer", comment="demo review")
        col_before, col_arrow, col_after = st.columns([2, 1, 2])
        with col_before:
            st.markdown("**Before Decision**")
            st.code(dossier.approval.state.value)
        with col_arrow:
            st.markdown("→")
        with col_after:
            st.markdown("**After Decision**")
            st.code(decided.approval.state.value)
        st.markdown(
            f"Reviewer: **{decided.approval.reviewer}** | "
            f"Comment: *{decided.approval.comment}*"
        )
        st.info(
            "Safeguard: Named human reviewer required to transition approval state. "
            "Human approval requirement is retained even after decision."
        )


# ============================================================================
# Metrics Tab
# ============================================================================
with tab_metrics:
    st.subheader("Evaluation Metrics")
    st.markdown(f"**{METRICS_LABEL}**")

    metrics = run_evaluation()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Scenarios Evaluated", metrics["scenarios_evaluated"])
        st.metric(
            "Expected-Status Agreement",
            f"{metrics['expected_status_agreement']:.0%}",
        )

    with col2:
        st.metric(
            "Evidence-Field Completeness",
            f"{metrics['evidence_field_completeness']:.0%}",
        )
        st.metric(
            "Human-Approval Gate Enforcement",
            f"{metrics['human_approval_gate_enforcement_rate']:.0%}",
        )

    st.metric(
        "Mean Dossier-Build Latency",
        f"{metrics['mean_dossier_build_latency_ms']:.1f} ms",
    )

    st.markdown("### Per-Scenario Results")
    for sc_result in metrics["per_scenario"]:
        with st.expander(sc_result["name"]):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Expected Status", sc_result["expected_status"])
                st.metric("Actual Status", sc_result["actual_status"])
            with col2:
                st.metric("Status Match", "OK" if sc_result["status_match"] else "FAIL")
                st.metric("Gate Enforced", "OK" if sc_result["gate_enforced"] else "FAIL")
            st.metric("Completeness", f"{sc_result['completeness']:.0%}")


# ============================================================================
# Integration Tab
# ============================================================================
with tab_integration:
    st.subheader("Nebius × NVIDIA Token Factory Integration")
    st.markdown(
        """
This public demo runs entirely **offline and deterministically**.

The Nebius Token Factory integration:
- **Live reviewer path** (validated separately): Uses NVIDIA/NVIDIA-Nemotron-3-Nano-30B-A3B
  for optional LLM-assisted brief generation when credentials are available.
- **This public demo**: Uses the deterministic workflow only.
  This demo intentionally runs the deterministic workflow only.
  The live integration was validated separately.
- **Approval safeguard**: Human approval cannot be bypassed. Every dossier
  starts `PENDING_HUMAN_APPROVAL` and requires a named human reviewer to transition
  to `APPROVED_BY_HUMAN` or `REJECTED_BY_HUMAN`.
- **Deterministic fallback**: If the LLM call fails, times out, or returns invalid
  content, the system gracefully falls back to a brief derived only from
  existing evidence fields, preserving the approval safeguard.
    """
    )
