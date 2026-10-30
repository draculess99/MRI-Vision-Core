"""Streamlit UI for MRI Evidence Dossier with optional Nebius reviewer brief integration.

Four deterministic synthetic scenarios with optional button to generate reviewer brief
for the Nebius scenario only. The brief is generated on-demand and cannot mutate status
or approval. Non-diagnostic workflow summary only.
"""

import json
import time
from typing import Optional

import streamlit as st

from hackathon.nebius_nvidia.eval_harness import default_scenarios
from hackathon.nebius_nvidia.evidence_dossier import (
    EvidenceDossier,
    build_dossier_from_images,
)
from hackathon.nebius_nvidia.local_env import get_nebius_config
from hackathon.nebius_nvidia.nemotron_review import (
    NebiusConfig,
    ReviewerBrief,
    UrllibTransport,
    fallback_brief,
    generate_reviewer_brief,
)


st.set_page_config(page_title="MRI Evidence Council", layout="wide")

st.title("MRI Evidence Council — Nebius × NVIDIA")
st.markdown(
    "Non-diagnostic, evidence-grounded review layer using deterministic MRI Core analysis "
    "with optional Nebius-powered reviewer brief generation."
)

if "current_brief_scenario" not in st.session_state:
    st.session_state.current_brief_scenario = None
if "current_brief" not in st.session_state:
    st.session_state.current_brief = None


def build_scenario_dossier(sc) -> EvidenceDossier:
    """Build dossier from scenario images."""
    return build_dossier_from_images(sc.image, baseline_image=sc.baseline)


def render_dossier_section(name: str, dossier: EvidenceDossier) -> None:
    """Render dossier fields as read-only info."""
    st.subheader("Evidence Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Status", dossier.status.value)
        st.metric(
            "Requires Human Approval",
            "Yes" if dossier.requires_human_approval else "No",
        )
    with col2:
        st.metric("Approval State", dossier.approval.state.value)
        if dossier.approval.reviewer:
            st.metric("Reviewer", dossier.approval.reviewer)

    st.subheader("Quality Evidence")
    q = dossier.quality
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dimensions", f"{q.width}×{q.height}")
        st.metric("Mean Intensity", f"{q.mean_intensity:.1f}")
    with col2:
        st.metric("Std Deviation", f"{q.std_deviation:.1f}")
        st.metric("Foreground %", f"{q.foreground_percentage:.1f}%")
    with col3:
        flags_text = ", ".join(q.flags) if q.flags else "none"
        st.text(f"Quality Flags:\n{flags_text}")

    if dossier.changeguard is not None:
        st.subheader("Change Evidence")
        st.metric("ChangeGuard Status", dossier.changeguard.status.value)
        deltas_text = ", ".join(
            f"{k}={v:+.2f}" for k, v in dossier.changeguard.deltas.items()
        )
        st.text(f"Deltas: {deltas_text}")
        if dossier.changeguard.reasons:
            st.text("Reasons:")
            for reason in dossier.changeguard.reasons:
                st.caption(f"  • {reason}")

    if dossier.review_notes:
        st.subheader("Review Notes")
        for note in dossier.review_notes:
            st.caption(note)


def render_brief_section(brief: ReviewerBrief) -> None:
    """Render a reviewer brief with source indication."""
    source_label = "✓ Validated Nebius Model" if brief.source == "model" else "⧉ Deterministic Fallback"
    st.info(f"**Brief Source:** {source_label}")

    if brief.fallback_reason:
        st.caption(f"Fallback reason: {brief.fallback_reason}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Review Priority", brief.review_priority)
    with col2:
        st.metric("Safety Statement", "Non-diagnostic")

    st.markdown("**Evidence Summary**")
    st.text(brief.evidence_summary)

    st.markdown("**Why Human Review Is Needed**")
    st.text(brief.why_human_review_is_needed)

    st.markdown("**Recommended Workflow Action**")
    st.text(brief.recommended_workflow_action)

    if brief.missing_or_uncertain_evidence:
        st.markdown("**Missing or Uncertain Evidence**")
        for item in brief.missing_or_uncertain_evidence:
            st.caption(f"  • {item}")

    st.caption(brief.safety_statement)


def generate_brief_with_config(dossier: EvidenceDossier) -> Optional[ReviewerBrief]:
    """Generate a reviewer brief using Nebius credentials from environment or .env, else use fallback."""
    api_key, base_url, model = get_nebius_config()

    if not (api_key and base_url and model):
        st.info(
            "Nebius configuration not set. Using deterministic fallback brief. "
            "To use the Nebius model, set NEBIUS_API_KEY, NEBIUS_BASE_URL, "
            "and NEBIUS_MODEL via environment variables or in the repository-root .env file."
        )
        return fallback_brief(dossier, "not_configured")

    try:
        config = NebiusConfig(api_key=api_key, base_url=base_url, model=model)
        transport = UrllibTransport(config, timeout=90.0)
        return generate_reviewer_brief(dossier=dossier, transport=transport, config=config)
    except Exception as e:
        st.error(f"Failed to generate brief: {type(e).__name__}")
        return None


def main() -> None:
    """Render the Streamlit interface."""
    st.markdown(
        "> **Disclaimer:** This is NOT a diagnostic tool. All outputs are "
        "evidence-grounded supporting analyses intended to assist human medical professionals. "
        "No clinical diagnosis or treatment recommendations are made."
    )

    scenarios = default_scenarios()

    for i, sc in enumerate(scenarios, 1):
        with st.expander(f"Scenario {i}: {sc.name}", expanded=(i == 1)):
            dossier = build_scenario_dossier(sc)

            render_dossier_section(sc.name, dossier)

            st.divider()
            st.subheader("Optional Nebius Reviewer Brief")

            if st.button(
                "Generate Reviewer Brief",
                key=f"brief_button_{i}",
                help="Click to generate a reviewer brief using Nebius model (or fallback)."
            ):
                with st.spinner("Generating brief..."):
                    start = time.time()
                    brief = generate_brief_with_config(dossier)
                    elapsed = time.time() - start
                    st.session_state.current_brief_scenario = i
                    st.session_state.current_brief = brief

            if st.session_state.current_brief and st.session_state.current_brief_scenario == i:
                if st.session_state.current_brief:
                    st.success(f"Brief generated from this scenario's dossier")
                    render_brief_section(st.session_state.current_brief)

                    if st.checkbox(
                        "Show raw JSON",
                        key=f"json_toggle_{i}",
                    ):
                        brief_dict = {
                            "source": st.session_state.current_brief.source,
                            "review_priority": st.session_state.current_brief.review_priority,
                            "evidence_summary": st.session_state.current_brief.evidence_summary,
                            "why_human_review_is_needed": st.session_state.current_brief.why_human_review_is_needed,
                            "missing_or_uncertain_evidence": list(st.session_state.current_brief.missing_or_uncertain_evidence),
                            "recommended_workflow_action": st.session_state.current_brief.recommended_workflow_action,
                            "safety_statement": st.session_state.current_brief.safety_statement,
                            "fallback_reason": st.session_state.current_brief.fallback_reason,
                        }
                        st.json(brief_dict)


if __name__ == "__main__":
    main()
