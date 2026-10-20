"""Interactive local demo: a real DICOM study, a real bridged Kaggle checkpoint, the shared
evidence-review agents, and a human-review gate -- all running locally on CPU, no network
calls, no credentials required.

Run with:
    streamlit run hackathon/amd/streamlit_app.py

Everything here reuses existing, already-tested modules; this file only wires them together
for an interactive UI:
  - study discovery / DICOM loading : mri_core.rsna_integration (unmodified)
  - checkpoint + forward pass       : hackathon.amd.kaggle_mil_bridge (unmodified)
  - StudyInferenceResult adapter    : hackathon.amd.agent_bridge_adapter (new, tested)
  - specialist/critic/consensus     : mri_core.agents.orchestrator (unmodified)
  - human review                   : mri_core.agents.review_gate, via
                                      hackathon.amd.review_actions (new, tested)
  - CPU benchmark evidence          : outputs/amd/benchmark_kaggle_bridge.json, if present
  - optional AMD provider           : hackathon.amd.providers (disabled by default)

No clinical conclusion is ever rendered: every panel below either restates a raw research
score verbatim, a deterministic structural check, or a simulated workflow/review status.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# `streamlit run` (and AppTest.from_file) execute this file standalone, outside the
# `hackathon.amd` package -- relative imports fail with "attempted relative import with
# no known parent package". Ensure the repo root is importable so the absolute imports
# below (hackathon.amd.*, mri_core.*) resolve the same way they do everywhere else in
# this project, without needing PYTHONPATH set externally.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from hackathon.amd.agent_bridge_adapter import to_study_inference_result
from hackathon.amd.benchmark_scenarios import BenchmarkScenarioError, build_labels, load_scenarios
from hackathon.amd.data_config import resolve_data_paths, validate_data_paths
from hackathon.amd.kaggle_mil_bridge import BridgeError, load_bridge_checkpoint, run_study_inference
from hackathon.amd.llm_config import build_adapter, resolve_llm_config
from hackathon.amd.public_demo_cases import (
    PublicDemoCasesError, build_case_inputs, case_label, load_public_demo_cases,
)
from hackathon.amd.llm_providers import LocalDeterministicAdapter, propose_with_fallback, redact_for_external_llm
from hackathon.amd.review_actions import approve, escalate, is_escalation, request_more_evidence
from hackathon.amd.run_real_checkpoint_smoke import NON_DIAGNOSTIC_NOTICE, pick_study_and_series, sha256_of_file
from hackathon.amd.synthetic_demo import build_synthetic_demo_result
from mri_core.agents.contracts import QualityGateStatus, ReviewState
from mri_core.agents.llm_adapter import ProposalStatus, screen_proposal
from mri_core.agents.orchestrator import process_inference_result
from mri_core.agents.review_gate import InvalidTransitionError, ReviewGate, StaleEvidenceError

st.set_page_config(page_title="MRI-Vision-Core — AMD bridged demo", layout="wide")

# Cosmetic only -- makes the curated-benchmark scenario selector stand out for screenshots/
# video. Scoped to the `.st-key-benchmark_scenario_selector` container (via st.container(key=...),
# which Streamlit renders as that exact CSS class) so this never touches the real-study selectbox
# or any other widget on the page. Yellow outline only -- no colored fill -- per direct feedback;
# the dropdown listbox/options below could not be scoped the same way (Streamlit/BaseWeb renders
# the open popup outside this container's DOM subtree), so that part is necessarily global.
st.markdown(
    """
    <style>
    .st-key-benchmark_scenario_selector div[data-baseweb="select"] > div {
        min-height: 64px !important;
        border: 3px solid #f5c518 !important;
        border-radius: 10px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    .st-key-benchmark_scenario_selector div[data-baseweb="select"] span {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    div[role="listbox"] {
        border: 3px solid #f5c518 !important;
    }
    div[role="option"] {
        min-height: 56px !important;
        font-size: 17px !important;
        padding: 14px 14px !important;
    }
    div[role="option"]:hover {
        background-color: #4a3f10 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------- helpers


def _list_available_studies(data_root: Path):
    from mri_core.rsna_integration import discover_available_studies, load_rsna_metadata_safe

    metadata = load_rsna_metadata_safe(data_root)
    return discover_available_studies(data_root, metadata)


@st.cache_resource(show_spinner="Loading checkpoint...")
def _cached_load_checkpoint(checkpoint_path: str):
    return load_bridge_checkpoint(Path(checkpoint_path))


def _run_inference(data_root: Path, checkpoint_path: Path, study_uid: str | None):
    study_uid, series_uid, plane = pick_study_and_series(data_root, study_uid)
    series_dir = data_root / "raw" / "train_series" / study_uid / series_uid
    model, model_provenance = _cached_load_checkpoint(str(checkpoint_path))
    started = time.monotonic()
    inference = run_study_inference(model, series_dir)
    latency_seconds = time.monotonic() - started
    result = to_study_inference_result(
        study_uid=study_uid,
        probabilities=inference["probabilities"],
        latency_seconds=latency_seconds,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=sha256_of_file(checkpoint_path),
        disclaimer=NON_DIAGNOSTIC_NOTICE,
        epoch=model_provenance.get("lightning_epoch"),
    )
    return study_uid, series_uid, plane, result, model_provenance


def sanitize_fallback_reason(reason: str, limit: int = 140) -> str:
    """Short, display-safe reason for a fallback: drops response bodies, redacts anything that looks
    like a credential, collapses whitespace and truncates."""
    text = re.sub(r"(Bearer\s+\S+|gsk_\w+|sk-[\w-]+)", "[redacted]", str(reason))
    text = re.split(r":\s*['\"]", text, maxsplit=1)[0]  # cut a quoted response body
    text = " ".join(text.split()).strip(" .")
    if not text:
        return "provider unavailable"
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


PUBLIC_DEMO_ENV_VAR = "AMD_PUBLIC_DEMO_MODE"


def _public_demo_mode_enabled() -> bool:
    """Public synthetic-demo mode is automatic when no real data_root/checkpoint is configured;
    AMD_PUBLIC_DEMO_MODE=false opts out (metadata-only behavior)."""
    return os.environ.get(PUBLIC_DEMO_ENV_VAR, "auto").strip().lower() not in ("false", "0", "no", "off")


def _gate_session_key(fingerprint: str) -> str:
    return f"review_gate::{fingerprint}"


def _get_or_create_gate(fingerprint: str) -> ReviewGate:
    key = _gate_session_key(fingerprint)
    if key not in st.session_state:
        st.session_state[key] = ReviewGate(fingerprint)
    return st.session_state[key]


_PROVIDER_LABELS = {"groq": "Groq", "amd_token_factory": "AMD Token Factory", "local": ""}


def render_workflow(
    result, *, key_prefix: str, extra_provenance: dict | None = None,
    quality_gate_status: QualityGateStatus = QualityGateStatus.NOT_ASSESSED,
    evidence_references: tuple = (), show_scores: bool = True, force_local_llm: bool = False,
) -> None:
    """Render scores -> evidence review -> human review gate -> optional LLM panel for one
    StudyInferenceResult. Used identically for the real bridged-inference result and the
    synthetic demonstration result below -- the only difference between the two call sites is
    which `result` object is passed in and `key_prefix` (so Streamlit's widget keys never
    collide when both sections are present on the same page). Nothing in this function knows
    or cares whether its input is real or synthetic; the distinction is made entirely by what
    `result.disclaimer`/`result.warnings` say, which this function always displays verbatim.
    """

    def k(name: str) -> str:
        return f"{key_prefix}_{name}"

    st.error(result.disclaimer)
    for warning in result.warnings:
        st.warning(warning)

    if show_scores:
        col_scores, col_provenance = st.columns([2, 1])
        with col_scores:
            st.markdown("**12 research-model output scores (raw, uncalibrated sigmoid outputs)**")
            st.bar_chart(result.probabilities)
            st.table({"target": list(result.probabilities), "score": [f"{v:.4f}" for v in result.probabilities.values()]})

        with col_provenance:
            st.markdown("**Checkpoint provenance**")
            provenance = result.model_provenance[0]
            st.json(provenance.to_dict())
            if extra_provenance is not None:
                st.markdown("**Bridge-specific metadata**")
                st.json(extra_provenance)
    else:
        st.caption("Synthetic placeholder inputs are intentionally not displayed as scores.")

    st.divider()
    st.subheader("Shared evidence review (mri_core.agents, unmodified)")
    workflow = process_inference_result(
        result, quality_gate_status=quality_gate_status, evidence_references=tuple(evidence_references),
    )

    col_specialist, col_critic, col_consensus = st.columns(3)
    with col_specialist:
        st.markdown("**Specialist summaries**")
        for summary in workflow.specialist_summaries:
            st.markdown(f"_{summary.group_name}_: {summary.note}")
            if show_scores:
                st.json(summary.scores)
    with col_critic:
        st.markdown("**Critic findings**")
        if workflow.critic_findings:
            for finding in workflow.critic_findings:
                st.text(f"[{finding.severity.value}] {finding.code}: {finding.message}")
        else:
            st.text("No findings.")
    with col_consensus:
        st.markdown("**Consensus**")
        st.write(f"workflow_status: `{workflow.consensus.workflow_status.value}`")
        for reason in workflow.consensus.reasons:
            st.text(f"- {reason}")
        st.markdown("_Required next steps:_")
        for step in workflow.consensus.required_next_steps:
            st.text(f"- {step}")

    st.divider()
    st.subheader("Human review gate (simulated — no clinician approval is implied)")
    fingerprint = workflow.bundle.fingerprint
    flash_key = k(f"review_flash::{fingerprint}")

    # Review actions run as on_click callbacks: Streamlit executes them BEFORE re-running the script,
    # so the state line, the buttons' disabled state and the audit table below are drawn from the
    # already-updated gate in the very same rerun. Only these human buttons ever call them.
    def _apply_review_action(label, action_fn):
        gate_ = _get_or_create_gate(fingerprint)
        actor_ = st.session_state.get(k("actor_input"), "demo_reviewer")
        reason_ = st.session_state.get(k("reason_input"), "")
        if not reason_.strip():
            st.session_state[flash_key] = ("error", "A reason is required before recording this action.")
            return
        try:
            action_fn(gate_, actor=actor_, reason=reason_)
            st.session_state[flash_key] = ("success", f"Recorded '{label}' by {actor_}.")
        except (InvalidTransitionError, StaleEvidenceError) as exc:
            st.session_state[flash_key] = ("error", str(exc))

    def _reset_review_state():
        st.session_state.pop(_gate_session_key(fingerprint), None)  # next render creates a fresh PENDING_REVIEW gate
        st.session_state.pop(flash_key, None)
        st.session_state[k("reason_input")] = ""
        st.session_state[k("actor_input")] = "demo_reviewer"

    gate = _get_or_create_gate(fingerprint)
    st.write(f"Current state: `{gate.state.value}`")

    st.text_input("Your name/role (actor)", value="demo_reviewer", key=k("actor_input"))
    st.text_area("Reason for this action", value="", key=k("reason_input"))

    action_cols = st.columns(3)
    locked = gate.state == ReviewState.REVIEWED
    with action_cols[0]:
        st.button("Approve", disabled=locked, key=k("approve_button"),
                  on_click=_apply_review_action, args=("approve", approve))
    with action_cols[1]:
        st.button("Request more evidence", disabled=locked, key=k("request_more_evidence_button"),
                  on_click=_apply_review_action, args=("request_more_evidence", request_more_evidence))
    with action_cols[2]:
        st.button("Escalate", disabled=locked, key=k("escalate_button"),
                  on_click=_apply_review_action, args=("escalate", escalate))

    flash = st.session_state.pop(flash_key, None)
    if flash is not None:
        (st.error if flash[0] == "error" else st.success)(flash[1])

    st.button(
        "Reset review state (demo)", key=k("reset_review_button"), on_click=_reset_review_state,
        help="Demo helper, user action only: restores this case to PENDING_REVIEW and clears the typed "
             "reason and this case's audit history. Nothing generated by an LLM can trigger it.",
    )
    st.markdown("**Audit history (append-only)**")
    if gate.history:
        rows = []
        for entry in gate.history:
            kind = "escalate" if is_escalation(entry) else entry.new_state.value.lower()
            rows.append({
                "actor": entry.actor, "action": kind, "from": entry.previous_state.value,
                "to": entry.new_state.value, "reason": entry.reason,
            })
        st.table(rows)
    else:
        st.caption("No review actions recorded yet.")

    st.divider()
    st.subheader("Optional LLM interpretation (AI-generated text, unverified — never auto-approved)")
    st.caption(
        "Untrusted free text from a chosen provider, run only through the shared, unmodified "
        "screen_proposal() gate (BLOCKED or QUARANTINED — there is no 'verified' outcome). This "
        "section is structurally separate from the raw scores, deterministic evidence, and review "
        "gate above: nothing here ever feeds into consensus, the review gate, or any approval "
        "decision. Only a strict allowlist (workflow status, grouped scores, finding codes, "
        "limitations) ever leaves this process for an external provider — never the study UID, "
        "checkpoint path, or any local file path."
    )

    llm_config = resolve_llm_config()
    # The provider the UI offers is exactly the one LLM_PROVIDER selects (never silently another one).
    intended_provider = "local" if force_local_llm else llm_config.provider
    provider_label = _PROVIDER_LABELS.get(intended_provider, "local deterministic")
    if intended_provider == "local":
        st.caption(
            "Local deterministic text only -- no external provider, API key or network. Set LLM_PROVIDER "
            "(groq or amd_token_factory) and LLM_ALLOW_EXTERNAL_CALLS=true to use a live provider. See .env.example."
        )
    elif intended_provider == "groq":
        st.caption(
            f"Configured via environment/.env — provider: `{llm_config.provider}`, "
            f"external calls allowed: `{llm_config.allow_external_calls}`, "
            f"GROQ_API_KEY present: `{bool(llm_config.groq_api_key)}`. "
            "A key being present is never enough on its "
            "own -- LLM_PROVIDER=groq and LLM_ALLOW_EXTERNAL_CALLS=true must also be set. See .env.example."
        )
    else:
        st.caption(
            f"Configured via environment/.env — provider: `{llm_config.provider}`, "
            f"external calls allowed: `{llm_config.allow_external_calls}`, "
            f"AMD_TOKENFACTORY_ENABLED: `{llm_config.amd_enabled}`, "
            f"AMD_TOKENFACTORY_API_KEY present: `{bool(llm_config.amd_api_key)}`, "
            f"base URL set: `{bool(llm_config.amd_base_url)}`, model set: `{bool(llm_config.amd_model)}`. "
            "A key being present is never enough on its own -- LLM_PROVIDER=amd_token_factory and "
            "LLM_ALLOW_EXTERNAL_CALLS=true must also be set. See .env.example."
        )

    button_label = (
        "Generate local deterministic explanation" if intended_provider == "local"
        else f"Generate live {provider_label} explanation"
    )
    if st.button(button_label, key=k("llm_generate_button")):
        if intended_provider == "local":
            primary, build_reason = LocalDeterministicAdapter(), "Local deterministic mode (no external provider)."
        else:
            primary, build_reason = build_adapter(llm_config)  # dispatches on llm_config.provider only
        st.caption(build_reason)

        proposal, provider_used = propose_with_fallback(primary, workflow.bundle, workflow.consensus)
        outcome = screen_proposal(proposal, workflow.bundle)
        st.session_state[k(f"llm_outcome::{workflow.bundle.fingerprint}")] = (provider_used, outcome)
        # The same allowlisted view of THIS run's deterministic review that a Groq request is built from
        # (redact_for_external_llm); kept only so the page can show what the request was based on.
        st.session_state[k(f"llm_meta::{workflow.bundle.fingerprint}")] = {
            "requested": intended_provider, "build_reason": build_reason,
            "model": getattr(primary, "model", None) if provider_used == intended_provider else None,
            "attempted": getattr(primary, "name", "") == intended_provider and intended_provider != "local",
            "payload": redact_for_external_llm(workflow.bundle, workflow.consensus),
        }

    llm_outcome_entry = st.session_state.get(k(f"llm_outcome::{workflow.bundle.fingerprint}"))
    if llm_outcome_entry is not None:
        provider_used, outcome = llm_outcome_entry
        if provider_used.startswith("local (fallback:"):
            # never echo a raw provider error (it can quote a response body) -- sanitized reason only
            st.caption(f"Produced by: local (fallback: {sanitize_fallback_reason(provider_used[len('local (fallback:'):].rstrip(') '))})")
        else:
            st.caption(f"Produced by: {provider_used}")
        if outcome.status == ProposalStatus.BLOCKED:
            st.error(f"BLOCKED — {outcome.reason}")
            st.caption(f"Rejected text (shown for audit only, not evidence): {outcome.raw_text!r}")
        else:
            # The internal state is still exactly ProposalStatus.QUARANTINED (unchanged) --
            # screen_proposal()'s own reason text is shown below, unedited, just de-emphasized
            # rather than leading. Nothing here implies the text is approved or verified; it is
            # presented as advisory only. outcome.raw_text is the canonical compact-JSON string
            # llm_providers._extract_structured_explanation() (or LocalDeterministicAdapter)
            # produced -- the four EXPLANATION_FIELDS, parsed here for display only. Parsing
            # never affects outcome.status/outcome.reason, which came from the unmodified,
            # shared screen_proposal() and are never recomputed from this parse.
            st.info("LLM interpretation generated — human review required")
            st.caption("Deterministic result → LLM explanation (below) → Human review")
            # Say plainly WHO produced this text, so a local fallback can never be mistaken for an
            # external provider's output. provider_used is exactly what propose_with_fallback()
            # reports: "<provider>" on success, "local (fallback: <reason>)" after a failure.
            meta = st.session_state.get(k(f"llm_meta::{workflow.bundle.fingerprint}"), {})
            requested_label = _PROVIDER_LABELS.get(meta.get("requested"), "")
            if provider_used in _PROVIDER_LABELS and provider_used != "local":
                model_name = meta.get("model")
                used_label = _PROVIDER_LABELS[provider_used]
                st.success(f"Live {used_label} explanation" + (f" — model: {model_name}" if model_name else ""))
                producer_sentence = (
                    f"The {used_label} response passed automated format and safety checks. Because this is "
                    "AI-generated text, it is"
                )
            elif requested_label:
                raw_reason = (
                    provider_used[len("local (fallback:"):].rstrip(") ")
                    if provider_used.startswith("local (fallback:") else meta.get("build_reason", "")
                )
                reason_text = sanitize_fallback_reason(raw_reason)
                if re.search(r"structured response|not valid JSON|message\.content is missing|exceeds", raw_reason):
                    st.warning(
                        f"Local deterministic fallback — {requested_label} response did not meet the structured-output requirements."
                    )
                else:
                    st.warning(f"Local deterministic fallback — {requested_label} did not produce this explanation.")
                with st.expander("Fallback details (sanitized)", expanded=True):
                    st.write(f"Reason: {reason_text}")
                producer_sentence = "This text was generated by the local deterministic template (no AI model) and is"
            else:
                st.info("Local deterministic explanation (no network, no AI model).")
                producer_sentence = "This text was generated by the local deterministic template and is"
            sent = meta.get("payload")
            if sent and meta.get("attempted"):
                with st.expander(f"Deterministic review fields this {requested_label or 'provider'} request was built from (allowlist only)"):
                    st.caption(
                        "Taken from the current run's Specialist/Critic/Consensus result. No MRI images, "
                        "files, study IDs, paths, model weights or secrets are included; score values are "
                        "not shown here."
                    )
                    st.json({
                        "workflow_status": sent["workflow_status"],
                        "specialist_groups": [g["group"] for g in sent["specialist_groups"]],
                        "critic_codes": sent["critic_codes"], "critic_severities": sent["critic_severities"],
                        "quality_gate_status": sent["quality_gate_status"], "is_ensemble": sent["is_ensemble"],
                        "model_count": sent["model_count"], "limitations": sent["limitations"],
                    })
            st.write(
                f"{producer_sentence} presented as advisory only and is not treated as verified "
                "evidence. The underlying MRI model scores, deterministic workflow status, and "
                "human-review gate remain unchanged."
            )
            st.caption(f"Status: {outcome.status.value} (advisory text; not verified evidence) — {outcome.reason}")

            explanation = None
            try:
                parsed = json.loads(outcome.raw_text)
                if isinstance(parsed, dict) and all(
                    isinstance(parsed.get(field_name), str) and parsed.get(field_name)
                    for field_name in ("specialist_summary", "critic_summary", "consensus_explanation", "reviewer_focus")
                ):
                    explanation = parsed
            except (ValueError, TypeError):
                explanation = None

            if explanation is not None:
                st.markdown("##### Specialist explanation")
                st.markdown(f"> {explanation['specialist_summary']}")
                st.markdown("##### Critic explanation")
                st.markdown(f"> {explanation['critic_summary']}")
                st.markdown("##### Consensus explanation")
                st.markdown(f"> {explanation['consensus_explanation']}")
                st.markdown("##### Reviewer focus")
                st.markdown(f"> {explanation['reviewer_focus']}")
            else:
                st.markdown(f"> {outcome.raw_text}")


# ------------------------------------------------------------------------------------- UI

st.title("MRI-Vision-Core — bridged Kaggle checkpoint, AMD-ready")
st.caption(
    "Research prototype. Not a diagnostic device. Runs entirely on local CPU by default; "
    "no network call or API credential is required."
)

data_paths_config = resolve_data_paths()

with st.sidebar:
    st.header("Data & checkpoint")
    data_root_input = st.text_input(
        "RSNA knee data_root",
        value=str(data_paths_config.data_root) if data_paths_config.data_root else "",
        key="data_root_input",
        help="A real local directory with train.csv and raw/train_series/... "
             "Prefilled from AMD_KAGGLE_RSNA_ROOT (environment or .env); works with both "
             "Windows paths (D:\\...) and Linux paths (/app/...). Override freely.",
    )
    checkpoint_input = st.text_input(
        "Kaggle checkpoint (.ckpt)",
        value=str(data_paths_config.checkpoint) if data_paths_config.checkpoint else "",
        key="checkpoint_input",
        help="A foldN-knee-mil-best.ckpt file. Prefilled from AMD_KAGGLE_CHECKPOINT "
             "(environment or .env). Override freely.",
    )


def render_public_demo() -> None:
    """Public synthetic-demo mode: synthetic cases only -- no MRI model, DICOM, checkpoint, API or
    network. Runs the unmodified deterministic specialist -> critic -> consensus -> human-gate flow."""
    st.warning(
        "**Synthetic workflow demonstration.** Every case below is invented: no real MRI study, "
        "patient, label, report or model prediction is involved, and the MRI model is never run."
    )
    st.caption("Research/demo only—not clinical advice or diagnosis.")
    try:
        case_set = load_public_demo_cases()
    except PublicDemoCasesError as exc:
        st.error(f"Could not load the public demonstration cases: {exc}")
        return
    st.caption(case_set.disclaimer)
    labels = [case_label(c) for c in case_set.cases]
    selected_label = st.selectbox("Select demonstration case", labels, key="public_demo_case_select")
    case = case_set.cases[labels.index(selected_label)]
    st.write(f"**Expected deterministic workflow status:** `{case.expected_workflow_status}`")
    st.caption(case.expected_reason)
    if st.button("Run demonstration workflow", type="primary", key="public_demo_run_button"):
        st.session_state["public_demo_case_id"] = case.case_id
    if st.session_state.get("public_demo_case_id") == case.case_id:
        result, quality, references = build_case_inputs(case)
        render_workflow(
            result, key_prefix="public", quality_gate_status=quality, evidence_references=references,
            show_scores=False,
        )


# Real inference needs BOTH paths set AND passing the same cheap structural checks used below.
# Blank or stale/invalid values (e.g. leftover /app/data/... Railway variables) therefore select
# the public synthetic mode, while valid local paths keep genuine MRI inference.
_real_inference_paths_usable = bool(data_root_input and checkpoint_input) and not validate_data_paths(
    Path(data_root_input), Path(checkpoint_input)
)
if _public_demo_mode_enabled() and not _real_inference_paths_usable:
    render_public_demo()
    st.stop()

st.divider()
# Only when real local inference is NOT available: with a real study and checkpoint this fabricated-
# numbers panel would be misleading, so it is not shown at all.
if not _real_inference_paths_usable:
    with st.expander(
        "🧪 Synthetic workflow demonstration — no real data, checkpoint, or inference required",
        expanded=not (data_root_input and checkpoint_input),
    ):
        st.warning(
            "This panel uses fixed, fabricated placeholder numbers to demonstrate the evidence-"
            "review / human-review / LLM-interpretation workflow when no real RSNA dataset or "
            "trained checkpoint is available (e.g. a fresh deployment, before real data has been "
            "authorized and bundled). No real DICOM study is read, no real checkpoint is loaded, "
            "and no real neural-network forward pass occurs here, ever."
        )
        if st.button("Run synthetic workflow demonstration", key="synthetic_demo_button"):
            st.session_state["synthetic_demo_result"] = build_synthetic_demo_result()
        if "synthetic_demo_result" in st.session_state:
            render_workflow(st.session_state["synthetic_demo_result"], key_prefix="synthetic")

st.divider()
st.subheader("Curated MRI Benchmark Scenarios")
st.caption("Research/demo benchmark only; not clinical accuracy or medical performance.")

# Set below, only when a selected scenario's study is confirmed locally available -- used to
# pre-select that study in the single "Run trained inference" selectbox/button further down the
# page, so there is exactly one inference button, not one here and a second, separate one below.
scenario_resolved_study_uid = None

try:
    scenario_set = load_scenarios()
except BenchmarkScenarioError as exc:
    st.error(f"Could not load benchmark scenarios: {exc}")
else:
    scenario_labels = build_labels(scenario_set)
    st.markdown("### 🧪 Select a curated MRI benchmark scenario")
    with st.container(key="benchmark_scenario_selector"):
        selected_scenario_label = st.selectbox(
            "Select a curated benchmark scenario", scenario_labels, key="benchmark_scenario_select",
            label_visibility="collapsed",
            help="22 real RSNA knee studies, each chosen for a specific presentation pattern. "
                 "Browsing this list never requires a configured data_root, checkpoint, API key, "
                 "or network connection -- it only reads benchmarks/amd/amd_demo_scenarios.json.",
        )
    selected_scenario = scenario_set.scenarios[scenario_labels.index(selected_scenario_label)]

    col_scenario_meta, col_scenario_labels = st.columns(2)
    with col_scenario_meta:
        st.write(f"**Scenario ID:** {selected_scenario['scenario_id']}")
        st.write(f"**Title:** {selected_scenario['title']}")
        st.write(f"**Scenario type:** {selected_scenario['scenario_type']}")
        st.write(f"**Why selected:** {selected_scenario['why_selected']}")
        if selected_scenario.get("presentation_tag"):
            st.write(f"**Presentation tag:** {selected_scenario['presentation_tag']}")
        st.write(f"**Tags:** {', '.join(selected_scenario.get('tags', [])) or 'none'}")
        st.write(f"**Series count:** {selected_scenario['series_count']}  ·  **DICOM count:** {selected_scenario['dicom_count']}")
        st.write(f"**Planes:** {', '.join(selected_scenario['planes'])}")
    with col_scenario_labels:
        st.markdown("**Competition reference labels** _(not a diagnosis)_")
        st.write(f"Positive target count: {selected_scenario['positive_target_count']}")
        st.write(f"Positive targets: {', '.join(selected_scenario['positive_targets']) or 'none'}")
        st.caption(
            "These are the RSNA competition's own supplied reference labels for this study, "
            "shown for context and later comparison only -- never a clinical finding, and "
            "never generated by this adapter's model."
        )
    st.caption(scenario_set.disclaimer)

    # Benchmark metadata selection is deliberately separate from local-study availability --
    # selecting a scenario above touches only the JSON file above; this check is the first
    # (and only) point where the configured data_root is consulted, and it never runs
    # inference, never fabricates a result, and never substitutes a different study.
    scenario_study_id = selected_scenario["study_id"]
    if not data_root_input:
        st.info(
            "Benchmark metadata loaded. No data_root is configured, so local availability for "
            "this scenario's study cannot be checked yet."
        )
    else:
        try:
            scenario_available_studies = _list_available_studies(Path(data_root_input))
            scenario_study_available = scenario_study_id in scenario_available_studies
        except Exception as exc:
            st.info(f"Benchmark metadata loaded. Could not check local availability: {exc}")
        else:
            if not scenario_study_available:
                st.warning(
                    "Benchmark metadata loaded. The underlying MRI study is not available on "
                    "this machine, so live inference cannot be run for this scenario."
                )
            else:
                st.success(
                    f"The underlying MRI study for {selected_scenario['scenario_id']} is "
                    "available locally under the configured data_root."
                )
                # No separate button here -- this scenario's study is passed as the default
                # selection to the single "Run trained inference" button further down the page
                # (see scenario_resolved_study_uid), so there is exactly one inference button.
                scenario_resolved_study_uid = scenario_study_id
                st.caption(
                    f"Scenario {selected_scenario['scenario_id']}'s study is pre-selected below "
                    "in 'Run trained inference on selected MRI study'."
                )
                if not checkpoint_input:
                    st.caption("Configure a checkpoint in the sidebar to enable running real inference on this scenario.")

if not data_root_input or not checkpoint_input:
    st.info(
        "Enter a real data_root and checkpoint path in the sidebar to run real inference, or "
        "use the synthetic workflow demonstration above."
    )
    st.stop()

data_root = Path(data_root_input)
checkpoint_path = Path(checkpoint_input)

# Cheap, structural checks only (existence, extension, one small metadata file's presence) --
# never loads the dataset or the checkpoint's weights just to validate the sidebar inputs.
path_problems = validate_data_paths(data_root, checkpoint_path)
if path_problems:
    for problem in path_problems:
        st.error(problem)
    st.stop()

try:
    available_studies = _list_available_studies(data_root)
except Exception as exc:
    st.error(f"Could not discover studies under {data_root}: {exc}")
    st.stop()

if not available_studies:
    st.warning(f"No fully downloaded labeled study found under {data_root}.")
    st.stop()

# Pre-select the curated scenario's study (if one was chosen above and confirmed locally
# available) so the single button below acts on it without the user having to find it again in
# this list; only affects the selectbox's initial default -- the user can still pick any other
# available study here, same as before this scenario selector existed.
default_study_index = (
    available_studies.index(scenario_resolved_study_uid)
    if scenario_resolved_study_uid in available_studies else 0
)
study_uid = st.selectbox(
    "Available MRI study (StudyInstanceUID)", available_studies, index=default_study_index, key="study_select",
)
run_clicked = st.button(
    "Run trained inference on selected MRI study", type="primary", key="run_inference_button",
)

result_key = "last_result"
if run_clicked:
    if not checkpoint_path.is_file():
        st.error(f"No checkpoint file at {checkpoint_path}.")
        st.stop()
    try:
        resolved_uid, series_uid, plane, result, model_provenance = _run_inference(
            data_root, checkpoint_path, study_uid
        )
    except BridgeError as exc:
        st.error(f"Inference failed: {exc}")
        st.stop()
    st.session_state[result_key] = {
        "study_uid": resolved_uid, "series_uid": series_uid, "plane": plane,
        "result": result, "model_provenance": model_provenance,
    }

if result_key not in st.session_state:
    st.stop()

state = st.session_state[result_key]
result = state["result"]

st.divider()
st.subheader(f"Study {state['study_uid']}  ·  series {state['series_uid']}  ·  plane {state['plane']}")
st.info(
    "**Local trained inference** — a real, locally available MRI study scored on this machine by the "
    "bridged checkpoint. Raw research scores only; not a diagnosis."
)
render_workflow(result, key_prefix="real", extra_provenance=state["model_provenance"])

st.divider()
st.subheader("CPU benchmark evidence")
benchmark_path = Path(st.text_input(
    "Benchmark result file", value="outputs/amd/benchmark_kaggle_bridge.json", key="benchmark_path_input",
    help="Produced by: python -m hackathon.amd.benchmark_kaggle_bridge ...",
))
if benchmark_path.is_file():
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cpu_b = next(iter(benchmark["cpu"]["batches"].values()))
    st.write(f"CPU median latency: **{cpu_b['latency']['median_ms']:.1f} ms**, "
             f"throughput: **{cpu_b['throughput_studies_per_s']:.2f} studies/s**")
    gpu = benchmark["gpu"]
    if gpu.get("status") == "unavailable":
        st.warning("AMD GPU benchmark: **not available** — " + "; ".join(gpu.get("reasons", [])))
    else:
        st.success(
            f"AMD GPU benchmark (is_amd_rocm={gpu.get('is_amd_rocm')}): "
            f"{benchmark['comparison'].get('gpu_speedup_vs_cpu', 'n/a')}x vs CPU median."
        )
    st.caption(f"checkpoint_sha256: {benchmark['workload']['checkpoint_sha256']}")
else:
    st.caption(
        f"No benchmark file found at {benchmark_path}. Run "
        "`python -m hackathon.amd.benchmark_kaggle_bridge --data-root ... --checkpoint ...` first."
    )
    st.info("AMD GPU benchmark: not available until measured on a real ROCm environment.")
