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
from hackathon.amd.data_config import resolve_data_paths, validate_data_paths
from hackathon.amd.kaggle_mil_bridge import BridgeError, load_bridge_checkpoint, run_study_inference
from hackathon.amd.llm_config import build_adapter, resolve_llm_config
from hackathon.amd.llm_providers import LocalDeterministicAdapter, propose_with_fallback
from hackathon.amd.providers import AMDTokenFactoryProvider, LocalProvider
from hackathon.amd.review_actions import approve, escalate, is_escalation, request_more_evidence
from hackathon.amd.run_real_checkpoint_smoke import NON_DIAGNOSTIC_NOTICE, pick_study_and_series, sha256_of_file
from hackathon.amd.synthetic_demo import build_synthetic_demo_result
from mri_core.agents.contracts import QualityGateStatus, ReviewState
from mri_core.agents.llm_adapter import ProposalStatus, screen_proposal
from mri_core.agents.orchestrator import process_inference_result
from mri_core.agents.review_gate import InvalidTransitionError, ReviewGate, StaleEvidenceError

st.set_page_config(page_title="MRI-Vision-Core — AMD bridged demo", layout="wide")


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


def _gate_session_key(fingerprint: str) -> str:
    return f"review_gate::{fingerprint}"


def _get_or_create_gate(fingerprint: str) -> ReviewGate:
    key = _gate_session_key(fingerprint)
    if key not in st.session_state:
        st.session_state[key] = ReviewGate(fingerprint)
    return st.session_state[key]


def render_workflow(result, *, key_prefix: str, extra_provenance: dict | None = None) -> None:
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

    st.divider()
    st.subheader("Shared evidence review (mri_core.agents, unmodified)")
    workflow = process_inference_result(result, quality_gate_status=QualityGateStatus.NOT_ASSESSED)

    col_specialist, col_critic, col_consensus = st.columns(3)
    with col_specialist:
        st.markdown("**Specialist summaries**")
        for summary in workflow.specialist_summaries:
            st.markdown(f"_{summary.group_name}_: {summary.note}")
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
    gate = _get_or_create_gate(workflow.bundle.fingerprint)
    st.write(f"Current state: `{gate.state.value}`")

    actor = st.text_input("Your name/role (actor)", value="demo_reviewer", key=k("actor_input"))
    reason = st.text_area("Reason for this action", value="", key=k("reason_input"))

    action_cols = st.columns(3)
    action_taken = None
    with action_cols[0]:
        if st.button("Approve", disabled=gate.state == ReviewState.REVIEWED, key=k("approve_button")):
            action_taken = ("approve", approve)
    with action_cols[1]:
        if st.button("Request more evidence", disabled=gate.state == ReviewState.REVIEWED, key=k("request_more_evidence_button")):
            action_taken = ("request_more_evidence", request_more_evidence)
    with action_cols[2]:
        if st.button("Escalate", disabled=gate.state == ReviewState.REVIEWED, key=k("escalate_button")):
            action_taken = ("escalate", escalate)

    if action_taken is not None:
        label, action_fn = action_taken
        if not reason.strip():
            st.error("A reason is required before recording this action.")
        else:
            try:
                action_fn(gate, actor=actor, reason=reason)
                st.success(f"Recorded '{label}' by {actor}.")
            except (InvalidTransitionError, StaleEvidenceError) as exc:
                st.error(str(exc))

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
    st.caption(
        f"Configured via environment/.env — provider: `{llm_config.provider}`, "
        f"external calls allowed: `{llm_config.allow_external_calls}`, "
        f"GROQ_API_KEY present: `{bool(llm_config.groq_api_key)}`. "
        "A credential being present is never enough on its own — LLM_ALLOW_EXTERNAL_CALLS must "
        "also be explicitly true. See .env.example."
    )

    llm_provider_choice = st.radio(
        "Try LLM text provider",
        ["Local (default, no network)", "Groq (requires GROQ_API_KEY)", "AMD Token Factory (LLM reasoning, requires .env config)"],
        key=k("llm_provider_choice"),
        help="This only has an effect if it matches the environment-configured provider above "
             "AND external calls are allowed there; otherwise it always falls back to local. "
             "This AMD option is the implemented, live-verified LLM-reasoning path -- unrelated "
             "to the 'Execution provider' AMD checkbox in the sidebar (hardware, not implemented).",
    )
    intended_provider = (
        "groq" if llm_provider_choice.startswith("Groq")
        else "amd_token_factory" if llm_provider_choice.startswith("AMD")
        else "local"
    )

    if st.button("Generate LLM interpretation", key=k("llm_generate_button")):
        if intended_provider == llm_config.provider:
            primary, build_reason = build_adapter(llm_config)
        else:
            primary, build_reason = LocalDeterministicAdapter(), (
                f"UI selected {intended_provider!r} but the environment-configured LLM_PROVIDER is "
                f"{llm_config.provider!r}; using local deterministic mode."
            )
        st.caption(build_reason)

        proposal, provider_used = propose_with_fallback(primary, workflow.bundle, workflow.consensus)
        outcome = screen_proposal(proposal, workflow.bundle)
        st.session_state[k(f"llm_outcome::{workflow.bundle.fingerprint}")] = (provider_used, outcome)

    llm_outcome_entry = st.session_state.get(k(f"llm_outcome::{workflow.bundle.fingerprint}"))
    if llm_outcome_entry is not None:
        provider_used, outcome = llm_outcome_entry
        st.caption(f"Produced by: {provider_used}")
        if outcome.status == ProposalStatus.BLOCKED:
            st.error(f"BLOCKED — {outcome.reason}")
            st.caption(f"Rejected text (shown for audit only, not evidence): {outcome.raw_text!r}")
        else:
            st.warning(f"QUARANTINED (unverified) — {outcome.reason}")
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

    st.header("Execution provider (hardware, not AI text)")
    st.caption(
        "Where the *model forward pass* above would run. This is separate from the "
        "'Optional LLM interpretation' section further down the page, which is about AI-"
        "generated text, not hardware."
    )
    local_status = LocalProvider().status()
    st.write(f"**Local CPU (default):** {local_status.detail}")
    amd_enabled = st.checkbox(
        "Enable AMD Token Factory hardware execution (optional, not implemented)",
        value=False, key="amd_enabled_checkbox",
        help="Status/UI seam only -- there is no real client behind this yet, and nothing "
             "here ever dispatches the model to AMD hardware. Not the same as the AMD Token "
             "Factory *LLM* option below, which is implemented and live-verified.",
    )
    if amd_enabled:
        base_url = st.text_input("Token Factory base URL", value=os.environ.get("AMD_TOKENFACTORY_BASE_URL", ""))
        api_key = st.text_input("Token Factory API key", value="", type="password")
        amd_status = AMDTokenFactoryProvider(enabled=True, base_url=base_url or None, api_key=api_key or None).status()
        st.warning(amd_status.detail)
    else:
        st.caption("AMD Token Factory hardware execution: disabled by default; no network call is made.")

st.divider()
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

study_uid = st.selectbox("Available MRI study (StudyInstanceUID)", available_studies, key="study_select")

run_clicked = st.button("Run trained inference on this study", type="primary", key="run_inference_button")

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
