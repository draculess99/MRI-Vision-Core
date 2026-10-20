"""Provider-neutral `mri_core.agents.llm_adapter.LLMAdapter` implementations.

Reuses the shared contract unchanged -- nothing here modifies `mri_core/agents/llm_adapter.py`:
  - `LLMAdapter` (Protocol): `propose(bundle, consensus) -> RawProposal`.
  - `RawProposal`: untrusted free text + claimed citations; never evidence on its own.
  - `screen_proposal(proposal, bundle) -> QuarantinedProposal`: the only gate between untrusted
    text and anything rendered (BLOCKED or QUARANTINED -- there is no "verified" outcome).
  - `render_reviewer_brief(bundle, consensus) -> ReviewerBrief`: the only trusted output type,
    template-only, never touched by anything in this module.

Three implementations:
  - `LocalDeterministicAdapter` -- always available, no network, no credentials. Mirrors the
    shared `OfflineStubAdapter`'s own safety pattern (deterministic template text).
  - `GroqAdapter` -- real network calls to Groq's OpenAI-compatible chat-completions endpoint,
    gated on `GROQ_API_KEY` being set. Every request body is built exclusively from
    `redact_for_external_llm()`'s strict allowlist (see below) -- never from
    `bundle.to_dict()` or any object carrying `study_uid`/`checkpoint_path`/local paths.
  - `AMDTokenFactoryLLMAdapter` -- real network calls to AMD Token Factory's chat-completions
    endpoint, against a contract verified by exactly one live, synthetic, non-medical smoke
    test (see README). Disabled unless `enabled`/`base_url`/`api_key`/`model` are all set, in
    addition to `llm_config.build_adapter()`'s own LLM_PROVIDER/LLM_ALLOW_EXTERNAL_CALLS gate.
    Same redaction, reasoning-field, and strict-schema safety properties as `GroqAdapter`.

Data-leakage boundary: `redact_for_external_llm()` builds a brand-new dict field by field from a
fixed allowlist -- it never copies an existing object and strips keys, so there is no code path
by which an unlisted field (study_uid, checkpoint_path, checkpoint_sha256, data_root, series_uid,
a critic finding's `evidence_reference`, or any local file path) can reach it. Every adapter in
this module (except LocalDeterministicAdapter, which makes no network call at all) builds its
request body exclusively from this function's output.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import requests

from mri_core.agents.consensus import ConsensusResult
from mri_core.agents.contracts import EvidenceBundle
from mri_core.agents.llm_adapter import MAX_PROPOSAL_LENGTH, RawProposal

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
# Default: a model that can run with reasoning OFF. Live diagnosis on the real-study path showed
# openai/gpt-oss-20b (reasoning cannot be disabled, only set to "low") spending 1,800-3,300 hidden tokens
# per answer: it routinely blew the 60-character field caps, tripped Groq's 8,000-tokens-per-minute limit
# (the reserved max_tokens counts) on the repair request, and intermittently returned an empty generation.
# qwen/qwen3.8-27b with reasoning_effort="none" answered 3/3 on the first request in ~1.5s using ~100
# completion tokens, every field within its cap. (llama-3.3-70b-versatile 404'd for a real key tested here.)
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
DEFAULT_TIMEOUT_SECONDS = 10.0

# openai/gpt-oss-20b is a reasoning model: hidden reasoning tokens and the final JSON share this one
# completion budget. Live diagnosis of an HTTP 400 {"code": "json_validate_failed"} with an EMPTY
# failed_generation showed the former 1200-token cap being exhausted by reasoning before any JSON was
# emitted; the identical request succeeded (finish_reason "stop") at 2500 and 4000, using roughly
# 400-1500 reasoning tokens plus ~120 tokens of JSON. This is only a ceiling -- unused budget is not spent.
GROQ_MAX_TOKENS = 4000
# Non-reasoning models (reasoning_effort "none") emit only the ~100-token JSON, so a small ceiling is
# enough -- and it matters: Groq counts the reserved max_tokens against the 8,000 tokens/minute limit, so a
# first request plus a repair request must not each reserve thousands of tokens.
GROQ_MAX_TOKENS_NO_REASONING = 1000


def groq_request_tuning(model: str) -> tuple:
    """(reasoning_effort, max_tokens) for a Groq model. Qwen models accept reasoning_effort="none" and
    then need only a small budget; every other model (the gpt-oss family, which cannot disable reasoning)
    keeps "low" and the large budget shared with its hidden reasoning tokens."""
    if str(model).lower().startswith("qwen/"):
        return "none", GROQ_MAX_TOKENS_NO_REASONING
    return "low", GROQ_MAX_TOKENS

# Every key an external provider is ever allowed to see. redact_for_external_llm() builds its
# output strictly from this set; nothing else is ever added.
_ALLOWED_KEYS = frozenset(
    {
        "workflow_status", "specialist_groups", "critic_codes", "critic_severities",
        "quality_gate_status", "is_ensemble", "model_count", "limitations",
    }
)

# The explanation layer's one and only job: restate WHY the deterministic Specialist/Critic/
# Consensus pipeline reached the result it already reached -- never to re-decide it. Every field
# below must be derivable from the redact_for_external_llm() payload alone (workflow_status,
# specialist_groups, critic_codes/critic_severities, quality_gate_status, is_ensemble,
# model_count, limitations); nothing here changes what is sent, only what is asked for in return.
_EXPLANATION_FIELDS = ("specialist_summary", "critic_summary", "consensus_explanation", "reviewer_focus")

# Per-field character budgets, tight enough that the combined canonical JSON still fits under
# mri_core.agents.llm_adapter.MAX_PROPOSAL_LENGTH (500) -- the same shared, unmodified cap
# screen_proposal() already enforces on the combined raw_text. Four short fields sharing one
# 500-char budget is tighter than the single-sentence design this replaces; that is intentional,
# not a relaxation of any limit. critic_summary is wider than the others: a live AMD Token
# Factory response naming all three findings in the common demo case (QUALITY_NOT_ASSESSED,
# ENGINE_WARNINGS_PRESENT, NO_EVIDENCE_REFERENCES) needed 107 characters.
_FIELD_MAX_LENGTHS = {
    "specialist_summary": 60,
    "critic_summary": 135,
    "consensus_explanation": 110,
    "reviewer_focus": 80,
}

# Targets sit safely below the hard caps so a slightly long answer still fits.
_FIELD_TARGET_LENGTHS = {
    "specialist_summary": 50, "critic_summary": 120, "consensus_explanation": 100, "reviewer_focus": 70,
}

_SYSTEM_PROMPT = (
    "You are a research-workflow assistant for a non-diagnostic MRI research prototype. You "
    "will be given deterministic, de-identified workflow evidence already produced by fixed, "
    "rule-based Specialist/Critic/Consensus logic: workflow_status, specialist_groups (grouped "
    "research model scores), critic_codes/critic_severities (structural findings), "
    "quality_gate_status, is_ensemble, model_count, and limitations. No patient identifier, "
    "study identifier, file path, or image data is present or available to you.\n\n"
    "Your only job is to EXPLAIN this already-decided deterministic result in plain language -- "
    "never to re-decide, override, or second-guess it. Respond with a JSON object with exactly "
    "these four required string fields, each ONE short sentence, derived ONLY from the fields "
    "given above (never invent a finding, score, or next step not implied by them):\n"
    f"  specialist_summary (<= {_FIELD_MAX_LENGTHS['specialist_summary']} chars): what the "
    "Specialist organized from the given specialist_groups, preserving scores verbatim.\n"
    f"  critic_summary (<= {_FIELD_MAX_LENGTHS['critic_summary']} chars): the actual critic_codes "
    "/critic_severities given, in plain language -- name the real issues, invent no others.\n"
    f"  consensus_explanation (<= {_FIELD_MAX_LENGTHS['consensus_explanation']} chars): why those "
    "critic findings plus quality_gate_status/inference state produced exactly the given "
    "workflow_status -- connect cause to effect explicitly.\n"
    f"  reviewer_focus (<= {_FIELD_MAX_LENGTHS['reviewer_focus']} chars): what a human reviewer "
    "should check or complete next, based only on the given findings.\n\n"
    "Never state or imply a diagnosis, a clinical finding, certainty, or that any clinician "
    "approved or reviewed anything. Never invent facts not present in the data you were given. "
    "Include no field other than these four."
)

_LIMITS_TEXT = "; ".join(f"{f} <= {_FIELD_MAX_LENGTHS[f]}" for f in _EXPLANATION_FIELDS)
_TARGETS_TEXT = ", ".join(f"{f} about {_FIELD_TARGET_LENGTHS[f]}" for f in _EXPLANATION_FIELDS)

# Groq-only additions (the shared prompt above is unchanged): output format and length discipline.
_GROQ_STRICT_OUTPUT_RULES = (
    "\n\nOUTPUT FORMAT -- STRICT: reply with ONE JSON object and nothing else: no markdown, no code "
    "fences, no commentary before or after it. HARD CHARACTER LIMITS (counted exactly, spaces included): "
    f"{_LIMITS_TEXT}. Aim safely BELOW every limit -- {_TARGETS_TEXT} characters -- to leave margin. "
    "If ANY field is over its limit the whole response is unusable and will be discarded."
)
_GROQ_SYSTEM_PROMPT = _SYSTEM_PROMPT + _GROQ_STRICT_OUTPUT_RULES

# Structured-output schema for Groq's response_format: four required string fields. Using
# json_schema (strict mode) rather than freeform text is what makes the *final* answer land in
# message.content reliably for reasoning models like openai/gpt-oss-20b -- without it, a
# reasoning model can spend its whole token budget on hidden reasoning and leave message.content
# empty (see hackathon/amd/README.md's Groq smoke-test notes for the real failure this fixes).
_REVIEWER_RESULT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "reviewer_explanation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {field: {"type": "string"} for field in _EXPLANATION_FIELDS},
            "required": list(_EXPLANATION_FIELDS),
            "additionalProperties": False,
        },
    },
}


class LLMProviderError(Exception):
    """Base class for provider-layer failures (as opposed to screen_proposal's own,
    separate classification of the *content* of a successfully returned proposal)."""


class LLMProviderUnavailableError(LLMProviderError):
    """Raised when a provider cannot produce a RawProposal at all: missing credentials,
    network failure, timeout, a non-200 response, or a response that does not match the
    expected schema. Callers should catch this and fall back to another provider -- it is
    never raised for "the text said something unsafe," which is screen_proposal's job."""


class EmptyGenerationError(LLMProviderUnavailableError):
    """Groq HTTP 400 {"code": "json_validate_failed"} whose `failed_generation` is EMPTY: the model
    emitted nothing at all, so there is no invalid JSON -- a transient provider-side failure observed
    intermittently (often on the first call after idle, taking ~5s before failing) with identical,
    otherwise-working requests. The only failure GroqAdapter._chat will resubmit, once."""


class FieldLengthError(LLMProviderUnavailableError):
    """The response was valid JSON with exactly the four required string fields, and ONLY a
    per-field length cap was exceeded. The one failure a single repair retry is allowed to address;
    it carries the parsed fields so the retry can ask for just those to be shortened."""

    def __init__(self, message: str, structured: dict, offending_fields: tuple):
        super().__init__(message)
        self.structured = structured
        self.offending_fields = offending_fields


@dataclass(frozen=True)
class LLMProviderStatus:
    name: str
    available: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "available": self.available, "detail": self.detail}


def redact_for_external_llm(bundle: EvidenceBundle, consensus: ConsensusResult) -> dict:
    """The ONLY thing ever sent to an external LLM provider's request body.

    Strict allowlist, built field by field -- never `bundle.to_dict()` with keys removed, so
    there is no path by which a future field added to EvidenceBundle/ConsensusResult leaks here
    by default. Excludes (always, structurally): study_uid, checkpoint_provenance (checkpoint
    path/SHA-256/fold), evidence_references, disclaimer text, latency, and every critic
    finding's free-form `message`/`evidence_reference` (kept to `code` and `severity` only).
    """
    payload = {
        "workflow_status": consensus.workflow_status.value,
        "specialist_groups": [
            {"group": s.group_name, "scores": {k: round(float(v), 4) for k, v in s.scores.items()}}
            for s in consensus.specialist_summaries
        ],
        "critic_codes": [f.code for f in consensus.critic_findings],
        "critic_severities": [f.severity.value for f in consensus.critic_findings],
        "quality_gate_status": bundle.quality_gate_status.value,
        "is_ensemble": bool(bundle.is_ensemble),
        "model_count": int(bundle.model_count),
        "limitations": list(bundle.limitations),
    }
    assert set(payload) <= _ALLOWED_KEYS, f"redact_for_external_llm produced an unlisted key: {set(payload) - _ALLOWED_KEYS!r}"
    return payload


_MIN_FORBIDDEN_VALUE_LENGTH = 6  # below this, a substring match is noise, not a real identifier leak


def _assert_no_leakage(payload: dict, bundle: EvidenceBundle) -> None:
    """Defensive, belt-and-suspenders check run before every outbound request: the redacted
    payload must not literally contain the study_uid or any checkpoint path substring, even if
    a future bug added them under an allowed key's value by mistake.

    Only values at least _MIN_FORBIDDEN_VALUE_LENGTH characters long are checked: real study
    UIDs and file paths are always far longer than that, and checking shorter strings produces
    false positives against ordinary English text (e.g. a one-character study_uid would match
    nearly any sentence) instead of catching a real leak.
    """
    serialized = json.dumps(payload)
    forbidden = [str(bundle.study_uid)] + [
        str(p.get("checkpoint_path", "")) for p in bundle.checkpoint_provenance if p.get("checkpoint_path")
    ]
    for value in forbidden:
        if value and len(value) >= _MIN_FORBIDDEN_VALUE_LENGTH and value in serialized:
            raise LLMProviderError(f"Refusing to send payload: it contains a forbidden value ({value!r}).")


def _truncate(text: str, max_length: int) -> str:
    """Used only by LocalDeterministicAdapter to keep its deterministic text within the same
    per-field budgets the network providers are asked to respect -- never used to validate or
    silently accept an oversized network response (see _extract_structured_explanation, which
    rejects those outright instead)."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


class LocalDeterministicAdapter:
    """Always available. No network call, no credential, no code path that could leak
    anything -- it only restates fields already present on bundle/consensus, same pattern as
    the shared OfflineStubAdapter.

    Produces the same four-field structured explanation shape as the network-backed providers
    (see _EXPLANATION_FIELDS), so the Streamlit UI's rendering is identical regardless of which
    provider actually ran. Unlike the network providers, nothing here is generated by a model --
    `consensus.reasons` and `consensus.required_next_steps` already ARE the deterministic
    Consensus layer's own explanation of its own decision (see mri_core/agents/consensus.py), so
    this adapter only reformats them; it never infers or invents anything beyond what those two
    fields (plus the specialist/critic summaries) already state.
    """

    name = "local"

    def status(self) -> LLMProviderStatus:
        return LLMProviderStatus(self.name, True, "Always available; no network call, no credential.")

    def propose(self, bundle: EvidenceBundle, consensus: ConsensusResult) -> RawProposal:
        specialist_summary = (
            f"Specialist grouped 12 scores into {len(consensus.specialist_summaries)} "
            "categories, unchanged."
        )

        if consensus.critic_findings:
            codes = ", ".join(f.code for f in consensus.critic_findings)
            critic_summary = f"Critic identified {len(consensus.critic_findings)} issue(s): {codes}."
        else:
            critic_summary = "Critic identified no evidence issues."

        consensus_explanation = (
            f"Consensus assigned {consensus.workflow_status.value}: " + "; ".join(consensus.reasons)
        )

        reviewer_focus = "; ".join(consensus.required_next_steps) or "No further action specified."

        structured = {
            "specialist_summary": _truncate(specialist_summary, _FIELD_MAX_LENGTHS["specialist_summary"]),
            "critic_summary": _truncate(critic_summary, _FIELD_MAX_LENGTHS["critic_summary"]),
            "consensus_explanation": _truncate(consensus_explanation, _FIELD_MAX_LENGTHS["consensus_explanation"]),
            "reviewer_focus": _truncate(reviewer_focus, _FIELD_MAX_LENGTHS["reviewer_focus"]),
        }
        raw_text = json.dumps(structured, separators=(",", ":"))
        return RawProposal(raw_text=raw_text, cited_evidence=bundle.evidence_references)


class GroqAdapter:
    """Real network calls to Groq's OpenAI-compatible chat-completions API.

    Unavailable (raises LLMProviderUnavailableError from propose(), reports unavailable from
    status()) whenever GROQ_API_KEY is not set -- no network call is attempted in that case.
    cited_evidence is always `()`: this adapter never asks the model to cite anything, so there
    is no fabricated citation for screen_proposal to even need to catch (it would anyway, since
    any cited_evidence is checked against bundle.evidence_references, but this adapter does not
    rely on that -- it simply never produces a citation).
    """

    name = "groq"

    def __init__(self, *, api_key: Optional[str] = None, model: Optional[str] = None,
                timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.timeout = timeout

    def status(self) -> LLMProviderStatus:
        if not self.api_key:
            return LLMProviderStatus(self.name, False, "GROQ_API_KEY not set; no network call attempted.")
        return LLMProviderStatus(self.name, True, f"Configured (model={self.model}). Not verified until a call succeeds.")

    def propose(self, bundle: EvidenceBundle, consensus: ConsensusResult) -> RawProposal:
        if not self.api_key:
            raise LLMProviderUnavailableError("GROQ_API_KEY not set; Groq provider unavailable.")

        payload = redact_for_external_llm(bundle, consensus)
        _assert_no_leakage(payload, bundle)

        data = self._chat([
            {"role": "system", "content": _GROQ_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ])
        try:
            text = _extract_groq_structured_explanation(data)
        except FieldLengthError as too_long:
            # Exactly ONE repair attempt, and only because the JSON was otherwise valid. Network errors,
            # credentials, invalid JSON, wrong keys and empty fields are never retried, and the repaired
            # response goes through the very same validation (and, afterwards, the same screening).
            over = ", ".join(too_long.offending_fields)
            limits = {f: _FIELD_MAX_LENGTHS[f] for f in too_long.offending_fields}
            repair_request = (
                f"Your previous JSON had field(s) over the character limit: {over} (limits {limits}). Shorten ONLY "
                "those field(s), keeping the same meaning, to safely below their limit; leave the other fields "
                "exactly as they are. Reply with the same four-field JSON object only -- no markdown, no prose.\n"
                + json.dumps(too_long.structured)
            )
            repaired = self._chat([
                {"role": "system", "content": _GROQ_SYSTEM_PROMPT},
                {"role": "user", "content": repair_request},
            ])
            text = _extract_groq_structured_explanation(repaired)
        return RawProposal(raw_text=text, cited_evidence=())

    def _chat(self, messages: list) -> Any:
        """One Groq chat-completions call (used for the first and the single length-repair request).
        Returns the decoded JSON body or raises LLMProviderUnavailableError.

        Exactly one resubmit, and ONLY for an EmptyGenerationError (HTTP 400 json_validate_failed with an
        empty failed_generation). Everything else -- timeouts, connection errors, 401/403/429/5xx, any other
        400, a 400 that did generate (invalid JSON) -- is raised immediately and never retried here."""
        try:
            return self._chat_once(messages)
        except EmptyGenerationError:
            return self._chat_once(messages)

    def _chat_once(self, messages: list) -> Any:
        reasoning_effort, max_tokens = groq_request_tuning(self.model)
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": _REVIEWER_RESULT_RESPONSE_FORMAT,
            "reasoning_effort": reasoning_effort,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            response = requests.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=body, timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise LLMProviderUnavailableError(f"Groq request timed out after {self.timeout}s.") from exc
        except requests.RequestException as exc:
            raise LLMProviderUnavailableError(f"Groq request failed: {exc}") from exc

        if response.status_code != 200:
            # Include Groq's short error CODE (e.g. json_validate_failed) so a failure is diagnosable from
            # the UI; never the message body, which may echo generated text.
            code, empty_generation = "", False
            try:
                error = response.json().get("error") or {}
                raw_code = error.get("code")
                if isinstance(raw_code, str) and raw_code.replace("_", "").isalnum() and len(raw_code) <= 60:
                    code = f" ({raw_code})"
                empty_generation = (
                    response.status_code == 400 and raw_code == "json_validate_failed"
                    and error.get("failed_generation") == ""
                )
            except (ValueError, AttributeError):
                pass
            error_class = EmptyGenerationError if empty_generation else LLMProviderUnavailableError
            raise error_class(f"Groq returned HTTP {response.status_code}{code}: {response.text[:200]!r}")

        try:
            return response.json()
        except ValueError as exc:
            raise LLMProviderUnavailableError("Groq response was not valid JSON.") from exc


_MAX_COMBINED_LENGTH = MAX_PROPOSAL_LENGTH  # the same 500-char cap screen_proposal() itself
# enforces on the combined raw_text; exceeding it is rejected here first, with a specific error,
# rather than reaching screen_proposal() and coming back only as an opaque BLOCKED. This is a
# second check at the SAME threshold, not a looser one -- it never permits anything
# screen_proposal() would reject, only fails earlier/clearer.


def _extract_structured_explanation(data: Any, *, provider_label: str) -> str:
    """Strict schema check shared by every network-backed adapter (Groq, AMD Token Factory),
    in two layers -- both rejecting outright rather than guessing. Returns the canonical,
    compact-JSON `raw_text` for RawProposal: `{"specialist_summary":...,"critic_summary":...,
    "consensus_explanation":...,"reviewer_focus":...}`, re-serialized from the validated fields
    (never the model's own, possibly differently-whitespaced JSON text) so length checks and
    downstream UI parsing are both exact and deterministic.

    1. The outer HTTP response: only `choices[0].message.content` is ever read. A reasoning
       model's `reasoning` (or any `reasoning_content`-style) field is NEVER read as a
       fallback or substitute for `content` -- that field holds the model's hidden reasoning
       trace, not its reviewed answer, and treating it as the answer would silently accept
       unreviewed, unstructured text that bypassed the schema this function enforces.
    2. The inner structured payload: `content` must itself be valid JSON -- a dict with
       exactly the four required _EXPLANATION_FIELDS keys (no more, no fewer), each a
       non-empty string within its own _FIELD_MAX_LENGTHS budget. Any other shape (missing
       key, extra key, non-string value, oversized value, or `content` that isn't valid JSON
       at all) is rejected. The re-serialized combined text is then checked once more against
       _MAX_COMBINED_LENGTH as a final backstop.
    """
    if not isinstance(data, dict):
        raise LLMProviderUnavailableError(f"{provider_label} response was not a JSON object.")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMProviderUnavailableError(f"{provider_label} response has no 'choices' list.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise LLMProviderUnavailableError(f"{provider_label} response's choices[0] has no 'message' object.")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMProviderUnavailableError(f"{provider_label} response's message.content is missing, empty, or not a string.")

    try:
        structured = json.loads(content)
    except ValueError as exc:
        raise LLMProviderUnavailableError(f"{provider_label} response's message.content was not valid JSON.") from exc

    if not isinstance(structured, dict) or set(structured.keys()) != set(_EXPLANATION_FIELDS):
        raise LLMProviderUnavailableError(
            f"{provider_label} structured response did not match the expected "
            f"{sorted(_EXPLANATION_FIELDS)} schema: {structured!r}"
        )

    for field_name in _EXPLANATION_FIELDS:
        value = structured[field_name]
        if not isinstance(value, str) or not value.strip():
            raise LLMProviderUnavailableError(
                f"{provider_label} structured response's {field_name!r} is missing, empty, or not a string."
            )
    too_long = tuple(f for f in _EXPLANATION_FIELDS if len(structured[f]) > _FIELD_MAX_LENGTHS[f])
    if too_long:
        raise FieldLengthError(
            f"{provider_label} structured response's {too_long[0]!r} exceeds max length "
            f"{_FIELD_MAX_LENGTHS[too_long[0]]} characters.",
            dict(structured), too_long,
        )

    canonical = {field_name: structured[field_name] for field_name in _EXPLANATION_FIELDS}
    raw_text = json.dumps(canonical, separators=(",", ":"))
    if len(raw_text) > _MAX_COMBINED_LENGTH:
        raise LLMProviderUnavailableError(
            f"{provider_label} structured response exceeds combined max length {_MAX_COMBINED_LENGTH} characters."
        )
    return raw_text


def _extract_groq_structured_explanation(data: Any) -> str:
    return _extract_structured_explanation(data, provider_label="Groq")


AMD_CHAT_COMPLETIONS_PATH = "/chat/completions"

# Qwen3.8-27B (the verified AMD Token Factory model) is a reasoning model whose hidden
# `reasoning` trace and final `content` share one max_tokens budget. The single-sentence
# summary this adapter used to ask for fit in 800, then 1500, tokens of budget. The four-field
# structured explanation is a meaningfully heavier completion task, and live debugging after
# that change (real requests against this endpoint, synthetic non-medical data) showed
# reasoning_tokens alone exhausting the full former 1500-token budget on every attempt
# (reasoning_tokens=1500/1500, content empty, finish_reason="length") -- a 100% fallback rate
# at that budget for this task, not the "thin margin" the single-sentence design had. Raising
# the budget to 2200 produced one full real success (reasoning_tokens=1613, content=431 chars,
# finish_reason="stop", all four fields within their own _FIELD_MAX_LENGTHS). A further test at
# 2600 still exhausted the budget on reasoning alone (reasoning_tokens=2600/2600) -- this
# model's reasoning length is genuinely non-deterministic run to run even at temperature=0.0,
# so no budget value can *guarantee* success every call; 2200 is the smallest budget with a
# confirmed real success, balancing that against request latency (an 89s real call was observed
# at a 2600-token budget). The local deterministic fallback (propose_with_fallback(), already
# tested) remains the safety net for whichever calls still exhaust it -- raising this budget
# does not change where the answer is parsed from (content was, and remains, the correct field)
# or weaken any screening.
AMD_MAX_TOKENS = 2200

# Doubled from the original 60s alongside the AMD_MAX_TOKENS increase above: generating up to
# ~1700 completion tokens for the heavier four-field explanation measurably takes longer than
# the single-sentence design's one real verified call did, and a timeout that fires before a
# would-be-successful generation finishes just forces an unnecessary local fallback.
DEFAULT_AMD_TIMEOUT_SECONDS = 120.0

# json_object (not Groq's strict json_schema) per the verified contract: this endpoint was
# tested live with response_format={"type": "json_object"} and returned valid structured JSON
# in message.content. The *outer* message.content -> inner {"summary": str} validation is
# identical to Groq's either way (_extract_structured_summary), so this adapter still never
# accepts a response that doesn't match that shape, even without schema-level enforcement.
_AMD_RESPONSE_FORMAT = {"type": "json_object"}


class AMDTokenFactoryLLMAdapter:
    """Real network calls to the verified AMD Token Factory chat-completions endpoint.

    Disabled unless ALL of the following hold -- each checked independently, so a partial
    configuration never silently proceeds: `enabled=True` (AMD_TOKENFACTORY_ENABLED),
    `base_url`, `api_key`, and `model` all set (AMD_TOKENFACTORY_BASE_URL/_API_KEY/_MODEL).
    `build_adapter()` in llm_config.py additionally requires LLM_PROVIDER=amd_token_factory
    AND LLM_ALLOW_EXTERNAL_CALLS=true before this class is ever constructed with a real
    base_url/api_key at all -- this class's own checks are defense in depth, not the only gate.

    Verified contract (one live, synthetic, non-medical smoke test; see README):
    POST {base_url}/chat/completions, `Authorization: Bearer <api_key>`, OpenAI-compatible
    request/response shape, `response_format={"type": "json_object"}`, HTTP 200, valid JSON.

    Same safety properties as GroqAdapter: the request body is built exclusively from
    `redact_for_external_llm()` (never study_uid/checkpoint_path/local paths), `reasoning` is
    never read as the answer, `cited_evidence` is always `()`, and every error path raises
    `LLMProviderUnavailableError` rather than crashing or guessing.
    """

    name = "amd_token_factory"

    def __init__(
        self, *, enabled: bool = False, base_url: Optional[str] = None,
        api_key: Optional[str] = None, model: Optional[str] = None,
        timeout: float = DEFAULT_AMD_TIMEOUT_SECONDS,
    ):
        self.enabled = enabled
        self.base_url = base_url or os.environ.get("AMD_TOKENFACTORY_BASE_URL")
        self.api_key = api_key or os.environ.get("AMD_TOKENFACTORY_API_KEY")
        self.model = model or os.environ.get("AMD_TOKENFACTORY_MODEL")
        self.timeout = timeout

    def _unavailable_reason(self) -> Optional[str]:
        if not self.enabled:
            return "AMD Token Factory is not enabled (AMD_TOKENFACTORY_ENABLED)."
        if not self.base_url:
            return "AMD_TOKENFACTORY_BASE_URL is not set."
        if not self.api_key:
            return "AMD_TOKENFACTORY_API_KEY is not set."
        if not self.model:
            return "AMD_TOKENFACTORY_MODEL is not set."
        return None

    def status(self) -> LLMProviderStatus:
        reason = self._unavailable_reason()
        if reason is not None:
            return LLMProviderStatus(self.name, False, f"{reason} No network call attempted.")
        return LLMProviderStatus(
            self.name, True,
            f"Configured (model={self.model}). Verified reachable by one prior synthetic smoke "
            "test; not verified for real evidence content.",
        )

    def propose(self, bundle: EvidenceBundle, consensus: ConsensusResult) -> RawProposal:
        reason = self._unavailable_reason()
        if reason is not None:
            raise LLMProviderUnavailableError(f"AMD Token Factory unavailable: {reason}")

        payload = redact_for_external_llm(bundle, consensus)
        _assert_no_leakage(payload, bundle)

        url = self.base_url.rstrip("/") + AMD_CHAT_COMPLETIONS_PATH
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            "response_format": _AMD_RESPONSE_FORMAT,
            "reasoning_effort": "low",
            "temperature": 0.0,
            "max_tokens": AMD_MAX_TOKENS,
            "stream": False,
        }
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=body, timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise LLMProviderUnavailableError(f"AMD Token Factory request timed out after {self.timeout}s.") from exc
        except requests.RequestException as exc:
            raise LLMProviderUnavailableError(f"AMD Token Factory request failed: {exc}") from exc

        if response.status_code in (401, 403):
            raise LLMProviderUnavailableError(
                f"AMD Token Factory rejected credentials (HTTP {response.status_code})."
            )
        if response.status_code == 429:
            raise LLMProviderUnavailableError("AMD Token Factory rate limit exceeded (HTTP 429).")
        if response.status_code in (400, 404):
            raise LLMProviderUnavailableError(
                f"AMD Token Factory returned HTTP {response.status_code} (possibly an unsupported "
                f"model {self.model!r}): {response.text[:200]!r}"
            )
        if response.status_code != 200:
            raise LLMProviderUnavailableError(
                f"AMD Token Factory returned HTTP {response.status_code}: {response.text[:200]!r}"
            )

        try:
            data: Any = response.json()
        except ValueError as exc:
            raise LLMProviderUnavailableError("AMD Token Factory response was not valid JSON.") from exc

        text = _extract_structured_explanation(data, provider_label="AMD Token Factory")
        return RawProposal(raw_text=text, cited_evidence=())


def propose_with_fallback(
    primary, bundle: EvidenceBundle, consensus: ConsensusResult, *, fallback=None,
) -> tuple[RawProposal, str]:
    """Try `primary.propose()`; on any LLMProviderUnavailableError or NotImplementedError,
    fall back to `fallback` (LocalDeterministicAdapter() by default). Returns
    (proposal, name_of_the_provider_that_actually_produced_it) so the caller can be honest
    about which one ran. Never raises for provider-availability reasons -- the fallback is
    always LocalDeterministicAdapter unless the caller passes a different one, and that
    adapter cannot itself be unavailable.
    """
    fallback = fallback if fallback is not None else LocalDeterministicAdapter()
    try:
        return primary.propose(bundle, consensus), getattr(primary, "name", type(primary).__name__)
    except (LLMProviderUnavailableError, NotImplementedError) as exc:
        proposal = fallback.propose(bundle, consensus)
        return proposal, f"{getattr(fallback, 'name', type(fallback).__name__)} (fallback: {exc})"
