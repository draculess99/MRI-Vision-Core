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
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"  # see .env.example: llama-3.3-70b-versatile 404'd for a real key tested here
DEFAULT_TIMEOUT_SECONDS = 10.0

# Every key an external provider is ever allowed to see. redact_for_external_llm() builds its
# output strictly from this set; nothing else is ever added.
_ALLOWED_KEYS = frozenset(
    {
        "workflow_status", "specialist_groups", "critic_codes", "critic_severities",
        "quality_gate_status", "is_ensemble", "model_count", "limitations",
    }
)

# A real AMD Token Factory response once produced a 619-character summary, which
# screen_proposal() correctly BLOCKED for exceeding mri_core's own MAX_PROPOSAL_LENGTH (500) --
# the shared gate did its job, but a tighter prompt avoids relying on that gate alone for every
# response. The adapter-level length check below (_MAX_SUMMARY_LENGTH) is a second, independent
# backstop at the same threshold, so an oversized response is rejected with a specific,
# diagnosable LLMProviderUnavailableError instead of silently reaching screen_proposal() and
# coming back opaquely BLOCKED.
_SUMMARY_CHAR_BUDGET = 350  # comfortable margin under MAX_PROPOSAL_LENGTH (500)

_SYSTEM_PROMPT = (
    "You are a research-workflow assistant for a non-diagnostic MRI research prototype. "
    "You will be given deterministic, de-identified workflow data: a workflow status, grouped "
    "research model scores, structural finding codes, and stated limitations. No patient "
    "identifier, study identifier, file path, or image data is present or available to you. "
    "Respond with a JSON object matching the given schema: a single \"summary\" field containing "
    f"a plain-language summary of this workflow data only, in ONE short sentence of no more than "
    f"{_SUMMARY_CHAR_BUDGET} characters total -- brevity is required, not optional; do not pad, "
    "repeat, or add caveats beyond what is asked. Never state or imply a diagnosis, a clinical "
    "finding, certainty, or that any clinician approved or reviewed anything. Never invent facts "
    "not present in the data you were given."
)

# Structured-output schema for Groq's response_format: a single required string field. Using
# json_schema (strict mode) rather than freeform text is what makes the *final* answer land in
# message.content reliably for reasoning models like openai/gpt-oss-20b -- without it, a
# reasoning model can spend its whole token budget on hidden reasoning and leave message.content
# empty (see hackathon/amd/README.md's Groq smoke-test notes for the real failure this fixes).
_REVIEWER_RESULT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "reviewer_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
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


class LocalDeterministicAdapter:
    """Always available. No network call, no credential, no code path that could leak
    anything -- it only restates fields already present on bundle/consensus, same pattern as
    the shared OfflineStubAdapter."""

    name = "local"

    def status(self) -> LLMProviderStatus:
        return LLMProviderStatus(self.name, True, "Always available; no network call, no credential.")

    def propose(self, bundle: EvidenceBundle, consensus: ConsensusResult) -> RawProposal:
        text = (
            f"Workflow status is {consensus.workflow_status.value}. "
            f"{bundle.model_count} model(s) were used. Quality gate: {bundle.quality_gate_status.value}. "
            f"Critic findings: {len(consensus.critic_findings)}."
        )
        return RawProposal(raw_text=text, cited_evidence=bundle.evidence_references)


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

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            "response_format": _REVIEWER_RESULT_RESPONSE_FORMAT,
            "reasoning_effort": "low",
            "temperature": 0.0,
            "max_tokens": 800,
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
            raise LLMProviderUnavailableError(f"Groq returned HTTP {response.status_code}: {response.text[:200]!r}")

        try:
            data: Any = response.json()
        except ValueError as exc:
            raise LLMProviderUnavailableError("Groq response was not valid JSON.") from exc

        text = _extract_groq_structured_summary(data)
        return RawProposal(raw_text=text, cited_evidence=())


_MAX_SUMMARY_LENGTH = MAX_PROPOSAL_LENGTH  # same 500-char cap screen_proposal() itself enforces;
# exceeding it is rejected here first, with a specific error, rather than reaching screen_proposal()
# and coming back only as an opaque BLOCKED. This is a second check at the SAME threshold, not a
# looser one -- it never permits anything screen_proposal() would reject, only fails earlier/clearer.


def _extract_structured_summary(data: Any, *, provider_label: str) -> str:
    """Strict schema check shared by every network-backed adapter (Groq, AMD Token Factory),
    in two layers -- both rejecting outright rather than guessing:

    1. The outer HTTP response: only `choices[0].message.content` is ever read. A reasoning
       model's `reasoning` (or any `reasoning_content`-style) field is NEVER read as a
       fallback or substitute for `content` -- that field holds the model's hidden reasoning
       trace, not its reviewed answer, and treating it as the answer would silently accept
       unreviewed, unstructured text that bypassed the schema this function enforces.
    2. The inner structured payload: `content` must itself be valid JSON -- a dict with
       exactly one key, `summary`, holding a non-empty string no longer than
       `_MAX_SUMMARY_LENGTH`. Any other shape (missing key, extra key, non-string value,
       oversized value, or `content` that isn't valid JSON at all) is rejected.
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

    if not isinstance(structured, dict) or set(structured.keys()) != {"summary"}:
        raise LLMProviderUnavailableError(
            f"{provider_label} structured response did not match the expected {{'summary': str}} schema: {structured!r}"
        )
    summary = structured["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise LLMProviderUnavailableError(f"{provider_label} structured response's 'summary' is missing, empty, or not a string.")
    if len(summary) > _MAX_SUMMARY_LENGTH:
        raise LLMProviderUnavailableError(
            f"{provider_label} structured response's 'summary' exceeds max length {_MAX_SUMMARY_LENGTH} characters."
        )
    return summary


def _extract_groq_structured_summary(data: Any) -> str:
    return _extract_structured_summary(data, provider_label="Groq")


AMD_CHAT_COMPLETIONS_PATH = "/chat/completions"
DEFAULT_AMD_TIMEOUT_SECONDS = 60.0  # matches the one verified live connectivity test's timeout

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
            "max_tokens": 800,
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

        text = _extract_structured_summary(data, provider_label="AMD Token Factory")
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
