"""Tests for hackathon/amd/llm_providers.py: the provider-neutral mri_core.agents.llm_adapter
implementations (LocalDeterministicAdapter, GroqAdapter, AMDTokenFactoryLLMAdapter) and the
redaction boundary / fallback logic around them.

Every test that could reach the network mocks `requests.post` and additionally asserts it was
never called when it shouldn't be -- no test in this file makes, or risks making, a real
network call, regardless of ambient environment variables. GROQ_API_KEY and
AMD_TOKENFACTORY_* are explicitly cleared via monkeypatch in every test (never left to ambient
environment), since this project's own shell has a real GROQ_API_KEY set.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
requests = pytest.importorskip("requests")

from hackathon.amd.agent_bridge_adapter import to_study_inference_result
from hackathon.amd.kaggle_mil_bridge import TARGET_COLUMNS
from hackathon.amd.llm_providers import (
    _ALLOWED_KEYS,
    _FIELD_MAX_LENGTHS,
    AMD_MAX_TOKENS,
    AMDTokenFactoryLLMAdapter,
    GroqAdapter,
    LLMProviderError,
    LLMProviderUnavailableError,
    LocalDeterministicAdapter,
    propose_with_fallback,
    redact_for_external_llm,
)
from mri_core.agents.contracts import QualityGateStatus
from mri_core.agents.llm_adapter import ProposalStatus, RawProposal, screen_proposal
from mri_core.agents.orchestrator import process_inference_result


@pytest.fixture(autouse=True)
def _no_ambient_llm_credentials(monkeypatch):
    """This project's own shell has a real GROQ_API_KEY set -- never let any test in this
    file see it or any other provider credential by accident."""
    for key in (
        "GROQ_API_KEY", "GROQ_MODEL", "AMD_TOKENFACTORY_ENABLED",
        "AMD_TOKENFACTORY_BASE_URL", "AMD_TOKENFACTORY_API_KEY", "AMD_TOKENFACTORY_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


class _NoNetworkGuard:
    """Fails the test immediately if requests.post is ever called."""

    def __enter__(self):
        self._original = requests.post

        def _blocked(*args, **kwargs):
            raise AssertionError(f"requests.post was called; no network access is allowed here. args={args!r}")

        requests.post = _blocked
        return self

    def __exit__(self, *exc_info):
        requests.post = self._original


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, text="{}"):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text

    def json(self):
        if isinstance(self._json_body, Exception):
            raise self._json_body
        return self._json_body


def _workflow(study_uid="1.2.3.secret", checkpoint_path="C:/secret/dir/fold0-knee-mil-best.ckpt"):
    probs = {col: round(0.1 * (i % 7), 4) for i, col in enumerate(TARGET_COLUMNS)}
    result = to_study_inference_result(
        study_uid=study_uid, probabilities=probs, latency_seconds=0.3,
        checkpoint_path=checkpoint_path, checkpoint_sha256="deadbeef" * 8,
        disclaimer="bridge disclaimer",
    )
    return process_inference_result(result, quality_gate_status=QualityGateStatus.NOT_ASSESSED)


# ---------------------------------------------------------------------------- data leakage


class TestRedactionBoundary:
    def test_output_is_limited_to_the_allowlist(self):
        wf = _workflow()
        payload = redact_for_external_llm(wf.bundle, wf.consensus)
        assert set(payload.keys()) <= _ALLOWED_KEYS

    def test_never_contains_study_uid(self):
        wf = _workflow(study_uid="1.2.3.super-secret-uid")
        payload = redact_for_external_llm(wf.bundle, wf.consensus)
        assert "1.2.3.super-secret-uid" not in json.dumps(payload)

    def test_never_contains_checkpoint_path_or_local_path_separators(self):
        wf = _workflow(checkpoint_path="C:/Users/someone/secret/fold0-knee-mil-best.ckpt")
        payload = redact_for_external_llm(wf.bundle, wf.consensus)
        serialized = json.dumps(payload)
        assert "secret" not in serialized
        assert "fold0-knee-mil-best.ckpt" not in serialized
        assert "C:/" not in serialized and "C:\\" not in serialized

    def test_never_contains_checkpoint_sha256(self):
        wf = _workflow()
        payload = redact_for_external_llm(wf.bundle, wf.consensus)
        assert "deadbeef" * 8 not in json.dumps(payload)

    def test_critic_findings_reduced_to_codes_and_severities_only(self):
        """Critic finding .message / .evidence_reference text is never sent -- only the
        structured .code and .severity are, since message text could in principle restate
        a checkpoint path (e.g. MISSING_PROVENANCE_HASH's evidence_reference)."""
        wf = _workflow()
        payload = redact_for_external_llm(wf.bundle, wf.consensus)
        for finding in wf.critic_findings:
            if finding.message:
                assert finding.message not in json.dumps(payload)

    def test_groq_propose_runs_the_leakage_assertion_before_any_network_call(self):
        """Belt-and-suspenders: even if redact_for_external_llm() were ever broken, GroqAdapter
        must refuse to call the network with a payload containing the study_uid."""
        wf = _workflow(study_uid="1.2.3.secret")
        adapter = GroqAdapter(api_key="fake-key")
        import hackathon.amd.llm_providers as llm_providers

        broken_payload = {**redact_for_external_llm(wf.bundle, wf.consensus), "leak": wf.bundle.study_uid}

        def _broken_redact(bundle, consensus):
            return broken_payload

        original = llm_providers.redact_for_external_llm
        llm_providers.redact_for_external_llm = _broken_redact
        try:
            with _NoNetworkGuard():
                with pytest.raises(LLMProviderError):
                    adapter.propose(wf.bundle, wf.consensus)
        finally:
            llm_providers.redact_for_external_llm = original


# ---------------------------------------------------------------------------- local adapter


class TestLocalDeterministicAdapter:
    def test_never_touches_network(self):
        import inspect

        source = inspect.getsource(LocalDeterministicAdapter)
        for forbidden in ("requests", "urllib", "socket", "http.client", "api_key", "API_KEY"):
            assert forbidden not in source

    def test_always_available(self):
        assert LocalDeterministicAdapter().status().available is True

    def test_proposal_is_quarantined_not_trusted(self):
        wf = _workflow()
        proposal = LocalDeterministicAdapter().propose(wf.bundle, wf.consensus)
        outcome = screen_proposal(proposal, wf.bundle)
        assert outcome.status == ProposalStatus.QUARANTINED

    def test_produces_all_four_structured_fields(self):
        """Local fallback must produce the SAME four logical sections as the network
        providers, deterministically, so the UI layout is identical regardless of provider."""
        wf = _workflow()
        proposal = LocalDeterministicAdapter().propose(wf.bundle, wf.consensus)
        parsed = json.loads(proposal.raw_text)
        assert set(parsed.keys()) == {
            "specialist_summary", "critic_summary", "consensus_explanation", "reviewer_focus",
        }
        assert all(isinstance(v, str) and v.strip() for v in parsed.values())

    def test_consensus_explanation_is_traceable_to_actual_critic_findings_and_status(self):
        """The consensus_explanation must explicitly connect the actual Critic findings/status
        to the actual deterministic workflow_status -- derived directly from consensus.reasons
        (the Consensus layer's own stated reasons), never invented."""
        wf = _workflow()
        proposal = LocalDeterministicAdapter().propose(wf.bundle, wf.consensus)
        parsed = json.loads(proposal.raw_text)
        assert wf.consensus.workflow_status.value in parsed["consensus_explanation"]
        for code in (f.code for f in wf.consensus.critic_findings):
            assert code in parsed["critic_summary"]

    def test_reviewer_focus_is_derived_from_actual_required_next_steps(self):
        wf = _workflow()
        proposal = LocalDeterministicAdapter().propose(wf.bundle, wf.consensus)
        parsed = json.loads(proposal.raw_text)
        if wf.consensus.required_next_steps:
            first_words = wf.consensus.required_next_steps[0][:20]
            assert first_words in parsed["reviewer_focus"]

    def test_fits_within_the_shared_combined_length_cap(self):
        """Even with every critic code concatenated, the local fallback's own output must stay
        within the same shared cap the network providers are held to -- never silently exceed
        it and rely on truncation alone (the _truncate() helper enforces this per field)."""
        from mri_core.agents.llm_adapter import MAX_PROPOSAL_LENGTH

        wf = _workflow()
        proposal = LocalDeterministicAdapter().propose(wf.bundle, wf.consensus)
        assert len(proposal.raw_text) <= MAX_PROPOSAL_LENGTH


# ---------------------------------------------------------------------------- Groq adapter


class TestGroqMissingCredentials:
    def test_status_reports_unavailable_without_key(self):
        status = GroqAdapter(api_key=None).status()
        assert status.available is False
        assert "GROQ_API_KEY" in status.detail

    def test_propose_raises_without_network_call(self):
        wf = _workflow()
        with _NoNetworkGuard():
            with pytest.raises(LLMProviderUnavailableError):
                GroqAdapter(api_key=None).propose(wf.bundle, wf.consensus)


def _valid_explanation_fields(**overrides) -> dict:
    """Safe, well-formed defaults for all four _EXPLANATION_FIELDS, each within its own
    _FIELD_MAX_LENGTHS budget -- override one or more to test a specific field."""
    fields = {
        "specialist_summary": "Specialist grouped 12 scores into 4 categories.",
        "critic_summary": "Critic found 3 issues: quality not assessed, warnings, no evidence refs.",
        "consensus_explanation": "Consensus assigned MORE_EVIDENCE_REQUIRED from those critic findings.",
        "reviewer_focus": "Complete the quality assessment before considering approval.",
    }
    # Only the untouched defaults must fit their own budget -- callers deliberately override a
    # field with an oversized/unsafe value in some tests, and that intentional override must
    # reach the real production validation code, not be caught here first.
    for field_name, value in fields.items():
        if field_name not in overrides:
            assert len(value) <= _FIELD_MAX_LENGTHS[field_name], (
                f"test fixture default for {field_name!r} ({len(value)} chars) exceeds the real "
                f"_FIELD_MAX_LENGTHS cap ({_FIELD_MAX_LENGTHS[field_name]}); update the fixture."
            )
    fields.update(overrides)
    return fields


def _structured_content(**overrides) -> str:
    """Build the `message.content` string a provider's structured-output mode actually
    returns: a JSON-encoded object matching _REVIEWER_RESULT_RESPONSE_FORMAT's four fields."""
    return json.dumps(_valid_explanation_fields(**overrides))


class TestGroqSuccess:
    def test_successful_response_produces_a_proposal(self, monkeypatch):
        """AMD Token Factory / Groq structured response parses correctly: all four fields
        round-trip through the canonical raw_text unchanged."""
        wf = _workflow()
        fields = _valid_explanation_fields()

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body={"choices": [{"message": {"content": json.dumps(fields)}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        proposal = GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)
        assert json.loads(proposal.raw_text) == fields
        assert proposal.cited_evidence == ()

    def test_request_body_contains_only_the_redacted_payload(self, monkeypatch):
        wf = _workflow(study_uid="1.2.3.secret", checkpoint_path="C:/secret/x.ckpt")
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content()}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)

        serialized_request = json.dumps(captured["json"])
        assert "1.2.3.secret" not in serialized_request
        assert "secret" not in serialized_request
        assert "x.ckpt" not in serialized_request

    def test_request_body_requests_structured_low_effort_output(self, monkeypatch):
        wf = _workflow()
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content()}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        GroqAdapter(api_key="fake-key", model="openai/gpt-oss-20b").propose(wf.bundle, wf.consensus)

        assert captured["json"]["reasoning_effort"] == "low"
        assert captured["json"]["temperature"] == 0.0
        from hackathon.amd.llm_providers import GROQ_MAX_TOKENS

        assert captured["json"]["max_tokens"] == GROQ_MAX_TOKENS
        assert captured["json"]["response_format"]["type"] == "json_schema"
        assert captured["json"]["response_format"]["json_schema"]["strict"] is True
        schema_props = captured["json"]["response_format"]["json_schema"]["schema"]["properties"]
        assert set(schema_props.keys()) == {
            "specialist_summary", "critic_summary", "consensus_explanation", "reviewer_focus",
        }

    def test_critic_findings_appear_in_the_structured_request(self, monkeypatch):
        """The actual Critic findings computed for this workflow (critic_codes/severities) must
        be present in the request payload the explanation is asked to explain."""
        wf = _workflow()
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content()}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)

        sent_payload = json.loads(captured["json"]["messages"][1]["content"])
        assert sent_payload["critic_codes"] == [f.code for f in wf.consensus.critic_findings]
        assert sent_payload["critic_severities"] == [f.severity.value for f in wf.consensus.critic_findings]

    def test_consensus_workflow_status_appears_in_the_structured_request(self, monkeypatch):
        """The actual deterministic Consensus workflow_status must be present in the request
        payload -- the explanation is asked to explain it, never to invent or re-derive it."""
        wf = _workflow()
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content()}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)

        sent_payload = json.loads(captured["json"]["messages"][1]["content"])
        assert sent_payload["workflow_status"] == wf.consensus.workflow_status.value

    def test_reasoning_field_is_never_read_as_the_answer(self, monkeypatch):
        """A reasoning model's hidden reasoning trace must never be treated as the reviewed
        answer, even if present alongside a well-formed structured message.content."""
        wf = _workflow()
        fields = _valid_explanation_fields(specialist_summary="The actual reviewed summary.")

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body={
                "choices": [{"message": {
                    "reasoning": "This is internal chain-of-thought and must never be used.",
                    "content": json.dumps(fields),
                }}]
            })

        monkeypatch.setattr(requests, "post", _fake_post)
        proposal = GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)
        assert json.loads(proposal.raw_text)["specialist_summary"] == "The actual reviewed summary."
        assert "chain-of-thought" not in proposal.raw_text

    def test_empty_content_with_only_reasoning_present_is_still_rejected(self, monkeypatch):
        """Reproduces the original failure this change fixes: a reasoning model that spends
        its whole budget on `reasoning` and leaves `content` empty must still be cleanly
        rejected, never silently fall back to reading `reasoning` as the answer."""
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body={
                "choices": [{"message": {"reasoning": "all the budget went here", "content": ""}}]
            })

        monkeypatch.setattr(requests, "post", _fake_post)
        with pytest.raises(LLMProviderUnavailableError, match="content"):
            GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)

    @pytest.mark.parametrize(
        "missing_field",
        ["specialist_summary", "critic_summary", "consensus_explanation", "reviewer_focus"],
    )
    def test_missing_required_field_is_rejected(self, monkeypatch, missing_field):
        """A response missing one of the four required fields must be rejected outright --
        never silently accepted with a blank/missing section."""
        wf = _workflow()
        fields = _valid_explanation_fields()
        del fields[missing_field]

        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": json.dumps(fields)}}]}),
        )
        with pytest.raises(LLMProviderUnavailableError, match="schema"):
            GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)

    def test_missing_required_field_falls_back_to_local_safely(self, monkeypatch):
        """End to end: a malformed (missing-field) network response must fall back to the
        local deterministic adapter, never raise out of propose_with_fallback, and the local
        fallback's own output must still carry all four fields."""
        wf = _workflow()
        fields = _valid_explanation_fields()
        del fields["reviewer_focus"]

        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": json.dumps(fields)}}]}),
        )
        proposal, used = propose_with_fallback(GroqAdapter(api_key="fake-key"), wf.bundle, wf.consensus)
        assert used.startswith("local (fallback:")
        parsed = json.loads(proposal.raw_text)
        assert set(parsed.keys()) == {"specialist_summary", "critic_summary", "consensus_explanation", "reviewer_focus"}


class TestGroqFailureModes:
    def test_timeout(self, monkeypatch):
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            raise requests.Timeout("simulated timeout")

        monkeypatch.setattr(requests, "post", _fake_post)
        with pytest.raises(LLMProviderUnavailableError, match="timed out"):
            GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)

    def test_connection_error(self, monkeypatch):
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            raise requests.ConnectionError("simulated connection error")

        monkeypatch.setattr(requests, "post", _fake_post)
        with pytest.raises(LLMProviderUnavailableError):
            GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)

    def test_http_error_status(self, monkeypatch):
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            return _FakeResponse(status_code=401, text="unauthorized")

        monkeypatch.setattr(requests, "post", _fake_post)
        with pytest.raises(LLMProviderUnavailableError, match="401"):
            GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)

    def test_invalid_json(self, monkeypatch):
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body=ValueError("not json"))

        monkeypatch.setattr(requests, "post", _fake_post)
        with pytest.raises(LLMProviderUnavailableError, match="JSON"):
            GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)

    @pytest.mark.parametrize(
        "bad_body",
        [
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": ""}}]},
            {"choices": [{"message": {"content": 12345}}]},
            {"choices": "not-a-list"},
            {"unexpected_top_level_field": "x"},
            {"choices": [{"message": {"content": "not valid json at all"}}]},
            {"choices": [{"message": {"content": "[1, 2, 3]"}}]},
            # extra unsupported field alongside all four required ones
            {"choices": [{"message": {"content": json.dumps({**_valid_explanation_fields(), "diagnosis": "x"})}}]},
            # one field wrong type
            {"choices": [{"message": {"content": json.dumps({**_valid_explanation_fields(), "critic_summary": 123})}}]},
            # one field empty string
            {"choices": [{"message": {"content": json.dumps({**_valid_explanation_fields(), "reviewer_focus": ""})}}]},
            # one field over its own per-field budget
            {"choices": [{"message": {"content": json.dumps({**_valid_explanation_fields(), "specialist_summary": "x" * 200})}}]},
        ],
    )
    def test_unexpected_or_malformed_schema_is_rejected(self, monkeypatch, bad_body):
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body=bad_body)

        monkeypatch.setattr(requests, "post", _fake_post)
        with pytest.raises(LLMProviderUnavailableError):
            GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)


# ------------------------------------------------------------------------- unsafe claims


class TestUnsafeClaimsAreBlockedRegardlessOfProvider:
    @pytest.mark.parametrize(
        "unsafe_text",
        [
            "This study shows a confirmed diagnosis of ACL tear.",
            "Approved by clinician on review.",
            "The patient has a definite meniscus tear.",
        ],
    )
    def test_groq_unsafe_text_is_blocked(self, monkeypatch, unsafe_text):
        """Screening remains unchanged: an unsafe claim in ANY of the four fields still
        BLOCKs the whole combined raw_text, exactly as it did for the single-field design."""
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content(consensus_explanation=unsafe_text)}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        proposal = GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)
        outcome = screen_proposal(proposal, wf.bundle)
        assert outcome.status == ProposalStatus.BLOCKED

    def test_fabricated_citation_from_any_provider_is_blocked(self):
        wf = _workflow()
        fabricated = RawProposal(raw_text="A safe-sounding summary.", cited_evidence=("fake_evidence.pt",))
        outcome = screen_proposal(fabricated, wf.bundle)
        assert outcome.status == ProposalStatus.BLOCKED


# --------------------------------------------------------------------- AMD Token Factory


def _amd_adapter(**overrides) -> AMDTokenFactoryLLMAdapter:
    kwargs = dict(
        enabled=True, base_url="https://developer.amd.com.cn/radeon/api/v1",
        api_key="fake-amd-key", model="Qwen3.8-27B",
    )
    kwargs.update(overrides)
    return AMDTokenFactoryLLMAdapter(**kwargs)


class TestAMDTokenFactoryDisabledOrMisconfigured:
    def test_disabled_by_default(self):
        assert AMDTokenFactoryLLMAdapter().enabled is False

    def test_disabled_reports_unavailable_without_network(self):
        with _NoNetworkGuard():
            status = AMDTokenFactoryLLMAdapter().status()
        assert status.available is False

    def test_disabled_propose_raises_without_network_call(self):
        with _NoNetworkGuard():
            with pytest.raises(LLMProviderUnavailableError, match="not enabled"):
                _amd_adapter(enabled=False).propose(*_dummy_bundle_consensus())

    @pytest.mark.parametrize("missing_field", ["base_url", "api_key", "model"])
    def test_enabled_but_missing_one_field_raises_without_network_call(self, missing_field):
        with _NoNetworkGuard():
            with pytest.raises(LLMProviderUnavailableError):
                _amd_adapter(**{missing_field: None}).propose(*_dummy_bundle_consensus())

    def test_status_reports_available_only_when_fully_configured(self):
        assert _amd_adapter().status().available is True
        assert _amd_adapter(model=None).status().available is False


class TestAMDTokenFactorySuccess:
    def test_successful_response_produces_a_proposal(self, monkeypatch):
        """AMD Token Factory structured response parses correctly: all four fields round-trip
        through the canonical raw_text unchanged."""
        wf = _workflow()
        fields = _valid_explanation_fields()

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body={"choices": [{"message": {"content": json.dumps(fields)}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        proposal = _amd_adapter().propose(wf.bundle, wf.consensus)
        assert json.loads(proposal.raw_text) == fields
        assert proposal.cited_evidence == ()

    def test_request_uses_the_verified_contract(self, monkeypatch):
        wf = _workflow()
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content()}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        _amd_adapter().propose(wf.bundle, wf.consensus)

        assert captured["url"] == "https://developer.amd.com.cn/radeon/api/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer fake-amd-key"
        assert captured["json"]["model"] == "Qwen3.8-27B"
        assert captured["json"]["response_format"] == {"type": "json_object"}
        assert captured["json"]["reasoning_effort"] == "low"
        assert captured["json"]["temperature"] == 0.0
        assert captured["json"]["stream"] is False

    def test_request_body_contains_only_the_redacted_payload(self, monkeypatch):
        wf = _workflow(study_uid="1.2.3.secret", checkpoint_path="C:/secret/x.ckpt")
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content()}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        _amd_adapter().propose(wf.bundle, wf.consensus)

        serialized_request = json.dumps(captured["json"])
        assert "1.2.3.secret" not in serialized_request
        assert "secret" not in serialized_request
        assert "x.ckpt" not in serialized_request

    def test_critic_findings_appear_in_the_structured_request(self, monkeypatch):
        wf = _workflow()
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content()}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        _amd_adapter().propose(wf.bundle, wf.consensus)

        sent_payload = json.loads(captured["json"]["messages"][1]["content"])
        assert sent_payload["critic_codes"] == [f.code for f in wf.consensus.critic_findings]

    def test_consensus_workflow_status_appears_in_the_structured_request(self, monkeypatch):
        wf = _workflow()
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content()}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        _amd_adapter().propose(wf.bundle, wf.consensus)

        sent_payload = json.loads(captured["json"]["messages"][1]["content"])
        assert sent_payload["workflow_status"] == wf.consensus.workflow_status.value

    def test_api_key_never_appears_in_any_exception_message(self, monkeypatch):
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            return _FakeResponse(status_code=401, text="unauthorized")

        monkeypatch.setattr(requests, "post", _fake_post)
        with pytest.raises(LLMProviderUnavailableError) as exc_info:
            _amd_adapter(api_key="super-secret-amd-key-value").propose(wf.bundle, wf.consensus)
        assert "super-secret-amd-key-value" not in str(exc_info.value)

    def test_reasoning_field_is_never_read_as_the_answer(self, monkeypatch):
        wf = _workflow()
        fields = _valid_explanation_fields(specialist_summary="the real reviewed summary")

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body={
                "choices": [{"message": {
                    "reasoning": "hidden chain-of-thought, must never be used",
                    "content": json.dumps(fields),
                }}]
            })

        monkeypatch.setattr(requests, "post", _fake_post)
        proposal = _amd_adapter().propose(wf.bundle, wf.consensus)
        assert json.loads(proposal.raw_text)["specialist_summary"] == "the real reviewed summary"
        assert "chain-of-thought" not in proposal.raw_text

    def test_real_response_shape_has_a_sibling_reasoning_field_alongside_content(self, monkeypatch):
        """Regression test for the actual AMD Token Factory response shape, captured from five
        live calls (synthetic, non-medical data) during debugging of a reported "message.content
        is missing" failure. Every real call returned top-level
        {id, object, created, model, choices, usage, metadata}, with choices[0].message holding
        {role, content, tool_calls, reasoning} -- content was always present and valid in all
        five calls, confirming it is (and remains) the correct field to parse. This fixture
        mirrors that exact shape, including the extra `tool_calls`/`metadata`/`usage` noise this
        adapter must ignore without erroring."""
        wf = _workflow()
        fields = _valid_explanation_fields(specialist_summary="A real captured-shape summary.")

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body={
                "id": "chatcmpl-abc123", "object": "chat.completion", "created": 1792274349,
                "model": "Qwen3.8-27B",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(fields),
                        "tool_calls": None,
                        "reasoning": "x" * 2000,  # a long hidden reasoning trace, never read
                    },
                    "logprobs": None, "finish_reason": "stop", "matched_stop": 248046,
                }],
                "usage": {"prompt_tokens": 329, "completion_tokens": 543, "reasoning_tokens": 498},
                "metadata": {"request_id": "req_abc", "used_provider": "some-upstream"},
            })

        monkeypatch.setattr(requests, "post", _fake_post)
        proposal = _amd_adapter().propose(wf.bundle, wf.consensus)
        assert json.loads(proposal.raw_text)["specialist_summary"] == "A real captured-shape summary."

    def test_max_tokens_gives_headroom_beyond_the_observed_reasoning_token_usage(self, monkeypatch):
        """Live debugging of the four-field structured explanation (heavier than the old
        single-sentence design) found reasoning_tokens exhausting the full former 1500-token
        budget on every real attempt (100% fallback rate at that budget for this task). Raising
        AMD_MAX_TOKENS to 2200 produced a confirmed real success (reasoning_tokens=1613,
        content=431 chars, finish_reason="stop"). AMD_MAX_TOKENS must stay at or above that
        proven-sufficient value."""
        wf = _workflow()
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content()}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        _amd_adapter().propose(wf.bundle, wf.consensus)

        assert captured["json"]["max_tokens"] == AMD_MAX_TOKENS
        assert AMD_MAX_TOKENS >= 2200  # the smallest budget with a confirmed real live success


class TestAMDTokenFactoryFailureModes:
    def test_timeout(self, monkeypatch):
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            raise requests.Timeout("simulated timeout")

        monkeypatch.setattr(requests, "post", _fake_post)
        with pytest.raises(LLMProviderUnavailableError, match="timed out"):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    def test_connection_error(self, monkeypatch):
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            raise requests.ConnectionError("simulated connection error")

        monkeypatch.setattr(requests, "post", _fake_post)
        with pytest.raises(LLMProviderUnavailableError):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    def test_invalid_credentials_401(self, monkeypatch):
        wf = _workflow()
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=401, text="unauthorized"))
        with pytest.raises(LLMProviderUnavailableError, match="401"):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    def test_invalid_credentials_403(self, monkeypatch):
        wf = _workflow()
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=403, text="forbidden"))
        with pytest.raises(LLMProviderUnavailableError, match="403"):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    def test_rate_limit_429(self, monkeypatch):
        wf = _workflow()
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=429, text="rate limited"))
        with pytest.raises(LLMProviderUnavailableError, match="429"):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    def test_unsupported_model_400(self, monkeypatch):
        wf = _workflow()
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=400, text="bad request"))
        with pytest.raises(LLMProviderUnavailableError, match="400"):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    def test_unsupported_model_404(self, monkeypatch):
        wf = _workflow()
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=404, text="not found"))
        with pytest.raises(LLMProviderUnavailableError, match="404"):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    def test_other_http_error(self, monkeypatch):
        wf = _workflow()
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=503, text="unavailable"))
        with pytest.raises(LLMProviderUnavailableError, match="503"):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    def test_invalid_json(self, monkeypatch):
        wf = _workflow()
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(json_body=ValueError("not json")))
        with pytest.raises(LLMProviderUnavailableError, match="JSON"):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    @pytest.mark.parametrize(
        "bad_body",
        [
            {},
            {"choices": []},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": ""}}]},
            {"choices": [{"message": {"content": "not valid json at all"}}]},
            {"choices": [{"message": {"content": json.dumps({**_valid_explanation_fields(), "diagnosis": "x"})}}]},
            {"choices": [{"message": {"content": json.dumps({**_valid_explanation_fields(), "critic_summary": 123})}}]},
        ],
    )
    def test_malformed_or_unexpected_schema_is_rejected(self, monkeypatch, bad_body):
        wf = _workflow()
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(json_body=bad_body))
        with pytest.raises(LLMProviderUnavailableError):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    @pytest.mark.parametrize(
        "field_name", ["specialist_summary", "critic_summary", "consensus_explanation", "reviewer_focus"]
    )
    def test_field_exceeding_its_own_budget_is_rejected(self, monkeypatch, field_name):
        """Regression for the historical AMD Token Factory incident (a real response once
        returned a 619-character single-field summary that reached screen_proposal() and came
        back opaquely BLOCKED). Each of the four fields now has its own, much tighter budget,
        and an oversized field must still be rejected outright by this adapter, before
        screen_proposal() ever sees it, with a specific, diagnosable error."""
        wf = _workflow()
        oversized_content = _structured_content(**{field_name: "x" * 3000})
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": oversized_content}}]}),
        )
        with pytest.raises(LLMProviderUnavailableError, match="exceeds max length"):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    def test_each_field_at_exactly_its_own_cap_is_accepted(self, monkeypatch):
        """The adapter-level per-field caps must match, not undercut, _FIELD_MAX_LENGTHS -- a
        response with every field at exactly its own max length must still pass through to
        screen_proposal(), and the combined text must still fit under the shared 500-char cap."""
        from hackathon.amd.llm_providers import _FIELD_MAX_LENGTHS
        from mri_core.agents.llm_adapter import MAX_PROPOSAL_LENGTH

        wf = _workflow()
        at_cap_fields = {name: "x" * max_len for name, max_len in _FIELD_MAX_LENGTHS.items()}
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": json.dumps(at_cap_fields)}}]}),
        )
        proposal = _amd_adapter().propose(wf.bundle, wf.consensus)
        assert json.loads(proposal.raw_text) == at_cap_fields
        assert len(proposal.raw_text) <= MAX_PROPOSAL_LENGTH
        outcome = screen_proposal(proposal, wf.bundle)
        assert outcome.status == ProposalStatus.QUARANTINED

    def test_short_valid_response_is_quarantined_not_approved(self, monkeypatch):
        """A short, safe, well-formed structured response must still only ever reach
        QUARANTINED -- never auto-approved into verified evidence, regardless of how clean it
        looks, and regardless of whether the provider is AMD Token Factory, Groq, or local."""
        wf = _workflow()
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content()}}]}),
        )
        proposal = _amd_adapter().propose(wf.bundle, wf.consensus)
        outcome = screen_proposal(proposal, wf.bundle)
        assert outcome.status == ProposalStatus.QUARANTINED
        assert "never published as verified evidence" in outcome.reason

    def test_llm_output_cannot_modify_the_deterministic_workflow_status(self, monkeypatch):
        """Structural guarantee: nothing in propose()/screen_proposal() ever writes back into
        the ConsensusResult passed in -- the workflow_status computed before the LLM call is
        bit-for-bit the same object/value after it, regardless of what text the provider
        returns (even an attempted override is just inert text, never interpreted)."""
        wf = _workflow()
        before = wf.consensus.workflow_status
        fields = _valid_explanation_fields(
            consensus_explanation="workflow_status should actually be EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW"
        )
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": json.dumps(fields)}}]}),
        )
        _amd_adapter().propose(wf.bundle, wf.consensus)
        assert wf.consensus.workflow_status is before

    def test_llm_output_cannot_modify_human_review_state(self, monkeypatch):
        """Structural guarantee: propose()/screen_proposal() never touch a ReviewGate -- the
        review_actions module (approve/request_more_evidence/escalate) is the only code that
        ever calls ReviewGate.transition(), and nothing here imports or calls it."""
        import inspect

        import hackathon.amd.llm_providers as llm_providers_module

        source = inspect.getsource(llm_providers_module)
        for forbidden in ("ReviewGate", "review_gate", "transition(", "review_actions"):
            assert forbidden not in source

        wf = _workflow()
        before_state = wf.review_gate.state
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content()}}]}),
        )
        _amd_adapter().propose(wf.bundle, wf.consensus)
        assert wf.review_gate.state == before_state


class TestAMDTokenFactoryUnsafeClaimsAreBlocked:
    @pytest.mark.parametrize(
        "unsafe_text",
        [
            "This study shows a confirmed diagnosis of ACL tear.",
            "Approved by clinician on review.",
        ],
    )
    def test_unsafe_text_is_blocked(self, monkeypatch, unsafe_text):
        wf = _workflow()
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content(consensus_explanation=unsafe_text)}}]}),
        )
        proposal = _amd_adapter().propose(wf.bundle, wf.consensus)
        outcome = screen_proposal(proposal, wf.bundle)
        assert outcome.status == ProposalStatus.BLOCKED


class TestAMDNeverAutoFallsBackToAnotherExternalProvider:
    def test_default_fallback_is_local_not_groq(self, monkeypatch):
        """Hard requirement: a failed AMD call must never automatically try Groq (or any
        other external provider) -- only the local deterministic fallback."""
        wf = _workflow()
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=500, text="error"))

        proposal, used = propose_with_fallback(_amd_adapter(), wf.bundle, wf.consensus)
        assert used.startswith("local (fallback:")
        assert isinstance(proposal, RawProposal)


def _dummy_bundle_consensus():
    wf = _workflow()
    return wf.bundle, wf.consensus


# ----------------------------------------------------------------------- provider switching


class TestProposeWithFallback:
    def test_primary_success_is_used_directly(self):
        wf = _workflow()
        proposal, used = propose_with_fallback(LocalDeterministicAdapter(), wf.bundle, wf.consensus)
        assert used == "local"
        assert isinstance(proposal, RawProposal)

    def test_unavailable_primary_falls_back_to_local(self):
        wf = _workflow()
        with _NoNetworkGuard():
            proposal, used = propose_with_fallback(GroqAdapter(api_key=None), wf.bundle, wf.consensus)
        assert used.startswith("local (fallback:")
        assert isinstance(proposal, RawProposal)

    def test_not_implemented_primary_falls_back_to_local(self):
        wf = _workflow()
        with _NoNetworkGuard():
            proposal, used = propose_with_fallback(
                AMDTokenFactoryLLMAdapter(enabled=True, base_url="x", api_key="y"), wf.bundle, wf.consensus
            )
        assert used.startswith("local (fallback:")

    def test_custom_fallback_is_honored(self):
        wf = _workflow()

        class _CustomFallback:
            name = "custom_fallback"

            def propose(self, bundle, consensus):
                return RawProposal(raw_text="custom fallback text", cited_evidence=())

        with _NoNetworkGuard():
            proposal, used = propose_with_fallback(
                GroqAdapter(api_key=None), wf.bundle, wf.consensus, fallback=_CustomFallback()
            )
        assert used.startswith("custom_fallback (fallback:")
        assert proposal.raw_text == "custom fallback text"

    def test_fallback_result_still_goes_through_screen_proposal(self):
        wf = _workflow()
        with _NoNetworkGuard():
            proposal, _used = propose_with_fallback(GroqAdapter(api_key=None), wf.bundle, wf.consensus)
        outcome = screen_proposal(proposal, wf.bundle)
        assert outcome.status in (ProposalStatus.BLOCKED, ProposalStatus.QUARANTINED)


# --------------------------------------------------------------- Groq length-only repair retry


class TestGroqLengthRepairRetry:
    """Valid JSON that fails ONLY a per-field length cap earns exactly one repair request; nothing else
    is retried, and the repaired answer faces the same validation and screening as any other."""

    LONG = "x" * 90  # over specialist_summary's 60-char cap

    @staticmethod
    def _response(fields_or_text, status=200):
        content = fields_or_text if isinstance(fields_or_text, str) else json.dumps(fields_or_text)
        return _FakeResponse(status_code=status, json_body={"choices": [{"message": {"content": content}}]}, text="body")

    def _script(self, monkeypatch, responses):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(json)
            item = responses[min(len(calls) - 1, len(responses) - 1)]
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(requests, "post", fake_post)
        return calls

    def test_caps_are_unchanged(self):
        from hackathon.amd.llm_providers import _FIELD_MAX_LENGTHS

        assert _FIELD_MAX_LENGTHS == {
            "specialist_summary": 60, "critic_summary": 135, "consensus_explanation": 110, "reviewer_focus": 80,
        }

    def test_prompt_demands_json_only_states_every_cap_and_targets_below_them(self, monkeypatch):
        calls = self._script(monkeypatch, [self._response(_valid_explanation_fields())])
        wf = _workflow()
        GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        system = calls[0]["messages"][0]["content"]
        assert "ONE JSON object and nothing else" in system and "no markdown" in system
        for cap in ("specialist_summary <= 60", "critic_summary <= 135", "consensus_explanation <= 110", "reviewer_focus <= 80"):
            assert cap in system
        for target in ("specialist_summary about 50", "critic_summary about 120", "consensus_explanation about 100", "reviewer_focus about 70"):
            assert target in system
        assert "unusable" in system

    def test_too_long_then_valid_repair_succeeds_with_one_extra_request(self, monkeypatch):
        repaired = _valid_explanation_fields(specialist_summary="Short repaired specialist text.")
        calls = self._script(monkeypatch, [
            self._response(_valid_explanation_fields(specialist_summary=self.LONG)), self._response(repaired),
        ])
        wf = _workflow()
        proposal = GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert len(calls) == 2
        assert json.loads(proposal.raw_text) == repaired
        repair_user = calls[1]["messages"][1]["content"]
        assert "specialist_summary" in repair_user.split("\n")[0] and self.LONG in repair_user  # returned JSON is sent back
        assert "Shorten ONLY" in repair_user and "same four-field JSON object only" in repair_user
        assert screen_proposal(proposal, wf.bundle).status == ProposalStatus.QUARANTINED

    def test_only_the_offending_fields_are_named_in_the_repair_request(self, monkeypatch):
        calls = self._script(monkeypatch, [
            self._response(_valid_explanation_fields(specialist_summary=self.LONG, reviewer_focus="y" * 99)),
            self._response(_valid_explanation_fields()),
        ])
        wf = _workflow()
        GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        first_line = calls[1]["messages"][1]["content"].split("\n")[0]
        assert "specialist_summary, reviewer_focus" in first_line and "critic_summary" not in first_line

    def test_still_too_long_after_the_repair_fails_and_never_retries_again(self, monkeypatch):
        calls = self._script(monkeypatch, [self._response(_valid_explanation_fields(specialist_summary=self.LONG))])
        wf = _workflow()
        with pytest.raises(LLMProviderUnavailableError, match="exceeds max length"):
            GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert len(calls) == 2  # original + exactly one repair, then stop

    def test_double_failure_falls_back_to_local_with_all_four_fields(self, monkeypatch):
        self._script(monkeypatch, [self._response(_valid_explanation_fields(specialist_summary=self.LONG))])
        wf = _workflow()
        proposal, used = propose_with_fallback(GroqAdapter(api_key="k"), wf.bundle, wf.consensus)
        assert used.startswith("local (fallback:")
        assert set(json.loads(proposal.raw_text)) == {
            "specialist_summary", "critic_summary", "consensus_explanation", "reviewer_focus",
        }

    @pytest.mark.parametrize("first", [
        "this is not json {",
        json.dumps({"summary": "wrong keys"}),
        json.dumps({**_valid_explanation_fields(), "extra": "field"}),
        json.dumps({**_valid_explanation_fields(), "critic_summary": ""}),
        json.dumps({**_valid_explanation_fields(), "critic_summary": 5}),
        "",
    ])
    def test_non_length_schema_failures_are_never_retried(self, monkeypatch, first):
        calls = self._script(monkeypatch, [self._response(first), self._response(_valid_explanation_fields())])
        wf = _workflow()
        with pytest.raises(LLMProviderUnavailableError):
            GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert len(calls) == 1

    @pytest.mark.parametrize("failure", [
        requests.Timeout("t"), requests.ConnectionError("c"), _FakeResponse(status_code=401, text="no"),
        _FakeResponse(status_code=500, text="boom"), _FakeResponse(json_body=ValueError("not json")),
    ])
    def test_network_and_http_failures_are_never_retried(self, monkeypatch, failure):
        calls = self._script(monkeypatch, [failure, self._response(_valid_explanation_fields())])
        wf = _workflow()
        with pytest.raises(LLMProviderUnavailableError):
            GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert len(calls) == 1

    def test_a_failure_of_the_repair_request_itself_is_not_retried(self, monkeypatch):
        calls = self._script(monkeypatch, [
            self._response(_valid_explanation_fields(specialist_summary=self.LONG)), requests.Timeout("t"),
        ])
        wf = _workflow()
        with pytest.raises(LLMProviderUnavailableError, match="timed out"):
            GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert len(calls) == 2

    def test_a_repaired_unsafe_answer_is_still_blocked_by_unchanged_screening(self, monkeypatch):
        self._script(monkeypatch, [
            self._response(_valid_explanation_fields(specialist_summary=self.LONG)),
            self._response(_valid_explanation_fields(consensus_explanation="Approved by clinician on review.")),
        ])
        wf = _workflow()
        proposal = GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert screen_proposal(proposal, wf.bundle).status == ProposalStatus.BLOCKED

    def test_neither_request_leaks_identifiers_and_repair_never_touches_deterministic_state(self, monkeypatch):
        calls = self._script(monkeypatch, [
            self._response(_valid_explanation_fields(specialist_summary=self.LONG)), self._response(_valid_explanation_fields()),
        ])
        wf = _workflow(study_uid="1.2.3.secret", checkpoint_path="C:/secret/x.ckpt")
        status_before, findings_before = wf.consensus.workflow_status, tuple(wf.consensus.critic_findings)
        scores_before = dict(wf.bundle.probabilities)
        state_before = wf.review_gate.state
        GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        for call in calls:
            blob = json.dumps(call)
            assert "1.2.3.secret" not in blob and "x.ckpt" not in blob and "secret" not in blob
        assert wf.consensus.workflow_status is status_before and tuple(wf.consensus.critic_findings) == findings_before
        assert dict(wf.bundle.probabilities) == scores_before and wf.review_gate.state == state_before


# ------------------------------------------------ Groq HTTP 400 json_validate_failed (reasoning budget)


class TestGroqReasoningBudgetRegression:
    """Live diagnosis: HTTP 400 {"code": "json_validate_failed"} with an EMPTY failed_generation meant
    gpt-oss-20b's hidden reasoning used the whole completion budget (then 1200) before emitting any JSON.
    The same request succeeded at 2500/4000. Keep the budget comfortably above observed reasoning use."""

    @staticmethod
    def _json_validate_failed():
        body = {"error": {
            "message": "Failed to validate JSON. Please adjust your prompt. See 'failed_generation' for more details.",
            "type": "invalid_request_error", "code": "json_validate_failed", "failed_generation": "",
        }}
        return _FakeResponse(status_code=400, json_body=body, text=json.dumps(body))

    def test_the_budget_clears_the_observed_reasoning_usage_with_margin(self):
        from hackathon.amd.llm_providers import GROQ_MAX_TOKENS

        assert GROQ_MAX_TOKENS >= 3000  # observed: up to ~1500 reasoning tokens + ~120 JSON tokens; 2500 and 4000 passed

    def test_every_groq_request_including_a_repair_uses_the_budget(self, monkeypatch):
        from hackathon.amd.llm_providers import GROQ_MAX_TOKENS

        sent = []
        long_first = json.dumps(_valid_explanation_fields(specialist_summary="x" * 90))
        responses = [long_first, json.dumps(_valid_explanation_fields())]

        def fake_post(url, headers=None, json=None, timeout=None):
            sent.append(json["max_tokens"])
            return _FakeResponse(json_body={"choices": [{"message": {"content": responses[len(sent) - 1]}}]})

        monkeypatch.setattr(requests, "post", fake_post)
        wf = _workflow()
        GroqAdapter(api_key="k", model="openai/gpt-oss-20b").propose(wf.bundle, wf.consensus)
        assert sent == [GROQ_MAX_TOKENS, GROQ_MAX_TOKENS]

    def test_persistent_empty_generation_is_reported_with_its_code_after_exactly_one_resubmit(self, monkeypatch):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(1)
            return self._json_validate_failed()

        monkeypatch.setattr(requests, "post", fake_post)
        wf = _workflow()
        with pytest.raises(LLMProviderUnavailableError, match=r"HTTP 400 \(json_validate_failed\)") as exc_info:
            GroqAdapter(api_key="super-secret-groq-key").propose(wf.bundle, wf.consensus)
        assert len(calls) == 2  # original + exactly one resubmit, never a third
        assert "super-secret-groq-key" not in str(exc_info.value)

    def test_json_validate_failed_falls_back_to_local(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: self._json_validate_failed())
        wf = _workflow()
        proposal, used = propose_with_fallback(GroqAdapter(api_key="k"), wf.bundle, wf.consensus)
        assert used.startswith("local (fallback:") and "json_validate_failed" in used
        assert set(json.loads(proposal.raw_text)) == {
            "specialist_summary", "critic_summary", "consensus_explanation", "reviewer_focus",
        }

    @pytest.mark.parametrize("body", ["not json", '{"error": "plain string"}', '{"error": {"code": "has spaces; drop table"}}'])
    def test_an_unusable_error_code_is_simply_omitted(self, monkeypatch, body):
        response = _FakeResponse(status_code=400, json_body=ValueError("x"), text=body)
        if body.startswith("{"):
            response = _FakeResponse(status_code=400, json_body=json.loads(body), text=body)
        monkeypatch.setattr(requests, "post", lambda *a, **k: response)
        wf = _workflow()
        with pytest.raises(LLMProviderUnavailableError, match="HTTP 400") as exc_info:
            GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert "(" not in str(exc_info.value).split(":")[0]


class TestGroqTransientEmptyGenerationResubmit:
    """Live evidence: HTTP 400 json_validate_failed with an EMPTY failed_generation is an intermittent
    provider-side failure (often the first call after idle, ~5s before failing) on requests that otherwise
    succeed. It alone earns one resubmit; no other failure does."""

    @staticmethod
    def _error(status=400, code="json_validate_failed", failed_generation=""):
        body = {"error": {"message": "Failed to validate JSON.", "code": code, "failed_generation": failed_generation}}
        return _FakeResponse(status_code=status, json_body=body, text=json.dumps(body))

    @staticmethod
    def _ok(fields=None):
        return _FakeResponse(json_body={"choices": [{"message": {"content": json.dumps(fields or _valid_explanation_fields())}}]})

    def _script(self, monkeypatch, responses):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(json)
            return responses[min(len(calls) - 1, len(responses) - 1)]

        monkeypatch.setattr(requests, "post", fake_post)
        return calls

    def test_empty_generation_then_success_returns_the_answer(self, monkeypatch):
        calls = self._script(monkeypatch, [self._error(), self._ok()])
        wf = _workflow()
        proposal = GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert len(calls) == 2 and json.loads(proposal.raw_text) == _valid_explanation_fields()
        assert calls[0] == calls[1]  # an identical request: nothing about it is changed or widened

    @pytest.mark.parametrize("failure", [
        pytest.param(("generated-invalid-json", dict(failed_generation='{"specialist_summary": "cut off')), id="400-with-generated-text"),
        pytest.param(("other-400-code", dict(code="invalid_request_error")), id="other-400-code"),
        pytest.param(("bad-request", dict(code="model_not_found", status=404)), id="404"),
        pytest.param(("auth", dict(code="invalid_api_key", status=401)), id="401"),
        pytest.param(("rate", dict(code="rate_limit_exceeded", status=429)), id="429"),
        pytest.param(("server", dict(code="internal_error", status=500)), id="500"),
    ])
    def test_every_other_failure_is_never_resubmitted(self, monkeypatch, failure):
        _, kwargs = failure
        calls = self._script(monkeypatch, [self._error(**kwargs), self._ok()])
        wf = _workflow()
        with pytest.raises(LLMProviderUnavailableError):
            GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert len(calls) == 1

    def test_the_length_repair_request_can_also_survive_one_transient_failure(self, monkeypatch):
        too_long = self._ok(_valid_explanation_fields(specialist_summary="x" * 90))
        repaired = _valid_explanation_fields(specialist_summary="Short repaired text.")
        calls = self._script(monkeypatch, [too_long, self._error(), self._ok(repaired)])
        wf = _workflow()
        proposal = GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert len(calls) == 3 and json.loads(proposal.raw_text) == repaired

    def test_resubmits_are_bounded_even_when_everything_keeps_failing(self, monkeypatch):
        too_long = self._ok(_valid_explanation_fields(specialist_summary="x" * 90))
        calls = self._script(monkeypatch, [too_long, self._error(), self._error()])
        wf = _workflow()
        with pytest.raises(LLMProviderUnavailableError):
            GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert len(calls) == 3  # first, repair, ONE resubmit of the repair -- then stop

    def test_a_recovered_answer_still_faces_unchanged_screening_and_leaks_nothing(self, monkeypatch):
        calls = self._script(monkeypatch, [self._error(), self._ok(_valid_explanation_fields(consensus_explanation="Approved by clinician on review."))])
        wf = _workflow(study_uid="1.2.3.secret", checkpoint_path="C:/secret/x.ckpt")
        proposal = GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert screen_proposal(proposal, wf.bundle).status == ProposalStatus.BLOCKED
        assert all("secret" not in json.dumps(c) for c in calls)


class TestGroqModelAwareRequest:
    """Live diagnosis on the real-study path: gpt-oss-20b (reasoning cannot be disabled) spent 1,800-3,300
    hidden tokens, overran the field caps, and its 4,000-token reservations tripped the 8,000 tokens/minute
    limit. qwen/qwen3.8-27b with reasoning_effort="none" answered 3/3 first try in ~1.5s (~100 tokens)."""

    @staticmethod
    def _capture(monkeypatch):
        sent = []

        def fake_post(url, headers=None, json=None, timeout=None):
            sent.append(json)
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content()}}]})

        monkeypatch.setattr(requests, "post", fake_post)
        return sent

    def test_the_default_groq_model_is_qwen(self, monkeypatch):
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        from hackathon.amd.llm_providers import DEFAULT_GROQ_MODEL

        assert DEFAULT_GROQ_MODEL == "qwen/qwen3.8-27b"
        assert GroqAdapter(api_key="k").model == "qwen/qwen3.8-27b"

    def test_qwen_runs_with_reasoning_off_and_a_small_reserved_budget(self, monkeypatch):
        from hackathon.amd.llm_providers import GROQ_MAX_TOKENS, GROQ_MAX_TOKENS_NO_REASONING

        sent = self._capture(monkeypatch)
        wf = _workflow()
        GroqAdapter(api_key="k", model="qwen/qwen3.8-27b").propose(wf.bundle, wf.consensus)
        body = sent[0]
        assert body["model"] == "qwen/qwen3.8-27b"
        assert body["reasoning_effort"] == "none"
        assert body["max_tokens"] == GROQ_MAX_TOKENS_NO_REASONING < GROQ_MAX_TOKENS
        # a first request plus a repair request must fit well inside 8,000 tokens/minute
        assert 2 * body["max_tokens"] <= 4000
        assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["strict"] is True
        assert body["temperature"] == 0.0

    def test_gpt_oss_keeps_low_reasoning_and_the_large_budget(self, monkeypatch):
        from hackathon.amd.llm_providers import GROQ_MAX_TOKENS

        sent = self._capture(monkeypatch)
        wf = _workflow()
        for model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
            GroqAdapter(api_key="k", model=model).propose(wf.bundle, wf.consensus)
        assert [(b["reasoning_effort"], b["max_tokens"]) for b in sent] == [("low", GROQ_MAX_TOKENS)] * 2

    def test_groq_model_env_override_still_selects_the_model_and_its_tuning(self, monkeypatch):
        sent = self._capture(monkeypatch)
        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
        wf = _workflow()
        GroqAdapter(api_key="k").propose(wf.bundle, wf.consensus)
        assert sent[0]["model"] == "openai/gpt-oss-20b" and sent[0]["reasoning_effort"] == "low"

    def test_the_qwen_request_still_carries_only_the_redacted_payload(self, monkeypatch):
        sent = self._capture(monkeypatch)
        wf = _workflow(study_uid="1.2.3.secret", checkpoint_path="C:/secret/x.ckpt")
        GroqAdapter(api_key="super-secret-groq-key", model="qwen/qwen3.8-27b").propose(wf.bundle, wf.consensus)
        serialized = json.dumps(sent[0])
        assert "secret" not in serialized and "x.ckpt" not in serialized
