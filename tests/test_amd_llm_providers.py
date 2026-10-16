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


def _structured_content(summary: str) -> str:
    """Build the `message.content` string Groq's structured-output mode (response_format=
    json_schema) actually returns: a JSON-encoded object matching _REVIEWER_RESULT_RESPONSE_FORMAT."""
    return json.dumps({"summary": summary})


class TestGroqSuccess:
    def test_successful_response_produces_a_proposal(self, monkeypatch):
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content("Short safe summary.")}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        proposal = GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)
        assert proposal.raw_text == "Short safe summary."
        assert proposal.cited_evidence == ()

    def test_request_body_contains_only_the_redacted_payload(self, monkeypatch):
        wf = _workflow(study_uid="1.2.3.secret", checkpoint_path="C:/secret/x.ckpt")
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content("ok")}}]})

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
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content("ok")}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)

        assert captured["json"]["reasoning_effort"] == "low"
        assert captured["json"]["temperature"] == 0.0
        assert captured["json"]["max_tokens"] == 800
        assert captured["json"]["response_format"]["type"] == "json_schema"
        assert captured["json"]["response_format"]["json_schema"]["strict"] is True

    def test_reasoning_field_is_never_read_as_the_answer(self, monkeypatch):
        """A reasoning model's hidden reasoning trace must never be treated as the reviewed
        answer, even if present alongside a well-formed structured message.content."""
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body={
                "choices": [{"message": {
                    "reasoning": "This is internal chain-of-thought and must never be used.",
                    "content": _structured_content("The actual reviewed summary."),
                }}]
            })

        monkeypatch.setattr(requests, "post", _fake_post)
        proposal = GroqAdapter(api_key="fake-key").propose(wf.bundle, wf.consensus)
        assert proposal.raw_text == "The actual reviewed summary."
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
            {"choices": [{"message": {"content": '{"not_summary": "x"}'}}]},
            {"choices": [{"message": {"content": '{"summary": "x", "extra": "y"}'}}]},
            {"choices": [{"message": {"content": '{"summary": 123}'}}]},
            {"choices": [{"message": {"content": '{"summary": ""}'}}]},
            {"choices": [{"message": {"content": "[1, 2, 3]"}}]},
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
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content(unsafe_text)}}]})

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
        wf = _workflow()

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content("AMD summary.")}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        proposal = _amd_adapter().propose(wf.bundle, wf.consensus)
        assert proposal.raw_text == "AMD summary."
        assert proposal.cited_evidence == ()

    def test_request_uses_the_verified_contract(self, monkeypatch):
        wf = _workflow()
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content("ok")}}]})

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
            return _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content("ok")}}]})

        monkeypatch.setattr(requests, "post", _fake_post)
        _amd_adapter().propose(wf.bundle, wf.consensus)

        serialized_request = json.dumps(captured["json"])
        assert "1.2.3.secret" not in serialized_request
        assert "secret" not in serialized_request
        assert "x.ckpt" not in serialized_request

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

        def _fake_post(*args, **kwargs):
            return _FakeResponse(json_body={
                "choices": [{"message": {
                    "reasoning": "hidden chain-of-thought, must never be used",
                    "content": _structured_content("the real reviewed summary"),
                }}]
            })

        monkeypatch.setattr(requests, "post", _fake_post)
        proposal = _amd_adapter().propose(wf.bundle, wf.consensus)
        assert proposal.raw_text == "the real reviewed summary"
        assert "chain-of-thought" not in proposal.raw_text


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
            {"choices": [{"message": {"content": '{"not_summary": "x"}'}}]},
            {"choices": [{"message": {"content": '{"summary": 123}'}}]},
            {"choices": [{"message": {"content": '{"summary": "x", "extra": "y"}'}}]},
        ],
    )
    def test_malformed_or_unexpected_schema_is_rejected(self, monkeypatch, bad_body):
        wf = _workflow()
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(json_body=bad_body))
        with pytest.raises(LLMProviderUnavailableError):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    def test_oversized_summary_is_rejected(self, monkeypatch):
        wf = _workflow()
        oversized = _structured_content("x" * 3000)
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": oversized}}]}))
        with pytest.raises(LLMProviderUnavailableError, match="exceeds max length"):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    def test_the_real_619_character_response_that_was_once_blocked_is_now_rejected_earlier(self, monkeypatch):
        """Regression test for a real live-call failure: AMD Token Factory once returned a
        619-character summary that reached screen_proposal() and came back opaquely BLOCKED
        (length). This adapter must now reject it itself, before screen_proposal() ever sees
        it, with a specific, diagnosable error -- same 500-char threshold, not a looser one."""
        wf = _workflow()
        real_failure_text = (
            "The workflow is marked MORE_EVIDENCE_REQUIRED, and the single research model produced "
            "uncalibrated scores that are numerically higher for the osteoarthritis, effusion, and "
            "synovitis categories than for the ligament and meniscus categories, though none of these "
            "values should be interpreted as a finding or a probability of an abnormality being present. "
            "The quality gate has not yet been assessed for this study, which is one of the stated "
            "limitations, and no ensemble of models was used to produce this single result, so the "
            "overall evidence remains incomplete pending further structured review by a qualified reviewer."
        )
        assert len(real_failure_text) > 500  # matches the real observed length (619 chars)
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content(real_failure_text)}}]}),
        )
        with pytest.raises(LLMProviderUnavailableError, match="exceeds max length 500"):
            _amd_adapter().propose(wf.bundle, wf.consensus)

    def test_summary_at_exactly_the_shared_cap_is_accepted(self, monkeypatch):
        """The adapter-level cap must match, not undercut, mri_core's own MAX_PROPOSAL_LENGTH --
        a summary of exactly that length must still pass through to screen_proposal()."""
        from mri_core.agents.llm_adapter import MAX_PROPOSAL_LENGTH

        wf = _workflow()
        exactly_at_cap = "x" * MAX_PROPOSAL_LENGTH
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content(exactly_at_cap)}}]}),
        )
        proposal = _amd_adapter().propose(wf.bundle, wf.consensus)
        assert proposal.raw_text == exactly_at_cap
        outcome = screen_proposal(proposal, wf.bundle)
        assert outcome.status == ProposalStatus.QUARANTINED

    def test_short_valid_summary_is_quarantined_not_approved(self, monkeypatch):
        """A short, safe, well-formed summary must still only ever reach QUARANTINED --
        never auto-approved into verified evidence, regardless of how clean it looks."""
        wf = _workflow()
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content("Workflow data summary only.")}}]}),
        )
        proposal = _amd_adapter().propose(wf.bundle, wf.consensus)
        outcome = screen_proposal(proposal, wf.bundle)
        assert outcome.status == ProposalStatus.QUARANTINED
        assert "never published as verified evidence" in outcome.reason


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
            lambda *a, **k: _FakeResponse(json_body={"choices": [{"message": {"content": _structured_content(unsafe_text)}}]}),
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
