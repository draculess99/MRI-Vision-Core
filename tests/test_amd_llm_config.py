"""Tests for hackathon/amd/llm_config.py: environment/.env-driven configuration for the
optional LLM reasoning layer, and the hard safety gate it enforces.

Every test explicitly clears the credential/config environment variables this project's own
shell may already have set (a real GROQ_API_KEY is present in this shell's ambient
environment during development) -- nothing here ever relies on, or risks acting on, that
ambient state.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
requests = pytest.importorskip("requests")

from hackathon.amd.llm_config import LLMConfig, build_adapter, load_dotenv_if_present, resolve_llm_config
from hackathon.amd.llm_providers import AMDTokenFactoryLLMAdapter, GroqAdapter, LocalDeterministicAdapter

_ALL_LLM_ENV_VARS = (
    "LLM_PROVIDER", "LLM_ALLOW_EXTERNAL_CALLS", "GROQ_API_KEY", "GROQ_MODEL",
    "AMD_TOKENFACTORY_ENABLED", "AMD_TOKENFACTORY_BASE_URL", "AMD_TOKENFACTORY_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ALL_LLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class _NoNetworkGuard:
    def __enter__(self):
        self._original = requests.post

        def _blocked(*args, **kwargs):
            raise AssertionError("requests.post was called; no network access is allowed here.")

        requests.post = _blocked
        return self

    def __exit__(self, *exc_info):
        requests.post = self._original


# --------------------------------------------------------------------------- defaults


class TestDefaults:
    def test_defaults_are_local_and_disallowed(self):
        config = resolve_llm_config(use_dotenv=False)
        assert config.provider == "local"
        assert config.allow_external_calls is False
        assert config.groq_api_key is None
        assert config.amd_enabled is False

    def test_to_dict_never_includes_raw_credential_values(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "super-secret-value")
        monkeypatch.setenv("AMD_TOKENFACTORY_API_KEY", "another-secret")
        config = resolve_llm_config(use_dotenv=False)
        serialized = str(config.to_dict())
        assert "super-secret-value" not in serialized
        assert "another-secret" not in serialized
        assert config.to_dict()["groq_api_key_present"] is True
        assert config.to_dict()["amd_api_key_present"] is True


# ------------------------------------------------------------------ the hard safety gate


class TestAmbientCredentialAloneCannotAuthorize:
    def test_groq_api_key_alone_does_not_select_groq_adapter(self, monkeypatch):
        """The exact incident this feature exists to prevent: a real-looking GROQ_API_KEY in
        the environment, with nothing else set, must never produce a live-capable adapter."""
        monkeypatch.setenv("GROQ_API_KEY", "sk-fake-looking-real-key-123456")
        config = resolve_llm_config(use_dotenv=False)
        with _NoNetworkGuard():
            adapter, reason = build_adapter(config)
        assert isinstance(adapter, LocalDeterministicAdapter)
        assert "LLM_ALLOW_EXTERNAL_CALLS" in reason

    def test_llm_provider_groq_alone_without_allow_flag_is_still_blocked(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        config = resolve_llm_config(use_dotenv=False)
        assert config.allow_external_calls is False
        with _NoNetworkGuard():
            adapter, _reason = build_adapter(config)
        assert isinstance(adapter, LocalDeterministicAdapter)

    def test_allow_flag_alone_without_a_credential_is_also_blocked(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        config = resolve_llm_config(use_dotenv=False)
        with _NoNetworkGuard():
            adapter, reason = build_adapter(config)
        assert isinstance(adapter, LocalDeterministicAdapter)
        assert "GROQ_API_KEY" in reason

    @pytest.mark.parametrize("truthy", ["true", "True", "TRUE", "1", "yes", "on"])
    def test_both_set_together_is_sufficient(self, monkeypatch, truthy):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", truthy)
        config = resolve_llm_config(use_dotenv=False)
        adapter, _reason = build_adapter(config)
        assert isinstance(adapter, GroqAdapter)

    @pytest.mark.parametrize("falsy", ["false", "False", "0", "no", "off", "garbage", ""])
    def test_non_truthy_values_are_treated_as_disallowed(self, monkeypatch, falsy):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", falsy)
        config = resolve_llm_config(use_dotenv=False)
        assert config.allow_external_calls is False


# --------------------------------------------------------------------- provider/model selection


class TestProviderAndModelSelection:
    def test_unrecognized_provider_falls_back_to_local_not_an_error(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "some-future-provider-not-yet-supported")
        config = resolve_llm_config(use_dotenv=False)
        assert config.provider == "local"

    def test_provider_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "GROQ")
        config = resolve_llm_config(use_dotenv=False)
        assert config.provider == "groq"

    def test_groq_model_is_configurable_without_code_change(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setenv("GROQ_MODEL", "some-other-model-name")
        config = resolve_llm_config(use_dotenv=False)
        adapter, _reason = build_adapter(config)
        assert isinstance(adapter, GroqAdapter)
        assert adapter.model == "some-other-model-name"

    def test_amd_token_factory_selected_and_enabled_still_never_claims_to_work(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "amd_token_factory")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setenv("AMD_TOKENFACTORY_ENABLED", "true")
        monkeypatch.setenv("AMD_TOKENFACTORY_BASE_URL", "https://example.invalid")
        monkeypatch.setenv("AMD_TOKENFACTORY_API_KEY", "fake-amd-key")
        config = resolve_llm_config(use_dotenv=False)
        adapter, reason = build_adapter(config)
        assert isinstance(adapter, AMDTokenFactoryLLMAdapter)
        assert "not implemented" in reason
        with pytest.raises(NotImplementedError):
            adapter.propose(None, None)


# -------------------------------------------------------------------- invalid-model fallback


class TestInvalidGroqModelFallsBackCleanly:
    def test_model_not_found_response_raises_a_catchable_unavailable_error(self, monkeypatch):
        from hackathon.amd.agent_bridge_adapter import to_study_inference_result
        from hackathon.amd.kaggle_mil_bridge import TARGET_COLUMNS
        from hackathon.amd.llm_providers import LLMProviderUnavailableError, propose_with_fallback
        from mri_core.agents.contracts import QualityGateStatus
        from mri_core.agents.orchestrator import process_inference_result

        class _NotFoundResponse:
            status_code = 404
            text = '{"error": {"message": "model does not exist", "code": "model_not_found"}}'

            def json(self):
                return {"error": {"message": "model does not exist", "code": "model_not_found"}}

        monkeypatch.setattr(requests, "post", lambda *a, **k: _NotFoundResponse())
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setenv("GROQ_MODEL", "a-model-that-does-not-exist")

        config = resolve_llm_config(use_dotenv=False)
        adapter, _reason = build_adapter(config)

        probs = {col: 0.1 for col in TARGET_COLUMNS}
        result = to_study_inference_result(
            study_uid="1.2.826.0.1.3680043.8.498.0000000000000000001", probabilities=probs, latency_seconds=0.1,
            checkpoint_path="fold0-knee-mil-best.ckpt", checkpoint_sha256="x", disclaimer="d",
        )
        wf = process_inference_result(result, quality_gate_status=QualityGateStatus.NOT_ASSESSED)

        with pytest.raises(LLMProviderUnavailableError, match="404"):
            adapter.propose(wf.bundle, wf.consensus)

        # The caller-facing fallback path turns this into a clean local result, never a crash.
        proposal, used = propose_with_fallback(adapter, wf.bundle, wf.consensus)
        assert used.startswith("local (fallback:")
        assert "404" in used


# ----------------------------------------------------------------------------- .env loading


class TestDotenvLoading:
    def test_missing_dotenv_file_is_silently_fine(self, tmp_path):
        assert load_dotenv_if_present(tmp_path / "does_not_exist.env") is False

    def test_dotenv_file_fills_in_unset_variables(self, tmp_path, monkeypatch):
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("LLM_PROVIDER=groq\nGROQ_API_KEY=from-dotenv-file\n", encoding="utf-8")
        loaded = load_dotenv_if_present(dotenv_path)
        assert loaded is True
        assert __import__("os").environ.get("GROQ_API_KEY") == "from-dotenv-file"
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

    def test_real_process_environment_always_wins_over_dotenv_file(self, tmp_path, monkeypatch):
        """Configuration precedence: a real environment variable already set must never be
        overwritten by a value from a .env file."""
        monkeypatch.setenv("GROQ_API_KEY", "from-real-environment")
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("GROQ_API_KEY=from-dotenv-file\n", encoding="utf-8")

        config = resolve_llm_config(use_dotenv=True, dotenv_path=dotenv_path)
        assert config.groq_api_key == "from-real-environment"
        assert config.dotenv_loaded is True

    def test_dotenv_reports_loaded_false_when_use_dotenv_is_false(self, tmp_path):
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("LLM_PROVIDER=groq\n", encoding="utf-8")
        config = resolve_llm_config(use_dotenv=False, dotenv_path=dotenv_path)
        assert config.dotenv_loaded is False
        assert config.provider == "local"


# ----------------------------------------------------------------------- deterministic fallback


class TestDeterministicFallbackIsAlwaysAvailable:
    def test_build_adapter_never_raises_regardless_of_config(self, monkeypatch):
        """Every combination of (provider, allow_external_calls, credentials) must produce
        *some* adapter without raising -- the gate fails safe, never fails open or crashes."""
        scenarios = [
            {}, {"LLM_PROVIDER": "groq"}, {"LLM_PROVIDER": "amd_token_factory"},
            {"LLM_ALLOW_EXTERNAL_CALLS": "true"},
            {"LLM_PROVIDER": "groq", "LLM_ALLOW_EXTERNAL_CALLS": "true"},
            {"LLM_PROVIDER": "nonsense", "LLM_ALLOW_EXTERNAL_CALLS": "true"},
        ]
        for env in scenarios:
            for var in _ALL_LLM_ENV_VARS:
                monkeypatch.delenv(var, raising=False)
            for key, value in env.items():
                monkeypatch.setenv(key, value)
            config = resolve_llm_config(use_dotenv=False)
            with _NoNetworkGuard():
                adapter, reason = build_adapter(config)
            assert adapter is not None
            assert isinstance(reason, str) and reason
