"""Environment-based configuration for the optional LLM reasoning layer (hackathon/amd/llm_providers.py).

See `.env.example` at the repo root for every variable this module reads.

Precedence (highest wins): the real process environment > an optional local `.env` file
(loaded without ever overriding a variable the process environment already has) > built-in
defaults (provider "local", external calls disabled).

The one hard safety rule this module exists to enforce, end to end:

    An ambient GROQ_API_KEY (or any other credential) sitting in the environment is NEVER,
    by itself, sufficient to authorize a live call to Groq or AMD Token Factory. That always
    additionally requires LLM_PROVIDER to name that provider AND LLM_ALLOW_EXTERNAL_CALLS to
    be explicitly "true". `build_adapter()` is the single choke point that enforces this --
    callers (the Streamlit app, tests, any future caller) should go through it rather than
    constructing GroqAdapter/AMDTokenFactoryLLMAdapter directly, which is how a real ambient
    GROQ_API_KEY was able to authorize one live call during development of this feature.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .llm_providers import AMDTokenFactoryLLMAdapter, GroqAdapter, LocalDeterministicAdapter

_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})
_KNOWN_PROVIDERS = frozenset({"local", "groq", "amd_token_factory"})
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DOTENV_PATH = _REPO_ROOT / ".env"


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_STRINGS


def load_dotenv_if_present(dotenv_path: Optional[Path] = None) -> bool:
    """Load a local `.env` file into `os.environ`, never overriding a variable the real
    process environment already has (python-dotenv's own `override=False`, which is the
    semantic this function relies on rather than reimplements). A missing `python-dotenv`
    package or a missing `.env` file are both silently fine -- this never raises either way.
    Returns True only if a `.env` file was actually found and loaded.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    path = dotenv_path if dotenv_path is not None else _DEFAULT_DOTENV_PATH
    if not path.is_file():
        return False
    load_dotenv(dotenv_path=path, override=False)
    return True


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    allow_external_calls: bool
    groq_api_key: Optional[str]
    groq_model: Optional[str]
    amd_enabled: bool
    amd_base_url: Optional[str]
    amd_api_key: Optional[str]
    dotenv_loaded: bool

    def to_dict(self) -> dict:
        """Never includes a raw credential value -- only whether one is present."""
        return {
            "provider": self.provider,
            "allow_external_calls": self.allow_external_calls,
            "groq_api_key_present": bool(self.groq_api_key),
            "groq_model": self.groq_model,
            "amd_enabled": self.amd_enabled,
            "amd_base_url_present": bool(self.amd_base_url),
            "amd_api_key_present": bool(self.amd_api_key),
            "dotenv_loaded": self.dotenv_loaded,
        }


def resolve_llm_config(*, use_dotenv: bool = True, dotenv_path: Optional[Path] = None) -> LLMConfig:
    """Resolve the effective configuration right now, from the real environment first and
    optionally a `.env` file for anything still unset. Always returns a value -- never raises
    for a missing or malformed environment; an unrecognized LLM_PROVIDER value is treated the
    same as "local" (fail safe, not fail open)."""
    dotenv_loaded = load_dotenv_if_present(dotenv_path) if use_dotenv else False

    provider = os.environ.get("LLM_PROVIDER", "local").strip().lower() or "local"
    if provider not in _KNOWN_PROVIDERS:
        provider = "local"

    return LLMConfig(
        provider=provider,
        allow_external_calls=_env_bool("LLM_ALLOW_EXTERNAL_CALLS", default=False),
        groq_api_key=os.environ.get("GROQ_API_KEY") or None,
        groq_model=os.environ.get("GROQ_MODEL") or None,
        amd_enabled=_env_bool("AMD_TOKENFACTORY_ENABLED", default=False),
        amd_base_url=os.environ.get("AMD_TOKENFACTORY_BASE_URL") or None,
        amd_api_key=os.environ.get("AMD_TOKENFACTORY_API_KEY") or None,
        dotenv_loaded=dotenv_loaded,
    )


def build_adapter(config: LLMConfig):
    """The single choke point for turning a resolved LLMConfig into a real adapter instance.

    Returns (adapter, reason) where `reason` is always a short, human-readable explanation
    suitable for display in the UI -- including, especially, *why* the result is
    LocalDeterministicAdapter when the caller may have expected something else.

    If `config.allow_external_calls` is False, this ALWAYS returns LocalDeterministicAdapter,
    regardless of `provider` or whether a credential is present -- this is the enforcement
    point for "an ambient GROQ_API_KEY alone cannot authorize a live call."
    """
    if not config.allow_external_calls:
        return LocalDeterministicAdapter(), "LLM_ALLOW_EXTERNAL_CALLS is not enabled; using local deterministic mode."

    if config.provider == "groq":
        if not config.groq_api_key:
            return LocalDeterministicAdapter(), "LLM_PROVIDER=groq but GROQ_API_KEY is not set; using local deterministic mode."
        model = config.groq_model
        return (
            GroqAdapter(api_key=config.groq_api_key, model=model),
            f"Using Groq (model={model or 'default'}).",
        )

    if config.provider == "amd_token_factory":
        return (
            AMDTokenFactoryLLMAdapter(enabled=config.amd_enabled, base_url=config.amd_base_url, api_key=config.amd_api_key),
            "Using AMD Token Factory -- not implemented; this adapter's propose() always raises "
            "NotImplementedError, so the caller's own fallback logic will produce the final result.",
        )

    return LocalDeterministicAdapter(), "LLM_PROVIDER is 'local' (or unrecognized); using local deterministic mode."
