"""Execution-provider abstraction: local CPU/GPU (default) vs. an optional AMD Token Factory.

The AMD Token Factory provider is disabled by default and performs no network I/O, requires
no API key, and is never silently substituted for local execution. Importing this module, or
constructing either provider with default arguments, never touches the network. This exists
so the Streamlit app can offer a "run on AMD cloud" toggle in its UI without that toggle doing
anything until it is both explicitly enabled and fully configured -- and even then, this
adapter has no real Token Factory HTTP client, so enabling it reports "not implemented," never
a fabricated success.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .rocm_env import check_environment


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    available: bool
    is_amd_rocm: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "available": self.available, "is_amd_rocm": self.is_amd_rocm, "detail": self.detail}


class InferenceProvider:
    name = "base"

    def status(self) -> ProviderStatus:
        raise NotImplementedError

    def device(self) -> str:
        raise NotImplementedError


class LocalProvider(InferenceProvider):
    """Always available: runs on this machine's own torch device. Reports AMD ROCm only when
    hackathon.amd.rocm_env.check_environment() actually confirms it -- never because a GPU
    device string was requested."""

    name = "local"

    def __init__(self, preferred_device: str = "cpu"):
        self._preferred_device = preferred_device

    def status(self) -> ProviderStatus:
        env = check_environment()
        if self._preferred_device != "cpu" and not env["accelerator_available"]:
            reasons = "; ".join(env["unavailable_reasons"]) or "no accelerator detected"
            return ProviderStatus(self.name, True, False, f"No local accelerator; falling back to cpu ({reasons}).")
        return ProviderStatus(self.name, True, bool(env["amd_gpu_available"]), "Local execution, always available.")

    def device(self) -> str:
        env = check_environment()
        if self._preferred_device != "cpu" and env["accelerator_available"]:
            return self._preferred_device
        return "cpu"


class AMDTokenFactoryProvider(InferenceProvider):
    """Optional remote AMD Developer Cloud / Token Factory execution.

    `enabled=False` (the default) means status()/device() never attempt any network call and
    never require AMD_TOKENFACTORY_BASE_URL/AMD_TOKENFACTORY_API_KEY to be set. There is
    currently no real HTTP client behind this provider -- even when enabled and fully
    configured, device() raises NotImplementedError rather than pretending to dispatch work
    to AMD hardware. This is the explicit, documented seam for a future real integration.
    """

    name = "amd_token_factory"

    def __init__(self, *, enabled: bool = False, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.enabled = enabled
        self.base_url = base_url or os.environ.get("AMD_TOKENFACTORY_BASE_URL")
        self.api_key = api_key or os.environ.get("AMD_TOKENFACTORY_API_KEY")

    def status(self) -> ProviderStatus:
        if not self.enabled:
            return ProviderStatus(self.name, False, False, "Disabled by default; not used for local operation.")
        if not self.base_url or not self.api_key:
            return ProviderStatus(
                self.name, False, False,
                "Enabled but not configured (missing base URL and/or API key); no network call attempted.",
            )
        return ProviderStatus(
            self.name, False, False,
            "Configured, but this adapter has no Token Factory HTTP client yet; no AMD GPU result "
            "has ever been produced through this provider.",
        )

    def device(self) -> str:
        raise NotImplementedError(
            "AMD Token Factory execution is not implemented in this adapter; call status() instead."
        )


def default_provider() -> InferenceProvider:
    """The provider used when the app starts: always local, never the optional AMD provider."""
    return LocalProvider()
