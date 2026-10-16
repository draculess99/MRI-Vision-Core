"""Tests for hackathon/amd/providers.py: the local (default) vs. optional AMD Token Factory
execution-provider interface. The core requirement under test is that the AMD provider is
disabled by default, never requires credentials or network access for local operation, and
never claims AMD ROCm availability it did not measure.
"""

import socket

import pytest

from hackathon.amd.providers import AMDTokenFactoryProvider, LocalProvider, default_provider


class _NoNetworkGuard:
    """Context manager that fails the test if anything tries to open a real socket."""

    def __enter__(self):
        self._original = socket.socket.__init__

        def _blocked(*args, **kwargs):
            raise AssertionError("A socket was constructed; no network access is allowed here.")

        socket.socket.__init__ = _blocked
        return self

    def __exit__(self, *exc_info):
        socket.socket.__init__ = self._original


class TestDefaultProvider:
    def test_default_provider_is_local(self):
        assert isinstance(default_provider(), LocalProvider)

    def test_local_provider_is_always_available(self):
        status = LocalProvider().status()
        assert status.available is True
        assert status.name == "local"

    def test_local_provider_only_claims_amd_rocm_when_environment_confirms_it(self):
        from hackathon.amd.rocm_env import check_environment

        status = LocalProvider().status()
        assert status.is_amd_rocm == check_environment()["amd_gpu_available"]

    def test_local_provider_device_falls_back_to_cpu_without_accelerator(self):
        from hackathon.amd.rocm_env import check_environment

        provider = LocalProvider(preferred_device="cuda:0")
        if not check_environment()["accelerator_available"]:
            assert provider.device() == "cpu"


class TestAMDTokenFactoryProviderDisabledByDefault:
    def test_disabled_by_default(self):
        provider = AMDTokenFactoryProvider()
        assert provider.enabled is False

    def test_disabled_provider_never_requires_credentials(self, monkeypatch):
        monkeypatch.delenv("AMD_TOKENFACTORY_BASE_URL", raising=False)
        monkeypatch.delenv("AMD_TOKENFACTORY_API_KEY", raising=False)
        status = AMDTokenFactoryProvider().status()
        assert status.available is False
        assert status.is_amd_rocm is False
        assert "Disabled by default" in status.detail

    def test_disabled_provider_makes_no_network_call(self, monkeypatch):
        monkeypatch.delenv("AMD_TOKENFACTORY_BASE_URL", raising=False)
        monkeypatch.delenv("AMD_TOKENFACTORY_API_KEY", raising=False)
        with _NoNetworkGuard():
            AMDTokenFactoryProvider().status()

    def test_enabled_without_credentials_reports_not_configured_without_network_call(self, monkeypatch):
        monkeypatch.delenv("AMD_TOKENFACTORY_BASE_URL", raising=False)
        monkeypatch.delenv("AMD_TOKENFACTORY_API_KEY", raising=False)
        with _NoNetworkGuard():
            status = AMDTokenFactoryProvider(enabled=True).status()
        assert status.available is False
        assert "not configured" in status.detail

    def test_enabled_and_configured_still_never_claims_a_real_amd_result(self, monkeypatch):
        with _NoNetworkGuard():
            status = AMDTokenFactoryProvider(
                enabled=True, base_url="https://example.invalid", api_key="dummy-key",
            ).status()
        assert status.available is False
        assert status.is_amd_rocm is False
        assert "no AMD GPU result has ever been produced" in status.detail

    def test_device_raises_not_implemented_even_when_enabled(self):
        provider = AMDTokenFactoryProvider(enabled=True, base_url="https://example.invalid", api_key="k")
        with pytest.raises(NotImplementedError):
            provider.device()
