"""Safety net: no test may reach AssemblyAI or Amazon Polly, or see real credentials."""

import pytest

# Real implementations, for tests that exercise them with the SDK itself mocked.
ORIGINALS = {}


@pytest.fixture(autouse=True)
def _block_paid_voice_services(monkeypatch):
    def blocked(*_a, **_k):
        raise AssertionError("A test attempted a real paid cloud call (AssemblyAI/Polly).")

    for var in ("ASSEMBLYAI_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    try:
        from hackathon.amazon_alexa import stt, tts
    except ImportError:
        return
    ORIGINALS.setdefault("assemblyai_transcribe", stt.assemblyai_transcribe)
    monkeypatch.setattr(stt, "assemblyai_transcribe", blocked)
    monkeypatch.setattr(tts, "polly_synthesize", blocked)
