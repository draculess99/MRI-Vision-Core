"""Tests for the Alexa-style voice layer. No credentials, network, or paid calls."""

import io
import wave

import pytest

from hackathon.amazon_alexa import stt, tts
from hackathon.amazon_alexa.tools import (
    ApprovalState,
    analyze_mri_evidence,
    explain_review_status,
)
from hackathon.amazon_alexa.usage import UsageLimits, UsageTracker
from hackathon.amazon_alexa.voice_router import Intent, VoiceSession, classify, route_command

CASES = ["case-001-consistent", "case-002-change", "case-003-uncertain"]
FAKE_KEY = "sk-test-SECRET-123"


def wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


class Boom(AssertionError):
    pass


def no_cloud(*_a, **_k):
    raise Boom("a paid/cloud call was made")


# ---------- 1. routing ----------

@pytest.mark.parametrize(
    "text,intent",
    [
        ("Analyze this MRI.", Intent.ANALYZE),
        ("Explain the review status", Intent.EXPLAIN),
        ("What changed between the images?", Intent.CHANGES),
        ("Generate an evidence report", Intent.REPORT),
        ("Request human review", Intent.REQUEST_REVIEW),
        ("What is the approval status?", Intent.APPROVAL_STATUS),
    ],
)
def test_supported_commands_route(text, intent):
    assert classify(text) == intent


@pytest.mark.parametrize("case_id", CASES)
def test_responses_derive_from_tool_outputs(case_id):
    ev = analyze_mri_evidence(case_id)
    ex = explain_review_status(ev)
    s = VoiceSession()
    r = route_command("Analyze this MRI", case_id, s)
    assert ev.case_id in r.spoken and f"{ev.quality_score:.2f}" in r.spoken
    r = route_command("Explain the review status", case_id, s)
    assert ex.summary in r.spoken
    r = route_command("What changed between the images?", case_id, s)
    assert ex.change_notes[0] in r.spoken
    r = route_command("Generate an evidence report", case_id, s)
    assert ev.status.value in r.display and "Non-diagnostic" in r.display
    assert ex.change_notes[0] in r.display


# ---------- 7. unknown ----------

@pytest.mark.parametrize("text", ["", "   ", "play some jazz", "what's the weather"])
def test_unknown_commands(text):
    r = route_command(text, "case-001-consistent", VoiceSession())
    assert r.intent == Intent.UNKNOWN and not r.ok
    assert "Supported commands" in r.display


# ---------- 8. human approval protection ----------

@pytest.mark.parametrize(
    "text",
    ["Approve this MRI", "approve it", "Sign off on this case", "Mark this as approved",
     "authorize the case", "looks good, accept it", "give it the green light"],
)
def test_voice_cannot_approve(text):
    s = VoiceSession()
    r = route_command(text, "case-002-change", s, reviewer_name="Dr. Smith")
    assert r.intent == Intent.APPROVAL_REFUSED and not r.ok
    assert r.approval_state == ApprovalState.PENDING
    assert s.requests == {}  # not even a review request is created
    assert any(e["event"] == "approval_refused" for e in s.audit)
    after = route_command("What is the approval status?", "case-002-change", s)
    assert ApprovalState.APPROVED.value not in after.spoken


def test_request_review_requires_named_reviewer():
    s = VoiceSession()
    r = route_command("Request human review", "case-001-consistent", s, reviewer_name="  ")
    assert not r.ok and "named human reviewer" in r.display
    assert s.requests == {}


def test_request_review_stays_pending_and_is_audited():
    s = VoiceSession()
    r = route_command("Request human review", "case-001-consistent", s, "Dr. Smith", "check")
    assert r.ok and r.approval_state == ApprovalState.PENDING
    req = s.requests["case-001-consistent"]
    assert req.reviewer_name == "Dr. Smith" and req.approval_state == ApprovalState.PENDING
    assert any(e["event"] == "review_requested" and e["reviewer"] == "Dr. Smith" for e in s.audit)
    status = route_command("What is the approval status?", "case-001-consistent", s)
    assert "PENDING_HUMAN_APPROVAL" in status.spoken and "Dr. Smith" in status.spoken
    assert "APPROVED_BY_HUMAN" not in status.spoken


def test_reviewer_name_in_speech_is_not_used():
    s = VoiceSession()
    r = route_command("Request human review from Dr. Jones", "case-001-consistent", s)
    assert not r.ok and s.requests == {}


# ---------- 9. existing tool compatibility ----------

def test_existing_tools_unchanged_and_importable():
    ev = analyze_mri_evidence("case-002-change")
    assert ev.status.value == "NEEDS_REVIEW"
    assert explain_review_status(ev).requires_human_approval is True


# ---------- 2. AssemblyAI (mocked) ----------

def test_assemblyai_success_mocked():
    calls = []
    usage = UsageTracker()
    r = stt.transcribe(
        wav_bytes(2), stt.BACKEND_ASSEMBLYAI, usage,
        env={stt.ENV_ASSEMBLYAI_KEY: FAKE_KEY},
        cloud_fn=lambda k, a: calls.append(k) or "  analyze this MRI ",
        local_fn=no_cloud, cloud_confirmed=True)
    assert r.ok and r.text == "analyze this MRI" and r.is_cloud and not r.fallback_used
    assert calls == [FAKE_KEY]
    assert usage.assemblyai_calls == 1 and usage.assemblyai_audio_seconds == pytest.approx(2.0)


# ---------- 3. local whisper + fallback ----------

def test_local_whisper_default_mocked():
    usage = UsageTracker()
    r = stt.transcribe(wav_bytes(), usage=usage, cloud_fn=no_cloud, local_fn=lambda a: "explain")
    assert r.ok and r.backend == stt.BACKEND_LOCAL and not r.is_cloud
    assert usage.local_transcriptions == 1 and usage.assemblyai_calls == 0


def test_local_unavailable_reports_error():
    def missing(_a):
        raise stt.BackendUnavailable("faster-whisper is not installed")

    r = stt.transcribe(wav_bytes(), stt.BACKEND_LOCAL, local_fn=missing)
    assert not r.ok and "not installed" in r.error


def test_cloud_failure_falls_back_to_local_and_redacts_key():
    def fail(key, _a):
        raise ConnectionError(f"network down for {key}")

    r = stt.transcribe(
        wav_bytes(), stt.BACKEND_ASSEMBLYAI, UsageTracker(),
        env={stt.ENV_ASSEMBLYAI_KEY: FAKE_KEY}, cloud_fn=fail, local_fn=lambda a: "analyze", cloud_confirmed=True)
    assert r.ok and r.fallback_used and r.backend == stt.BACKEND_LOCAL and not r.is_cloud


def test_cloud_failure_without_local_redacts_key():
    def fail(key, _a):
        raise ConnectionError(f"bad credit for {key}")

    def missing(_a):
        raise stt.BackendUnavailable("no whisper")

    r = stt.transcribe(
        wav_bytes(), stt.BACKEND_ASSEMBLYAI, UsageTracker(),
        env={stt.ENV_ASSEMBLYAI_KEY: FAKE_KEY}, cloud_fn=fail, local_fn=missing, cloud_confirmed=True)
    assert not r.ok and FAKE_KEY not in r.error and "[redacted]" in r.error
    assert "no whisper" in r.error


def test_fallback_can_be_disabled():
    r = stt.transcribe(
        wav_bytes(), stt.BACKEND_ASSEMBLYAI, UsageTracker(), allow_fallback=False,
        env={stt.ENV_ASSEMBLYAI_KEY: FAKE_KEY}, cloud_fn=lambda k, a: "",
        local_fn=no_cloud, cloud_confirmed=True)
    assert not r.ok and "empty" in r.error


# ---------- 6. missing credentials ----------

def test_missing_api_key_never_calls_cloud_and_falls_back():
    usage = UsageTracker()
    r = stt.transcribe(
        wav_bytes(), stt.BACKEND_ASSEMBLYAI, usage, env={},
        cloud_fn=no_cloud, local_fn=lambda a: "analyze", cloud_confirmed=True)
    assert r.ok and r.fallback_used and usage.assemblyai_calls == 0


def test_missing_api_key_and_no_local_gives_clear_error():
    def missing(_a):
        raise stt.BackendUnavailable("no whisper")

    r = stt.transcribe(wav_bytes(), stt.BACKEND_ASSEMBLYAI, env={}, cloud_fn=no_cloud, local_fn=missing, cloud_confirmed=True)
    assert not r.ok and stt.ENV_ASSEMBLYAI_KEY in r.error


def test_no_audio_is_an_error_without_any_call():
    r = stt.transcribe(b"", stt.BACKEND_ASSEMBLYAI, env={stt.ENV_ASSEMBLYAI_KEY: "k"},
                       cloud_fn=no_cloud, local_fn=no_cloud, cloud_confirmed=True)
    assert not r.ok and "No audio" in r.error


# ---------- 4. TTS backend selection ----------

def test_browser_tts_is_default_and_free():
    usage = UsageTracker()
    out = tts.speak("hello", usage=usage, polly_fn=no_cloud)
    assert out.backend == tts.BACKEND_BROWSER and not out.is_cloud and out.audio_mp3 is None
    assert usage.polly_requests == 0 and usage.browser_speech_requests == 1


def test_polly_requires_explicit_opt_in():
    out = tts.speak("hello", tts.BACKEND_POLLY, polly_enabled=False, polly_fn=no_cloud)
    assert out.backend == tts.BACKEND_BROWSER and out.fallback_used and "opt-in" in out.error


def test_polly_when_enabled_mocked():
    seen = {}

    def fake(text, voice, engine, region):
        seen.update(text=text, voice=voice, engine=engine, region=region)
        return b"mp3"

    usage = UsageTracker()
    out = tts.speak(
        "hello", tts.BACKEND_POLLY, usage, polly_enabled=True, polly_fn=fake,
        env={tts.ENV_POLLY_VOICE: "Matthew", tts.ENV_POLLY_ENGINE: "standard", "AWS_REGION": "us-east-1"},
    )
    assert out.backend == tts.BACKEND_POLLY and out.is_cloud and out.audio_mp3 == b"mp3"
    assert seen == {"text": "hello", "voice": "Matthew", "engine": "standard", "region": "us-east-1"}
    assert usage.polly_requests == 1 and usage.polly_characters == 5


def test_unknown_tts_backend_falls_back():
    out = tts.speak("hi", "nonsense", polly_fn=no_cloud)
    assert out.backend == tts.BACKEND_BROWSER and out.error


def test_browser_html_escapes_script_injection():
    html_doc = tts.browser_speech_html('</script><script>alert(1)</script>')
    assert "</script><script>alert" not in html_doc


# ---------- 5. cloud failures (Polly) ----------

def test_polly_failure_falls_back_to_browser():
    def fail(*_a):
        raise RuntimeError("AccessDenied")

    out = tts.speak("hello", tts.BACKEND_POLLY, polly_enabled=True, polly_fn=fail)
    assert out.backend == tts.BACKEND_BROWSER and out.fallback_used and "Polly failed" in out.error


# ---------- 10. no unnecessary paid calls / limits ----------

def test_router_makes_no_cloud_calls(monkeypatch):
    monkeypatch.setattr(stt, "assemblyai_transcribe", no_cloud)
    monkeypatch.setattr(tts, "polly_synthesize", no_cloud)
    s = VoiceSession()
    for text in ["Analyze this MRI", "Explain", "What changed?", "Evidence report",
                 "Request human review", "approval status", "approve it", "gibberish"]:
        route_command(text, "case-002-change", s, "Dr. A")


def test_assemblyai_call_limit_blocks_cloud():
    usage = UsageTracker(limits=UsageLimits(max_assemblyai_calls=1))
    env = {stt.ENV_ASSEMBLYAI_KEY: FAKE_KEY}
    kw = dict(env=env, cloud_fn=lambda k, a: "analyze", local_fn=lambda a: "local")
    assert stt.transcribe(wav_bytes(), stt.BACKEND_ASSEMBLYAI, usage, **kw, cloud_confirmed=True).is_cloud
    r = stt.transcribe(wav_bytes(), stt.BACKEND_ASSEMBLYAI, usage, **kw, cloud_confirmed=True)
    assert r.fallback_used and r.text == "local" and usage.assemblyai_calls == 1


def test_assemblyai_seconds_limit_blocks_cloud():
    usage = UsageTracker(limits=UsageLimits(max_assemblyai_audio_seconds=1.0))
    r = stt.transcribe(wav_bytes(5), stt.BACKEND_ASSEMBLYAI, usage,
                       env={stt.ENV_ASSEMBLYAI_KEY: FAKE_KEY}, cloud_fn=no_cloud,
                       local_fn=lambda a: "local", cloud_confirmed=True)
    assert r.fallback_used and usage.assemblyai_calls == 0


def test_failed_cloud_attempt_still_counts():
    usage = UsageTracker()

    def fail(*_a):
        raise TimeoutError

    stt.transcribe(wav_bytes(), stt.BACKEND_ASSEMBLYAI, usage,
                   env={stt.ENV_ASSEMBLYAI_KEY: FAKE_KEY}, cloud_fn=fail, local_fn=lambda a: "x", cloud_confirmed=True)
    assert usage.assemblyai_calls == 1


def test_polly_character_limit_blocks_cloud():
    usage = UsageTracker(limits=UsageLimits(max_polly_characters=3))
    out = tts.speak("hello", tts.BACKEND_POLLY, usage, polly_enabled=True, polly_fn=no_cloud)
    assert out.backend == tts.BACKEND_BROWSER and "limit" in out.error and usage.polly_requests == 0


def test_limits_from_env_and_bad_values():
    lim = UsageLimits.from_env({"ALEXA_MAX_ASSEMBLYAI_CALLS": "3", "ALEXA_MAX_POLLY_CHARS": "abc"})
    assert lim.max_assemblyai_calls == 3 and lim.max_polly_characters == 3000
    assert UsageLimits.from_env({"ALEXA_MAX_POLLY_REQUESTS": "0"}).max_polly_requests == 0


def test_local_is_default_backend():
    assert stt.DEFAULT_STT_BACKEND == stt.BACKEND_LOCAL
    assert tts.DEFAULT_TTS_BACKEND == tts.BACKEND_BROWSER


# ---------- local Whisper / PyAV compatibility regression ----------

def test_faster_whisper_audio_decoding_works_with_installed_pyav():
    """Regression: PyAV 19 rejects av.open(metadata_errors=...) used by faster-whisper."""
    pytest.importorskip("faster_whisper")
    from faster_whisper.audio import decode_audio

    audio = decode_audio(io.BytesIO(wav_bytes(1.0)), sampling_rate=16000)
    assert len(audio) == pytest.approx(16000, abs=160)


def test_pyav_metadata_errors_typeerror_gives_actionable_message(monkeypatch):
    import sys
    import types

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(stt, "_WHISPER_MODELS", {})
    r = stt.transcribe(wav_bytes(), stt.BACKEND_LOCAL)
    assert not r.ok and 'pip install "av<19"' in r.error
