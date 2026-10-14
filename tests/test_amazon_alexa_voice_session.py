"""Session-state / rerun regression tests for the Voice Assistant (no network, no credentials)."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.runtime.uploaded_file_manager import DeletedFile
from streamlit.testing.v1 import AppTest

from hackathon.amazon_alexa import stt, tts, voice_ui
from hackathon.amazon_alexa.stt import TranscriptionResult
from hackathon.amazon_alexa.usage import UsageTracker

APP = str(Path(__file__).parent.parent / "hackathon" / "amazon_alexa" / "streamlit_app.py")
AUDIO = b"RIFF-fake-audio-bytes"


class FakeRecording:
    def __init__(self, data=AUDIO):
        self._data = data

    def getvalue(self):
        return self._data


class ClosedRecording:
    def getvalue(self):
        raise ValueError("I/O operation on closed file")


def fake_result(text="Analyze this MRI", cloud=False):
    return TranscriptionResult(text, stt.BACKEND_ASSEMBLYAI if cloud else stt.BACKEND_LOCAL,
                               cloud, stt.BACKEND_LOCAL)


# ---------- recording_bytes ----------

def test_valid_recording():
    assert voice_ui.recording_bytes(FakeRecording()) == AUDIO


@pytest.mark.parametrize(
    "value",
    [None, DeletedFile("file-id"), ClosedRecording(), FakeRecording(b""), object(), "text"],
)
def test_invalid_recordings_return_none_without_raising(value):
    assert voice_ui.recording_bytes(value) is None


# ---------- run_transcription ----------

def base_state(**kw):
    ss = {"voice_usage": UsageTracker(), "voice_stt_backend": stt.BACKEND_LOCAL}
    ss.update(kw)
    return ss


def test_missing_recording_is_a_friendly_message_not_a_crash():
    calls = []
    ss = base_state(voice_audio_bytes=None)
    voice_ui.run_transcription(ss, lambda *a, **k: calls.append(1))
    assert "No recording available" in ss["voice_stt_error"] and not calls


def test_deleted_file_in_raw_widget_state_is_never_touched():
    calls = []
    ss = base_state(voice_audio=DeletedFile("gone"), voice_audio_bytes=None)
    voice_ui.run_transcription(ss, lambda *a, **k: calls.append(1))
    assert "No recording available" in ss["voice_stt_error"] and not calls


def test_valid_recording_transcribes_once_and_sets_command_text():
    calls = []

    def fake(data, backend, usage, **kw):
        calls.append((data, backend))
        return fake_result()

    ss = base_state(voice_audio_bytes=AUDIO)
    voice_ui.run_transcription(ss, fake)
    assert ss["voice_command_text"] == "Analyze this MRI" and ss["voice_stt_error"] is None
    voice_ui.run_transcription(ss, fake)  # second click on the same recording
    assert len(calls) == 1 and "already transcribed" in ss["voice_stt_error"]
    assert ss["voice_command_text"] == "Analyze this MRI"  # preserved


def test_new_recording_is_transcribed_again():
    calls = []
    ss = base_state(voice_audio_bytes=AUDIO)
    fake = lambda *a, **k: calls.append(1) or fake_result()  # noqa: E731
    voice_ui.run_transcription(ss, fake)
    ss["voice_audio_bytes"] = AUDIO + b"2"
    voice_ui.run_transcription(ss, fake)
    assert len(calls) == 2


def test_cloud_backend_needs_confirmation_and_confirmation_is_single_use():
    calls = []
    fake = lambda *a, **k: calls.append(k) or fake_result(cloud=True)  # noqa: E731
    ss = base_state(voice_audio_bytes=AUDIO, voice_stt_backend=stt.BACKEND_ASSEMBLYAI)
    voice_ui.run_transcription(ss, fake)
    assert not calls and "confirmation" in ss["voice_stt_error"]
    ss["voice_cloud_stt_confirmed"] = True
    voice_ui.run_transcription(ss, fake)
    assert len(calls) == 1 and calls[0]["cloud_confirmed"] is True
    assert ss["voice_cloud_stt_confirmed"] is False


# ---------- Streamlit reruns ----------

@pytest.fixture
def counted(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    calls = []
    monkeypatch.setattr(stt, "transcribe", lambda *a, **k: calls.append(a[1]) or fake_result())
    monkeypatch.setattr(st, "audio_input", lambda *a, **k: FakeRecording())
    return calls


def click(at, key):
    next(b for b in at.button if b.key == key).click()
    at.run()
    assert not at.exception


def test_execute_and_speak_do_not_retranscribe_and_text_survives_reruns(counted):
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    click(at, "voice_transcribe_btn")
    assert counted == [stt.BACKEND_LOCAL]
    assert at.text_input(key="voice_command_text").value == "Analyze this MRI"
    click(at, "voice_execute_btn")
    click(at, "voice_speak_btn")
    at.run()  # plain rerun
    assert counted == [stt.BACKEND_LOCAL]
    assert at.text_input(key="voice_command_text").value == "Analyze this MRI"


def test_transcribe_without_recording_shows_message(monkeypatch):
    monkeypatch.setattr(st, "audio_input", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    click(at, "voice_transcribe_btn")
    assert any("No recording available" in e.value for e in at.error)


def test_deleted_file_returned_by_widget_does_not_crash(monkeypatch):
    monkeypatch.setattr(st, "audio_input", lambda *a, **k: DeletedFile("x"))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    click(at, "voice_transcribe_btn")
    assert any("No recording available" in e.value for e in at.error)


# ---------- cloud guards (stt / tts level) ----------

def test_cloud_stt_requires_confirmation_and_never_sends_same_audio_twice():
    calls = []
    kw = dict(env={stt.ENV_ASSEMBLYAI_KEY: "k"}, cloud_fn=lambda k, a: calls.append(1) or "ok",
              local_fn=lambda a: "local")
    usage = UsageTracker()
    r = stt.transcribe(AUDIO, stt.BACKEND_ASSEMBLYAI, usage, **kw)  # not confirmed
    assert not calls and not r.ok and "confirmation" in r.error and usage.assemblyai_calls == 0
    assert stt.transcribe(AUDIO, stt.BACKEND_ASSEMBLYAI, usage, cloud_confirmed=True, **kw).is_cloud
    r = stt.transcribe(AUDIO, stt.BACKEND_ASSEMBLYAI, usage, cloud_confirmed=True, **kw)
    assert len(calls) == 1 and r.fallback_used and r.text == "local"


def test_polly_repeat_of_same_text_uses_cache_not_a_second_request():
    calls = []
    usage = UsageTracker()
    fn = lambda *a: calls.append(1) or b"mp3"  # noqa: E731
    a = tts.speak("hello", tts.BACKEND_POLLY, usage, polly_enabled=True, polly_fn=fn)
    b = tts.speak("hello", tts.BACKEND_POLLY, usage, polly_enabled=True, polly_fn=fn)
    assert len(calls) == 1 and a.audio_mp3 == b.audio_mp3 and usage.polly_requests == 1


# ---------- live-validation script is inert without --confirm-paid ----------

def test_live_validation_dry_run_makes_no_cloud_calls(monkeypatch, tmp_path, capsys):
    from hackathon.amazon_alexa import live_validation as lv

    def boom(*a, **k):
        raise AssertionError("paid call attempted")

    monkeypatch.setattr(stt, "assemblyai_transcribe", boom)
    monkeypatch.setattr(tts, "polly_synthesize", boom)
    monkeypatch.setattr(stt, "transcribe", boom)
    monkeypatch.setattr(tts, "speak", boom)
    wav = tmp_path / "a.wav"
    wav.write_bytes(AUDIO)
    assert lv.main(["stt", "--wav", str(wav)]) == 0
    assert lv.main(["tts"]) == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_live_validation_confirmed_runs_use_mocks_and_report_counters(monkeypatch, tmp_path, capsys):
    from hackathon.amazon_alexa import live_validation as lv

    monkeypatch.setenv(stt.ENV_ASSEMBLYAI_KEY, "k")
    monkeypatch.setattr(stt, "assemblyai_transcribe", lambda k, a: "hello")
    monkeypatch.setattr(tts, "polly_synthesize", lambda *a: b"mp3")
    wav = tmp_path / "a.wav"
    wav.write_bytes(AUDIO)
    assert lv.main(["stt", "--wav", str(wav), "--confirm-paid"]) == 0
    assert lv.main(["tts", "--out", str(tmp_path / "o.mp3"), "--confirm-paid"]) == 0
    out = capsys.readouterr().out
    assert "AssemblyAI calls: 0 / 1 -> 1 / 1" in out and "Polly requests: 0 / 1 -> 1 / 1" in out
    assert (tmp_path / "o.mp3").read_bytes() == b"mp3"


# ---------- AssemblyAI authentication path (no network) ----------

SECRET = "0123456789abcdef0123456789abcdef"


def test_diagnostics_report_safe_facts_and_never_the_key():
    info = stt.assemblyai_auth_diagnostics({stt.ENV_ASSEMBLYAI_KEY: f"  {SECRET}\n"})
    assert info["env_var_read"] == "ASSEMBLYAI_API_KEY"
    assert info["key_present"] and info["key_length"] == 32
    assert info["had_surrounding_whitespace"] is True
    assert info["has_quote_chars"] is False
    assert info["endpoint_host"] == "api.assemblyai.com"
    assert info["authorization_header_matches_key"] is True
    assert info["key_fingerprint_sha256_8"] == stt.key_fingerprint(SECRET)
    assert SECRET not in repr(info)


def test_diagnostics_flag_quotes_and_alternate_env_vars():
    info = stt.assemblyai_auth_diagnostics(
        {stt.ENV_ASSEMBLYAI_KEY: f'"{SECRET}"', "ASSEMBLYAI_BASE_URL": "https://eu.example.test"}
    )
    assert info["has_quote_chars"] is True and info["key_length"] == 34
    assert info["other_assemblyai_env_vars"] == ["ASSEMBLYAI_BASE_URL"]


def test_diagnostics_without_key():
    info = stt.assemblyai_auth_diagnostics({})
    assert info["key_present"] is False and info["key_fingerprint_sha256_8"] is None


def test_sdk_receives_exact_stripped_key_via_explicit_client_not_globals(monkeypatch):
    """The real assemblyai_transcribe must hand the SDK the exact key per call."""
    import assemblyai as aai

    seen = {}

    class FakeTranscript:
        status = "completed"
        error = None
        text = "hello"

    class FakeTranscriber:
        def __init__(self, **kw):
            seen["kwargs"] = kw

        def transcribe(self, audio):
            return FakeTranscript()

    monkeypatch.setattr(aai, "Transcriber", FakeTranscriber)
    aai.settings.api_key = "STALE-GLOBAL-KEY"
    try:
        from tests.conftest import ORIGINALS

        assert ORIGINALS["assemblyai_transcribe"](SECRET, AUDIO) == "hello"
    finally:
        aai.settings.api_key = None
    assert seen["kwargs"] == {"api_key": SECRET}


def test_upload_request_carries_authorization_header_equal_to_key():
    """Build the real SDK client and capture the outgoing request via a mock transport."""
    import assemblyai as aai
    import httpx

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["host"] = request.url.host
        captured["auth_ok"] = request.headers.get("authorization") == SECRET
        return httpx.Response(200, json={"upload_url": "https://cdn.example.test/x"})

    client = aai.Client(api_key=SECRET)
    client.http_client._transport = httpx.MockTransport(handler)
    url = aai.api.upload_file(client.http_client, AUDIO)
    assert url == "https://cdn.example.test/x"
    assert captured == {"host": "api.assemblyai.com", "auth_ok": True}


def test_invalid_key_error_is_redacted_and_falls_back(monkeypatch):
    def reject(key, _audio):
        raise RuntimeError(f"Failed to upload audio file: Invalid API key ({key})")

    usage = UsageTracker()
    r = stt.transcribe(AUDIO, stt.BACKEND_ASSEMBLYAI, usage, cloud_confirmed=True,
                       env={stt.ENV_ASSEMBLYAI_KEY: SECRET}, cloud_fn=reject,
                       local_fn=lambda a: "local text")
    assert r.fallback_used and r.text == "local text" and usage.assemblyai_calls == 1


def test_diagnose_cli_makes_no_network_calls_and_hides_key(monkeypatch, capsys):
    from hackathon.amazon_alexa import live_validation as lv

    monkeypatch.setenv(stt.ENV_ASSEMBLYAI_KEY, SECRET)
    assert lv.main(["diagnose"]) == 0
    out = capsys.readouterr().out
    assert "key_length: 32" in out and SECRET not in out


# ---------- failure classification ----------

@pytest.mark.parametrize(
    "exc,label",
    [
        (__import__("assemblyai").TranscriptError("Failed to upload audio file: Invalid API key", 401), "AUTHENTICATION"),
        (__import__("assemblyai").TranscriptError("forbidden", 403), "PERMISSION"),
        (__import__("assemblyai").TranscriptError("no credits", 402), "BILLING"),
        (__import__("assemblyai").TranscriptError("slow down", 429), "RATE LIMIT"),
        (__import__("assemblyai").TranscriptError("oops", 503), "SERVICE"),
        (TypeError("unexpected keyword argument"), "SDK INCOMPATIBILITY"),
        (ConnectionError("down"), "NETWORK"),
        (ValueError("???"), "UNCLASSIFIED"),
    ],
)
def test_assemblyai_errors_are_classified(exc, label):
    assert stt.classify_assemblyai_error(exc).startswith(label)


def test_classified_auth_failure_is_surfaced_without_key():
    import assemblyai as aai

    def reject(key, _a):
        raise aai.TranscriptError(f"Failed to upload audio file: Invalid API key {key}", 401)

    def no_local(_a):
        raise stt.BackendUnavailable("no whisper")

    r = stt.transcribe(AUDIO, stt.BACKEND_ASSEMBLYAI, UsageTracker(), cloud_confirmed=True,
                       env={stt.ENV_ASSEMBLYAI_KEY: SECRET}, cloud_fn=reject, local_fn=no_local)
    assert "AUTHENTICATION" in r.error and SECRET not in r.error
