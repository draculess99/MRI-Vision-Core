"""Streamlit Voice Assistant section for the simulated Alexa+ MRI copilot.

Push-to-talk only: audio is recorded with st.audio_input and transcribed only
when the user presses "Transcribe". Local Whisper is the default backend; the
cloud backends are opt-in. This is an Alexa-style simulation, not an Alexa Skill.
"""

from __future__ import annotations

import hashlib
from typing import Callable, MutableMapping, Optional

import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.uploaded_file_manager import DeletedFile

from hackathon.amazon_alexa import stt, tts
from hackathon.amazon_alexa.tools import NON_DIAGNOSTIC_NOTICE
from hackathon.amazon_alexa.usage import BILLING_CONTROLS_NOTICE, UsageTracker
from hackathon.amazon_alexa.voice_router import VoiceSession, route_command

STT_LABELS = {
    stt.BACKEND_LOCAL: "Local Whisper (offline, free) — default",
    stt.BACKEND_ASSEMBLYAI: "AssemblyAI (cloud, uses credits)",
}
TTS_LABELS = {
    tts.BACKEND_BROWSER: "Browser speech (local, free) — default",
    tts.BACKEND_POLLY: "Amazon Polly (cloud, AWS charges)",
}


def _state() -> tuple[UsageTracker, VoiceSession]:
    ss = st.session_state
    ss.setdefault("voice_usage", UsageTracker())
    ss.setdefault("voice_session", VoiceSession())
    return ss["voice_usage"], ss["voice_session"]


def _badge(label: str, cloud: bool) -> str:
    return f"{'☁️ CLOUD' if cloud else '💻 LOCAL'} · {label}"


def recording_bytes(recording: object) -> Optional[bytes]:
    """Bytes of a valid recording, else None.

    st.audio_input returns an UploadedFile, None, or (in raw session_state) a
    DeletedFile placeholder once the upload manager has dropped the file. Only
    objects that can actually provide bytes are accepted; nothing here raises.
    """
    if recording is None or isinstance(recording, DeletedFile):
        return None
    getvalue = getattr(recording, "getvalue", None)
    if not callable(getvalue):
        return None
    try:
        data = getvalue()
    except Exception:  # noqa: BLE001 - expired/closed file objects
        return None
    return data if isinstance(data, (bytes, bytearray)) and len(data) > 0 else None


def run_transcription(
    ss: MutableMapping, transcribe_fn: Optional[Callable] = None, *, local_only: bool = False
) -> None:
    """Transcribe the cached recording once per (recording, backend).

    Pure session-state logic so it can be tested without a browser. Executes
    only from the Transcribe button callback, never on plain reruns.
    """
    data = ss.get("voice_audio_bytes")
    if not data:
        ss["voice_stt_error"] = (
            "No recording available. Record a command, then press Transcribe recording."
        )
        return
    backend = ss.get("voice_stt_backend", stt.DEFAULT_STT_BACKEND)
    if local_only or ss.get("voice_local_only"):
        backend = stt.BACKEND_LOCAL
        ss["voice_stt_backend"] = backend
    digest = hashlib.sha256(data).hexdigest()
    if ss.get("voice_last_transcribed") == (digest, backend):
        ss["voice_stt_error"] = (
            "This recording was already transcribed with this backend. "
            "Record again to transcribe a new one."
        )
        return
    if backend == stt.BACKEND_ASSEMBLYAI and not ss.get("voice_cloud_stt_confirmed"):
        ss["voice_stt_error"] = (
            "Tick the AssemblyAI confirmation box first; nothing was sent."
        )
        return
    result = (transcribe_fn or stt.transcribe)(
        data, backend, ss["voice_usage"],
        cloud_confirmed=bool(ss.get("voice_cloud_stt_confirmed")),
    )
    if backend == stt.BACKEND_ASSEMBLYAI:
        ss["voice_cloud_stt_confirmed"] = False  # confirmation is single-use
    ss["voice_last_transcribed"] = (digest, backend)
    ss["voice_stt_result"] = result
    ss["voice_stt_error"] = result.error
    if result.text:
        ss["voice_command_text"] = result.text


def _on_transcribe() -> None:
    """Button callback: the only place a speech-to-text backend is invoked."""
    run_transcription(st.session_state)


def render_voice_assistant(
    case_id: str, reviewer_name: str, comment: str, *, local_evidence=None
) -> None:
    usage, session = _state()
    ss = st.session_state
    ss.setdefault("voice_command_text", "")
    local_image_mode = local_evidence is not None
    ss["voice_local_only"] = local_image_mode

    st.markdown("---")
    st.markdown("## 🎙️ Voice Assistant (Alexa-style simulation)")
    st.caption(
        "Push-to-talk. Not an Amazon Alexa Skill or device integration. "
        "Responses come only from the demonstration case's tool outputs."
    )

    last = ss.get("last_voice_result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current case", case_id)
    c2.metric("Review status", last_status(case_id, local_evidence))
    c3.metric("Approval", session.requests[case_id].approval_state.value
              if case_id in session.requests else "PENDING_HUMAN_APPROVAL")

    key_set = stt.get_assemblyai_key() is not None
    whisper_ok = stt.local_whisper_available()
    b1, b2 = st.columns(2)
    with b1:
        stt_backend = st.selectbox(
            "Speech-to-text backend", [stt.BACKEND_LOCAL] if local_image_mode else stt.STT_BACKENDS,
            format_func=STT_LABELS.get, key="voice_stt_backend",
        )
        st.caption(
            f"Local Whisper installed: {'yes' if whisper_ok else 'no'} · "
            f"{stt.ENV_ASSEMBLYAI_KEY}: {'configured' if key_set else 'not set'}"
        )
        if local_image_mode:
            st.caption("Local-image mode blocks AssemblyAI; audio stays local.")
        elif stt_backend == stt.BACKEND_ASSEMBLYAI:
            st.warning("☁️ Cloud mode: audio is sent to AssemblyAI only when you press Transcribe.")
            st.checkbox(
                "I confirm sending this recording to AssemblyAI (uses paid credits)",
                key="voice_cloud_stt_confirmed",
            )
    with b2:
        tts_backend = st.selectbox(
            "Text-to-speech backend", [tts.BACKEND_BROWSER] if local_image_mode else tts.TTS_BACKENDS,
            format_func=TTS_LABELS.get, key="voice_tts_backend",
        )
        polly_on = False
        if tts_backend == tts.BACKEND_POLLY:
            polly_on = st.checkbox(
                "I understand Amazon Polly is billed by AWS and want to enable it",
                key="voice_polly_enabled",
            )
        st.caption(
            "Browser speech only in local-image mode."
            if local_image_mode else "Using browser speech unless Polly is selected and enabled."
        )

    # Use the widget's return value (never raw session_state, which can hold a
    # DeletedFile placeholder) and cache the bytes for the Transcribe callback.
    ss["voice_audio_bytes"] = recording_bytes(
        st.audio_input("Record a command (push-to-talk)", key="voice_audio")
    )
    st.button("Transcribe recording", on_click=_on_transcribe, key="voice_transcribe_btn")

    res = ss.get("voice_stt_result")
    if res is not None and res.text:
        origin = _badge(stt_label(res.backend), res.is_cloud)
        note = " (fell back from cloud)" if res.fallback_used else ""
        st.success(f"Heard: “{res.text}” — {origin}{note}")
    if ss.get("voice_stt_error"):
        st.error(ss["voice_stt_error"])

    st.text_input(
        "Command (editable — type here if you prefer, or correct the transcript)",
        key="voice_command_text",
        placeholder='e.g. "Analyze this MRI"',
    )
    st.caption("Try: " + " · ".join(
        ["Analyze this MRI", "Explain the review status", "What changed between the images?",
         "Generate an evidence report", "Request human review", "What is the approval status?"]
    ))

    if st.button("Execute command", type="primary", key="voice_execute_btn"):
        source = "voice" if res is not None and res.text == ss["voice_command_text"] else "typed"
        ss["last_voice_result"] = route_command(
            ss["voice_command_text"], case_id, session, reviewer_name, comment,
            source=source, evidence=local_evidence,
        )
        last = ss["last_voice_result"]
        ss.pop("voice_speech", None)

    if last is not None:
        (st.success if last.ok else st.warning)(f"Intent: `{last.intent.value}`")
        st.markdown(last.display)

        if st.button("🔊 Speak response", key="voice_speak_btn"):
            try:
                ss["voice_speech"] = tts.speak(
                    last.spoken, tts.BACKEND_BROWSER if local_image_mode else tts_backend,
                    usage, polly_enabled=False if local_image_mode else polly_on
                )
            except Exception as exc:  # noqa: BLE001 - never crash the page
                ss["voice_speech"] = None
                st.error(f"Speech failed: {type(exc).__name__}")
        speech = ss.get("voice_speech")
        if speech is not None:
            st.caption(_badge(tts_label(speech.backend), speech.is_cloud))
            if speech.error:
                st.warning(speech.error)
            if speech.audio_mp3:
                st.audio(speech.audio_mp3, format="audio/mp3", autoplay=True)
            else:
                components.html(tts.browser_speech_html(speech.text), height=60)

    st.info(f"{NON_DIAGNOSTIC_NOTICE} Voice commands cannot approve a case.")

    with st.expander("Usage counters & cost safeguards"):
        for name, value in usage.snapshot().items():
            st.markdown(f"- **{name}:** {value}")
        st.caption(BILLING_CONTROLS_NOTICE)
    with st.expander("Audit trail (this session)"):
        if session.audit:
            st.dataframe(session.audit, use_container_width=True)
        else:
            st.caption("No voice activity yet.")


def last_status(case_id: str, local_evidence=None) -> str:
    from hackathon.amazon_alexa.tools import analyze_mri_evidence

    return (local_evidence or analyze_mri_evidence(case_id)).status.value


def stt_label(backend: str) -> str:
    return "AssemblyAI" if backend == stt.BACKEND_ASSEMBLYAI else "Local Whisper"


def tts_label(backend: str) -> str:
    return "Amazon Polly" if backend == tts.BACKEND_POLLY else "Browser speech"
