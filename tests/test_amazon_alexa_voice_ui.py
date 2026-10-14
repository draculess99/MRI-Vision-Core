"""Smoke tests for the Voice Assistant section (Streamlit AppTest, no credentials)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "hackathon" / "amazon_alexa" / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def _button(at, key):
    return next(b for b in at.button if b.key == key)


def test_voice_section_renders_with_local_defaults(app):
    assert any("Voice Assistant" in m.value for m in app.markdown)
    stt_box = next(s for s in app.selectbox if s.key == "voice_stt_backend")
    tts_box = next(s for s in app.selectbox if s.key == "voice_tts_backend")
    assert stt_box.value == "local_whisper" and tts_box.value == "browser"


def test_typed_command_executes_and_approval_is_refused(app):
    app.text_input(key="voice_command_text").set_value("Analyze this MRI")
    _button(app, "voice_execute_btn").click()
    app.run()
    assert not app.exception
    assert any("analyze" in m.value for m in app.markdown)

    app.text_input(key="voice_command_text").set_value("Approve this MRI")
    _button(app, "voice_execute_btn").click()
    app.run()
    assert any("can't approve" in m.value for m in app.markdown)
    assert not any("APPROVED_BY_HUMAN`" in m.value and "Approval State" in m.value for m in app.markdown)
