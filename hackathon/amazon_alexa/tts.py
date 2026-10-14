"""Text-to-speech backends: browser speech synthesis (default) and Amazon Polly.

Browser synthesis runs entirely client-side and costs nothing. Amazon Polly is
opt-in: it is used only when the caller selects it AND passes polly_enabled=True
AND the usage limits allow it. Any Polly failure falls back to browser speech.
AWS credentials come from the standard AWS environment/credential chain used by
boto3; nothing is read from or written to the UI.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from hackathon.amazon_alexa.usage import UsageTracker

BACKEND_BROWSER = "browser"
BACKEND_POLLY = "amazon_polly"
TTS_BACKENDS = (BACKEND_BROWSER, BACKEND_POLLY)
DEFAULT_TTS_BACKEND = BACKEND_BROWSER

ENV_POLLY_VOICE = "ALEXA_POLLY_VOICE_ID"
ENV_POLLY_ENGINE = "ALEXA_POLLY_ENGINE"  # "neural" (default) or "standard"
ENV_POLLY_REGION = "AWS_REGION"
DEFAULT_POLLY_VOICE = "Joanna"

# (text, voice_id, engine, region) -> mp3 bytes; raises on failure.
PollyFn = Callable[[str, str, str, Optional[str]], bytes]


@dataclass(frozen=True)
class SpeechOutput:
    text: str
    backend: str  # backend that actually produced the output
    requested_backend: str
    is_cloud: bool
    audio_mp3: Optional[bytes] = None  # set for Polly; None for browser speech
    fallback_used: bool = False
    error: Optional[str] = None


def polly_synthesize(text: str, voice_id: str, engine: str, region: Optional[str]) -> bytes:
    """Real Amazon Polly call. Billed by AWS per character."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is not installed.") from exc
    client = boto3.client("polly", region_name=region) if region else boto3.client("polly")
    response = client.synthesize_speech(
        Text=text, OutputFormat="mp3", VoiceId=voice_id, Engine=engine
    )
    return response["AudioStream"].read()


def speak(
    text: str,
    backend: str = DEFAULT_TTS_BACKEND,
    usage: Optional[UsageTracker] = None,
    *,
    polly_enabled: bool = False,
    env: Optional[Mapping[str, str]] = None,
    polly_fn: Optional[PollyFn] = None,
) -> SpeechOutput:
    """Prepare speech for `text`. Never raises; failures fall back to browser."""
    usage = usage if usage is not None else UsageTracker()
    env = os.environ if env is None else env
    polly_fn = polly_fn or polly_synthesize  # resolved at call time

    def browser(error: Optional[str] = None, fallback: bool = False) -> SpeechOutput:
        usage.browser_speech_requests += 1
        return SpeechOutput(
            text, BACKEND_BROWSER, backend, False, fallback_used=fallback, error=error
        )

    if backend not in TTS_BACKENDS:
        return browser(error=f"Unknown TTS backend: {backend}; using browser speech.", fallback=True)
    if backend == BACKEND_BROWSER:
        return browser()

    # Amazon Polly: strictly opt-in.
    if not polly_enabled:
        return browser(
            error="Amazon Polly is not enabled (opt-in required); using browser speech.",
            fallback=True,
        )
    cache_key = hashlib.sha256(f"{text}".encode("utf-8")).hexdigest()
    cached = usage.polly_cache.get(cache_key)
    if cached:  # identical text already synthesised this session: no second paid request
        return SpeechOutput(text, BACKEND_POLLY, backend, True, audio_mp3=cached)
    allowed, reason = usage.check_polly(len(text))
    if not allowed:
        return browser(error=f"{reason} Using browser speech.", fallback=True)

    voice = env.get(ENV_POLLY_VOICE, DEFAULT_POLLY_VOICE)
    engine = env.get(ENV_POLLY_ENGINE, "neural")
    if engine not in ("neural", "standard"):
        engine = "neural"
    usage.record_polly(len(text))  # count the attempt, even if it fails
    try:
        audio = polly_fn(text, voice, engine, env.get(ENV_POLLY_REGION))
    except Exception as exc:  # noqa: BLE001 - botocore errors vary widely
        return browser(
            error=f"Amazon Polly failed: {type(exc).__name__}. Using browser speech.",
            fallback=True,
        )
    if not audio:
        return browser(error="Amazon Polly returned no audio. Using browser speech.", fallback=True)
    usage.polly_cache[cache_key] = audio
    return SpeechOutput(text, BACKEND_POLLY, backend, True, audio_mp3=audio)


def browser_speech_html(text: str) -> str:
    """Self-contained HTML/JS that speaks `text` via window.speechSynthesis.

    Includes a Speak button because browsers may block speech that is not tied
    to a user gesture inside the iframe; it also attempts to speak on load.
    """
    payload = json.dumps(text).replace("<", "\\u003c").replace(">", "\\u003e")
    label = html.escape("Speak response")
    return f"""
<div style="font-family:sans-serif">
  <button id="s" style="padding:6px 14px;font-size:14px;cursor:pointer">&#128266; {label}</button>
  <button id="x" style="padding:6px 14px;font-size:14px;cursor:pointer">Stop</button>
  <span id="m" style="margin-left:8px;font-size:12px;color:#888"></span>
</div>
<script>
const text = {payload};
const m = document.getElementById('m');
function say() {{
  if (!('speechSynthesis' in window)) {{ m.textContent = 'Speech synthesis not supported in this browser.'; return; }}
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.onerror = (e) => {{ m.textContent = 'Speech blocked or failed: ' + e.error; }};
  window.speechSynthesis.speak(u);
}}
document.getElementById('s').onclick = say;
document.getElementById('x').onclick = () => window.speechSynthesis.cancel();
setTimeout(say, 150);
</script>
"""
