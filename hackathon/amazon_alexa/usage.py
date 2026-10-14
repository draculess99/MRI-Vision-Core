"""Application-level usage counters and spend safeguards for voice features.

These counters live in process memory (one tracker per Streamlit session) and
only limit what *this app* will attempt. They do NOT replace provider-side
billing controls (AssemblyAI usage limits, AWS Budgets / billing alarms).
"""

from __future__ import annotations

import io
import os
import wave
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Set, Tuple

ENV_MAX_ASSEMBLYAI_CALLS = "ALEXA_MAX_ASSEMBLYAI_CALLS"
ENV_MAX_ASSEMBLYAI_SECONDS = "ALEXA_MAX_ASSEMBLYAI_SECONDS"
ENV_MAX_POLLY_REQUESTS = "ALEXA_MAX_POLLY_REQUESTS"
ENV_MAX_POLLY_CHARS = "ALEXA_MAX_POLLY_CHARS"

BILLING_CONTROLS_NOTICE = (
    "These counters are application-level safeguards only. They reset when the "
    "session restarts and do not replace provider-side billing controls "
    "(AssemblyAI usage limits, AWS Budgets and billing alarms)."
)


def _env_number(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


@dataclass(frozen=True)
class UsageLimits:
    """Per-session ceilings for paid/cloud calls. Zero disables the backend."""

    max_assemblyai_calls: int = 20
    max_assemblyai_audio_seconds: float = 300.0
    max_polly_requests: int = 20
    max_polly_characters: int = 3000

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "UsageLimits":
        env = os.environ if env is None else env
        d = cls()
        return cls(
            max_assemblyai_calls=int(
                _env_number(env, ENV_MAX_ASSEMBLYAI_CALLS, d.max_assemblyai_calls)
            ),
            max_assemblyai_audio_seconds=_env_number(
                env, ENV_MAX_ASSEMBLYAI_SECONDS, d.max_assemblyai_audio_seconds
            ),
            max_polly_requests=int(
                _env_number(env, ENV_MAX_POLLY_REQUESTS, d.max_polly_requests)
            ),
            max_polly_characters=int(
                _env_number(env, ENV_MAX_POLLY_CHARS, d.max_polly_characters)
            ),
        )


def estimate_audio_seconds(audio_bytes: bytes) -> float:
    """Duration of WAV audio; falls back to a 16 kHz mono 16-bit estimate."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            rate = wav.getframerate()
            if rate > 0:
                return wav.getnframes() / float(rate)
    except (wave.Error, EOFError):
        pass
    return len(audio_bytes) / 32000.0


@dataclass
class UsageTracker:
    """Counts cloud calls attempted by this session and enforces limits."""

    limits: UsageLimits = field(default_factory=UsageLimits.from_env)
    assemblyai_calls: int = 0
    assemblyai_audio_seconds: float = 0.0
    polly_requests: int = 0
    polly_characters: int = 0
    local_transcriptions: int = 0
    browser_speech_requests: int = 0
    # Digests of recordings already sent to AssemblyAI, and cached Polly audio,
    # so the same input can never trigger a second paid request in a session.
    sent_audio_digests: Set[str] = field(default_factory=set)
    polly_cache: Dict[str, bytes] = field(default_factory=dict)

    def check_assemblyai(self, audio_seconds: float) -> Tuple[bool, str]:
        lim = self.limits
        if self.assemblyai_calls + 1 > lim.max_assemblyai_calls:
            return False, (
                f"AssemblyAI call limit reached ({lim.max_assemblyai_calls} per session)."
            )
        if self.assemblyai_audio_seconds + audio_seconds > lim.max_assemblyai_audio_seconds:
            return False, (
                "AssemblyAI audio-seconds limit would be exceeded "
                f"({lim.max_assemblyai_audio_seconds:.0f}s per session)."
            )
        return True, ""

    def record_assemblyai(self, audio_seconds: float) -> None:
        self.assemblyai_calls += 1
        self.assemblyai_audio_seconds += audio_seconds

    def check_polly(self, characters: int) -> Tuple[bool, str]:
        lim = self.limits
        if self.polly_requests + 1 > lim.max_polly_requests:
            return False, f"Polly request limit reached ({lim.max_polly_requests} per session)."
        if self.polly_characters + characters > lim.max_polly_characters:
            return False, (
                f"Polly character limit would be exceeded ({lim.max_polly_characters} per session)."
            )
        return True, ""

    def record_polly(self, characters: int) -> None:
        self.polly_requests += 1
        self.polly_characters += characters

    def snapshot(self) -> dict:
        lim = self.limits
        return {
            "AssemblyAI calls": f"{self.assemblyai_calls} / {lim.max_assemblyai_calls}",
            "AssemblyAI audio seconds": (
                f"{self.assemblyai_audio_seconds:.1f} / {lim.max_assemblyai_audio_seconds:.0f}"
            ),
            "Polly requests": f"{self.polly_requests} / {lim.max_polly_requests}",
            "Polly characters": f"{self.polly_characters} / {lim.max_polly_characters}",
            "Local Whisper transcriptions": str(self.local_transcriptions),
            "Browser speech requests": str(self.browser_speech_requests),
        }
