"""Speech-to-text backends: local faster-whisper (default) and AssemblyAI.

The cloud backend is only invoked when the caller explicitly selects it, passes
recorded audio, and the usage safeguards allow it. There is no streaming and no
background polling. If the cloud path fails for any reason the transcriber
falls back to local Whisper when it is installed.

Credentials are read only from the ASSEMBLYAI_API_KEY environment variable and
are never included in results or error messages.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional
from urllib.parse import urlparse

from hackathon.amazon_alexa.usage import UsageTracker, estimate_audio_seconds

BACKEND_LOCAL = "local_whisper"
BACKEND_ASSEMBLYAI = "assemblyai"
STT_BACKENDS = (BACKEND_LOCAL, BACKEND_ASSEMBLYAI)
DEFAULT_STT_BACKEND = BACKEND_LOCAL

ENV_ASSEMBLYAI_KEY = "ASSEMBLYAI_API_KEY"
ENV_WHISPER_MODEL = "ALEXA_WHISPER_MODEL"
DEFAULT_WHISPER_MODEL = "base.en"

# (api_key, audio_bytes) -> transcript text; raises on any failure.
CloudTranscribeFn = Callable[[str, bytes], str]
# (audio_bytes) -> transcript text; raises BackendUnavailable if not installed.
LocalTranscribeFn = Callable[[bytes], str]


class BackendUnavailable(RuntimeError):
    """A transcription backend cannot be used (missing package, key, limit)."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    backend: str  # backend that actually produced the text ("" if none)
    is_cloud: bool
    requested_backend: str
    fallback_used: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and self.error is None


def get_assemblyai_key(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    env = os.environ if env is None else env
    key = env.get(ENV_ASSEMBLYAI_KEY, "").strip()
    return key or None


def local_whisper_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        return False


def _redact(message: str, secret: Optional[str]) -> str:
    if secret:
        message = message.replace(secret, "[redacted]")
    return message


def assemblyai_transcribe(api_key: str, audio_bytes: bytes) -> str:
    """Real AssemblyAI call (one-shot upload + transcribe, no streaming)."""
    try:
        import assemblyai as aai
    except ImportError as exc:
        raise BackendUnavailable("The 'assemblyai' package is not installed.") from exc

    # Explicit per-call credentials: no dependence on mutable global SDK settings.
    transcript = aai.Transcriber(api_key=api_key).transcribe(io.BytesIO(audio_bytes))
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(transcript.error or "AssemblyAI transcription failed.")
    return transcript.text or ""


def key_fingerprint(key: str) -> str:
    """Short SHA-256 prefix: lets two keys be compared without revealing either."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def assemblyai_auth_diagnostics(env: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    """Describe the AssemblyAI auth configuration without any network call.

    Never includes the key, the Authorization header, or any token. Builds the
    SDK client exactly as the transcription path does and inspects the headers
    and endpoint it would use.
    """
    env = os.environ if env is None else env
    raw = env.get(ENV_ASSEMBLYAI_KEY, "")
    key = raw.strip()
    info: Dict[str, object] = {
        "env_var_read": ENV_ASSEMBLYAI_KEY,
        "key_present": bool(key),
        "key_length": len(key),
        "key_fingerprint_sha256_8": key_fingerprint(key) if key else None,
        "had_surrounding_whitespace": raw != key,
        "has_quote_chars": any(c in key for c in "\"'`"),
        "has_non_ascii_or_inner_whitespace": any((not c.isascii()) or c.isspace() for c in key),
        "other_assemblyai_env_vars": sorted(
            k for k in env if k.upper().startswith("ASSEMBLYAI") and k != ENV_ASSEMBLYAI_KEY
        ),
    }
    try:
        import importlib.metadata as md

        import assemblyai as aai

        info["sdk_version"] = md.version("assemblyai")
        client = aai.Client(api_key=key) if key else None
        settings = client.settings if client else aai.Settings()
        info["endpoint_host"] = urlparse(settings.base_url).hostname
        if client is not None:
            sent = client.http_client.headers.get("authorization", "")
            info["authorization_header_matches_key"] = sent == key
            info["authorization_header_length"] = len(sent)
    except ImportError:
        info["sdk_version"] = None
    except Exception as exc:  # noqa: BLE001
        info["diagnostics_error"] = type(exc).__name__
    return info


def classify_assemblyai_error(exc: BaseException) -> str:
    """Label a cloud failure so auth problems are not confused with other causes."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = next((a for a in getattr(exc, "args", ()) if isinstance(a, int)), None)
    text = str(exc).lower()
    if status == 401 or "invalid api key" in text or "unauthorized" in text:
        return "AUTHENTICATION (HTTP 401): the service rejected the key. Run the 'diagnose' command and compare the fingerprint."
    if status == 403 or "forbidden" in text or "permission" in text:
        return "PERMISSION (HTTP 403): the key is recognised but lacks access (plan, credits or feature)."
    if status == 402 or "insufficient" in text or "credit" in text or "billing" in text:
        return "BILLING (HTTP 402): the account has no available credits."
    if status == 429:
        return "RATE LIMIT (HTTP 429): too many requests."
    if status is not None and status >= 500:
        return f"SERVICE (HTTP {status}): AssemblyAI server-side problem."
    if isinstance(exc, (TypeError, AttributeError, ImportError)):
        return "SDK INCOMPATIBILITY: the installed assemblyai package does not match this code."
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)) or "connect" in type(exc).__name__.lower() or "timeout" in type(exc).__name__.lower():
        return "NETWORK: could not reach the AssemblyAI endpoint."
    return "UNCLASSIFIED provider error."


_WHISPER_MODELS: Dict[str, object] = {}


def local_whisper_transcribe(audio_bytes: bytes) -> str:
    """Offline transcription with faster-whisper (CPU, int8)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise BackendUnavailable(
            "faster-whisper is not installed (pip install faster-whisper)."
        ) from exc

    name = os.environ.get(ENV_WHISPER_MODEL, DEFAULT_WHISPER_MODEL)
    model = _WHISPER_MODELS.get(name)
    if model is None:
        model = WhisperModel(name, device="cpu", compute_type="int8")
        _WHISPER_MODELS[name] = model
    try:
        segments, _info = model.transcribe(io.BytesIO(audio_bytes), beam_size=1)
        return " ".join(seg.text.strip() for seg in segments).strip()
    except TypeError as exc:
        # faster-whisper 1.2.x calls av.open(metadata_errors=...), which PyAV >= 19 rejects.
        if "metadata_errors" in str(exc):
            raise BackendUnavailable(
                'Incompatible PyAV version for faster-whisper. Run: pip install "av<19"'
            ) from exc
        raise


def _run_local(audio_bytes: bytes, local_fn: LocalTranscribeFn, usage: UsageTracker) -> str:
    text = local_fn(audio_bytes)
    usage.local_transcriptions += 1
    return text


def transcribe(
    audio_bytes: Optional[bytes],
    backend: str = DEFAULT_STT_BACKEND,
    usage: Optional[UsageTracker] = None,
    *,
    allow_fallback: bool = True,
    cloud_confirmed: bool = False,
    env: Optional[Mapping[str, str]] = None,
    cloud_fn: Optional[CloudTranscribeFn] = None,
    local_fn: Optional[LocalTranscribeFn] = None,
) -> TranscriptionResult:
    """Transcribe recorded audio. Never raises; failures are in `error`.

    The cloud backend is called only if `cloud_confirmed` is True (the caller's
    explicit confirmation), a key is configured, the limits allow it, and this
    exact recording has not already been sent in this session.
    """
    usage = usage if usage is not None else UsageTracker()
    cloud_fn = cloud_fn or assemblyai_transcribe  # resolved at call time
    local_fn = local_fn or local_whisper_transcribe

    if backend not in STT_BACKENDS:
        return TranscriptionResult("", "", False, backend, error=f"Unknown STT backend: {backend}")
    if not audio_bytes:
        return TranscriptionResult("", "", False, backend, error="No audio was recorded.")

    cloud_error: Optional[str] = None
    if backend == BACKEND_ASSEMBLYAI:
        if not cloud_confirmed:
            return TranscriptionResult(
                "", "", True, backend,
                error="Cloud transcription needs explicit confirmation; nothing was sent.",
            )
        api_key = get_assemblyai_key(env)
        seconds = estimate_audio_seconds(audio_bytes)
        digest = hashlib.sha256(audio_bytes).hexdigest()
        if api_key is None:
            cloud_error = f"{ENV_ASSEMBLYAI_KEY} is not set."
        elif digest in usage.sent_audio_digests:
            cloud_error = "This recording was already sent to AssemblyAI; not sending it again."
        else:
            allowed, reason = usage.check_assemblyai(seconds)
            if not allowed:
                cloud_error = reason
            else:
                usage.record_assemblyai(seconds)  # count the attempt, even if it fails
                usage.sent_audio_digests.add(digest)
                try:
                    text = cloud_fn(api_key, audio_bytes)
                    if text and text.strip():
                        return TranscriptionResult(text.strip(), BACKEND_ASSEMBLYAI, True, backend)
                    cloud_error = "AssemblyAI returned an empty transcript."
                except Exception as exc:  # noqa: BLE001 - provider/network errors vary widely
                    cloud_error = _redact(
                        f"AssemblyAI failed [{classify_assemblyai_error(exc)}] {type(exc).__name__}: {exc}", api_key
                    )

        if not allow_fallback:
            return TranscriptionResult("", "", True, backend, error=cloud_error)
        try:
            text = _run_local(audio_bytes, local_fn, usage)
        except BackendUnavailable as exc:
            return TranscriptionResult(
                "", "", True, backend, error=f"{cloud_error} Local fallback unavailable: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            return TranscriptionResult(
                "", "", True, backend,
                error=f"{cloud_error} Local fallback failed: {type(exc).__name__}: {exc}",
            )
        if not text.strip():
            return TranscriptionResult(
                "", "", False, backend, fallback_used=True,
                error=f"{cloud_error} Local fallback produced an empty transcript.",
            )
        return TranscriptionResult(
            text.strip(), BACKEND_LOCAL, False, backend, fallback_used=True,
            error=None,
        )

    # Local backend requested.
    try:
        text = _run_local(audio_bytes, local_fn, usage)
    except BackendUnavailable as exc:
        return TranscriptionResult("", "", False, backend, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return TranscriptionResult(
            "", "", False, backend, error=f"Local Whisper failed: {type(exc).__name__}: {exc}"
        )
    if not text.strip():
        return TranscriptionResult("", "", False, backend, error="No speech was recognised.")
    return TranscriptionResult(text.strip(), BACKEND_LOCAL, False, backend)
