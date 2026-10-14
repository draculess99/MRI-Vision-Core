"""Controlled live validation: ONE AssemblyAI request or ONE Amazon Polly request.

Nothing paid happens unless you pass --confirm-paid. Each run makes at most one
cloud request (limits are forced to 1, fallback is disabled so the result shows
the real provider behaviour) and prints the backend used, success/failure, and
the usage-counter changes. Credentials come from the environment only.

  python -m hackathon.amazon_alexa.live_validation diagnose        # no network, no audio
  python -m hackathon.amazon_alexa.live_validation stt --wav clip.wav [--confirm-paid]
  python -m hackathon.amazon_alexa.live_validation tts [--text "..."] [--out out.mp3] [--confirm-paid]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from hackathon.amazon_alexa import stt, tts
from hackathon.amazon_alexa.usage import UsageLimits, UsageTracker, estimate_audio_seconds

DEFAULT_TEXT = "This is a synthetic, non-diagnostic demonstration. Human approval is required."
MAX_WAV_SECONDS = 30.0


def _show(label: str, usage: UsageTracker, before: dict) -> None:
    after = usage.snapshot()
    print(f"\nUsage counters ({label}):")
    for name, new in after.items():
        old = before[name]
        print(f"  {name}: {old} -> {new}" + ("   <-- changed" if old != new else ""))
    print("\nNote: these are application-level counters only. Check provider dashboards "
          "(AssemblyAI usage, AWS Billing/Budgets) for actual charges.")


def run_stt(wav: Path, confirm: bool) -> int:
    key_set = stt.get_assemblyai_key() is not None
    audio = wav.read_bytes()
    seconds = estimate_audio_seconds(audio)
    print(f"Backend requested : {stt.BACKEND_ASSEMBLYAI} (cloud, paid credits)")
    print(f"{stt.ENV_ASSEMBLYAI_KEY}: {'configured' if key_set else 'NOT SET'}")
    print(f"Audio             : {wav.name}, ~{seconds:.1f}s, {len(audio)} bytes")
    if seconds > MAX_WAV_SECONDS:
        print(f"Refusing: clip longer than {MAX_WAV_SECONDS:.0f}s.")
        return 2
    if not confirm:
        print("\nDRY RUN: nothing was sent. Re-run with --confirm-paid to make ONE AssemblyAI request.")
        return 0
    usage = UsageTracker(limits=UsageLimits(max_assemblyai_calls=1, max_assemblyai_audio_seconds=MAX_WAV_SECONDS))
    before = usage.snapshot()
    result = stt.transcribe(audio, stt.BACKEND_ASSEMBLYAI, usage, allow_fallback=False, cloud_confirmed=True)
    print(f"\nBackend used      : {result.backend or 'none'} ({'cloud' if result.is_cloud else 'local'})")
    print(f"Success           : {result.ok}")
    print(f"Transcript        : {result.text!r}" if result.ok else f"Error             : {result.error}")
    _show("AssemblyAI", usage, before)
    return 0 if result.ok else 1


def run_tts(text: str, out: Path, confirm: bool) -> int:
    print(f"Backend requested : {tts.BACKEND_POLLY} (cloud, billed per character by AWS)")
    print(f"Voice/engine      : {os.environ.get(tts.ENV_POLLY_VOICE, tts.DEFAULT_POLLY_VOICE)} / {os.environ.get(tts.ENV_POLLY_ENGINE, 'neural')}")
    print(f"Characters        : {len(text)}")
    if len(text) > 200:
        print("Refusing: text longer than 200 characters.")
        return 2
    if not confirm:
        print("\nDRY RUN: nothing was sent. Re-run with --confirm-paid to make ONE Polly request.")
        return 0
    usage = UsageTracker(limits=UsageLimits(max_polly_requests=1, max_polly_characters=200))
    before = usage.snapshot()
    speech = tts.speak(text, tts.BACKEND_POLLY, usage, polly_enabled=True)
    ok = speech.backend == tts.BACKEND_POLLY and bool(speech.audio_mp3)
    print(f"\nBackend used      : {speech.backend} ({'cloud' if speech.is_cloud else 'local'})")
    print(f"Success           : {ok}")
    if ok:
        out.write_bytes(speech.audio_mp3)
        print(f"Audio saved       : {out} ({len(speech.audio_mp3)} bytes)")
    else:
        print(f"Error             : {speech.error}")
    _show("Amazon Polly", usage, before)
    return 0 if ok else 1


def run_diagnose() -> int:
    """Print the auth configuration. Makes no network request; never prints secrets."""
    print("AssemblyAI authentication diagnostics (no network request is made)\n")
    for name, value in stt.assemblyai_auth_diagnostics().items():
        print(f"  {name}: {value}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("diagnose", help="print auth configuration (no network)")
    s = sub.add_parser("stt", help="one AssemblyAI request")
    s.add_argument("--wav", type=Path, required=True)
    s.add_argument("--confirm-paid", action="store_true")
    t = sub.add_parser("tts", help="one Amazon Polly request")
    t.add_argument("--text", default=DEFAULT_TEXT)
    t.add_argument("--out", type=Path, default=Path("polly_test.mp3"))
    t.add_argument("--confirm-paid", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "diagnose":
        return run_diagnose()
    if a.cmd == "stt":
        return run_stt(a.wav, a.confirm_paid)
    return run_tts(a.text, a.out, a.confirm_paid)


if __name__ == "__main__":
    sys.exit(main())
