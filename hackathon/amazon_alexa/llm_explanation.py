"""Optional LLM-based explanation providers for MRI evidence.

Provides enhanced explanations via Amazon Bedrock (primary) or Groq (fallback),
with deterministic review support as the final fallback.

Safety: LLMs receive only case ID, measurements, status, and non-diagnostic
evidence. They cannot approve, diagnose, recommend treatment, or access reviewer
identity. All output is validated and length-limited.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Optional

from hackathon.amazon_alexa.review_agents import ReviewSupportPacket

logger = logging.getLogger(__name__)


# Environment variable names
ENV_EXPLANATION_MODE = "EXPLANATION_MODE"
ENV_BEDROCK_ENABLED = "BEDROCK_ENABLED"
ENV_BEDROCK_REGION = "AWS_REGION"
ENV_BEDROCK_MODEL_ID = "BEDROCK_MODEL_ID"
ENV_GROQ_ENABLED = "GROQ_ENABLED"
ENV_GROQ_API_KEY = "GROQ_API_KEY"
ENV_GROQ_MODEL_ID = "GROQ_MODEL_ID"
ENV_LLM_TIMEOUT = "LLM_TIMEOUT_SECONDS"

# Explanation mode constants
MODE_DETERMINISTIC = "deterministic"
MODE_AUTO = "auto"

# Explanation constraints
MAX_EXPLANATION_LENGTH = 500
DEFAULT_LLM_TIMEOUT = 10
MIN_TIMEOUT = 1
MAX_TIMEOUT = 60


@dataclass(frozen=True)
class EnhancedExplanation:
    """LLM-generated explanation with source attribution."""

    text: str
    """Plain-text explanation from the LLM or deterministic fallback."""

    source: str
    """'Amazon Bedrock', 'Groq', or 'Deterministic fallback'."""

    valid: bool = True
    """True if output passed validation and safety checks."""

    error: Optional[str] = None
    """Error message if validation failed."""


def _validate_explanation_output(text: str, max_len: int = MAX_EXPLANATION_LENGTH) -> tuple[bool, Optional[str]]:
    """Validate LLM output against safety constraints.

    Args:
        text: Output from LLM
        max_len: Maximum allowed length

    Returns:
        (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Empty response from provider."

    if len(text) > max_len:
        return False, f"Response exceeds maximum length ({len(text)} > {max_len})."

    # Check for approval-like language
    approval_patterns = [
        r"\bapprove[d]?\b",
        r"\bapproval\b",
        r"\bapproved_by_human\b",
        r"\bauthori[zs]",  # authorize, authorization
        r"sign.*off",  # sign off, signed off, etc.
    ]
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in approval_patterns):
        return False, "Response contains approval language (not allowed)."

    # Check for diagnostic/treatment language
    diagnostic_patterns = [
        r"\bdiagnose[sd]?\b",
        r"\bdiagnosis\b",
        r"\btreatment\b",
        r"\brecommend[ed]?\b\s+(treatment|therapy|procedure|surgery|medication)",
        r"\b(should|must)\s+(be\s+)?treated\b",
        r"\bshould\s+receive\s+(therapy|treatment)\b",
    ]
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in diagnostic_patterns):
        return False, "Response contains diagnostic or treatment language (not allowed)."

    return True, None


def _get_explanation_mode(env: Optional[dict[str, str]] = None) -> str:
    """Get explanation mode from environment.

    Args:
        env: Environment dict (defaults to os.environ)

    Returns:
        'deterministic' (default) or 'auto'
    """
    env = os.environ if env is None else env
    mode = env.get(ENV_EXPLANATION_MODE, MODE_DETERMINISTIC).strip().lower()
    return mode if mode in (MODE_DETERMINISTIC, MODE_AUTO) else MODE_DETERMINISTIC


def _is_bedrock_enabled(env: Optional[dict[str, str]] = None) -> bool:
    """Check if Bedrock is enabled.

    Args:
        env: Environment dict (defaults to os.environ)

    Returns:
        True only if BEDROCK_ENABLED is explicitly 'true'
    """
    env = os.environ if env is None else env
    value = env.get(ENV_BEDROCK_ENABLED, "false").strip().lower()
    return value == "true"


def _get_bedrock_region(env: Optional[dict[str, str]] = None) -> Optional[str]:
    """Get AWS region for Bedrock from environment.

    Args:
        env: Environment dict (defaults to os.environ)

    Returns:
        Region string or None if not set
    """
    env = os.environ if env is None else env
    region = env.get(ENV_BEDROCK_REGION, "").strip()
    return region or None


def _get_bedrock_model_id(env: Optional[dict[str, str]] = None) -> Optional[str]:
    """Get Bedrock model ID from environment.

    Args:
        env: Environment dict (defaults to os.environ)

    Returns:
        Model ID or None if not set
    """
    env = os.environ if env is None else env
    model_id = env.get(ENV_BEDROCK_MODEL_ID, "").strip()
    return model_id or None


def _is_groq_enabled(env: Optional[dict[str, str]] = None) -> bool:
    """Check if Groq is enabled.

    Args:
        env: Environment dict (defaults to os.environ)

    Returns:
        True only if GROQ_ENABLED is explicitly 'true'
    """
    env = os.environ if env is None else env
    value = env.get(ENV_GROQ_ENABLED, "false").strip().lower()
    return value == "true"


def _get_groq_api_key(env: Optional[dict[str, str]] = None) -> Optional[str]:
    """Get Groq API key from environment.

    Args:
        env: Environment dict (defaults to os.environ)

    Returns:
        API key or None if not set (never the actual key in return value)
    """
    env = os.environ if env is None else env
    key = env.get(ENV_GROQ_API_KEY, "").strip()
    return key or None


def _get_groq_model_id(env: Optional[dict[str, str]] = None) -> Optional[str]:
    """Get Groq model ID from environment.

    Args:
        env: Environment dict (defaults to os.environ)

    Returns:
        Model ID or None if not set
    """
    env = os.environ if env is None else env
    model_id = env.get(ENV_GROQ_MODEL_ID, "").strip()
    return model_id or None


def _get_llm_timeout(env: Optional[dict[str, str]] = None) -> int:
    """Get LLM timeout in seconds from environment.

    Args:
        env: Environment dict (defaults to os.environ)

    Returns:
        Timeout in seconds, bounded to [1, 60], defaults to 10 on invalid input
    """
    env = os.environ if env is None else env
    timeout_str = env.get(ENV_LLM_TIMEOUT, "").strip()
    if not timeout_str:
        return DEFAULT_LLM_TIMEOUT
    try:
        timeout = int(timeout_str)
        if MIN_TIMEOUT <= timeout <= MAX_TIMEOUT:
            return timeout
    except ValueError:
        pass
    return DEFAULT_LLM_TIMEOUT


def _call_bedrock(packet: ReviewSupportPacket, env: Optional[dict[str, str]] = None) -> Optional[str]:
    """Call Amazon Bedrock for enhanced explanation.

    Args:
        packet: Review support packet with evidence and findings
        env: Environment dict (defaults to os.environ)

    Returns:
        Explanation text or None on failure
    """
    env = os.environ if env is None else env

    # Check if Bedrock is enabled and fully configured
    if not _is_bedrock_enabled(env):
        return None

    region = _get_bedrock_region(env)
    model_id = _get_bedrock_model_id(env)
    if not region or not model_id:
        return None

    try:
        import boto3
    except ImportError:
        return None

    try:
        timeout = _get_llm_timeout(env)
        client = boto3.client("bedrock-runtime", region_name=region)

        # Compact, non-diagnostic prompt (keep short to fit response in 500-char limit)
        prompt = f"""Review assistant. Concise summary only. No diagnosis, treatment, or approval language.

Evidence: {packet.evidence_summary[:100]}

Key findings: {', '.join(packet.critic_findings[:2])}

Summary (max 150 chars):"""

        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 200},
        )

        if response.get("output") and response["output"].get("message"):
            content = response["output"]["message"].get("content", [])
            if content and len(content) > 0:
                text = content[0].get("text", "").strip()
                return text if text else None

    except Exception as e:
        # Log safe failure category only (no message, body, credentials, or payloads)
        error_code = None
        if hasattr(e, "response") and isinstance(e.response, dict):
            error_code = e.response.get("Error", {}).get("Code")
        if error_code:
            logger.debug(f"Bedrock failed with error code: {error_code}")
        else:
            logger.debug(f"Bedrock failed: {type(e).__name__}")
        # Will fallback to Groq

    return None


def _call_groq(packet: ReviewSupportPacket, env: Optional[dict[str, str]] = None) -> Optional[str]:
    """Call Groq API for enhanced explanation.

    Args:
        packet: Review support packet with evidence and findings
        env: Environment dict (defaults to os.environ)

    Returns:
        Explanation text or None on failure
    """
    env = os.environ if env is None else env

    # Check if Groq is enabled and fully configured
    if not _is_groq_enabled(env):
        return None

    api_key = _get_groq_api_key(env)
    model_id = _get_groq_model_id(env)
    if not api_key or not model_id:
        return None

    try:
        timeout = _get_llm_timeout(env)

        # Non-diagnostic prompt with plain-language output instructions
        prompt = f"""Review assistant. No diagnosis, treatment, or approval language.

Evidence: {packet.evidence_summary[:100]}

Key findings: {', '.join(packet.critic_findings[:2])}

Write exactly three short sentences in clear, plain language for a non-specialist workflow user.
Explain the supplied evidence and deterministic review status without listing raw metric values unless essential.
State what a named human reviewer should verify next.
State that this copilot does not diagnose, make clinical decisions, or approve cases.
Do not introduce diagnoses, new findings, probabilities, or recommendations beyond the supplied evidence."""

        body = json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.3,
        })

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body.encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "MRI-Evidence-Copilot/1.0",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode())
            if result.get("choices") and len(result["choices"]) > 0:
                text = result["choices"][0].get("message", {}).get("content", "").strip()
                if text:
                    valid, error = _validate_explanation_output(text)
                    if valid:
                        return text
                    logger.warning("Groq response rejected by output validation")
                    return None
                return None

    except Exception as e:
        # Log safe failure category only (no message, URL, headers, or payloads)
        http_status = None
        if hasattr(e, "code"):
            http_status = e.code
        if http_status:
            logger.warning(f"Groq failed with HTTP status: {http_status}")
        else:
            logger.warning(f"Groq failed: {type(e).__name__}")
        # Will fallback to deterministic

    return None


def generate_enhanced_explanation(
    packet: ReviewSupportPacket, env: Optional[dict[str, str]] = None
) -> EnhancedExplanation:
    """Generate enhanced explanation with configurable provider chain.

    Respects EXPLANATION_MODE environment variable:
    - 'deterministic' (default): always returns deterministic fallback
    - 'auto': tries Bedrock → Groq → deterministic based on enabled/configured providers

    Args:
        packet: Review support packet with evidence and findings
        env: Environment dict (defaults to os.environ)

    Returns:
        EnhancedExplanation with text and source attribution
    """
    env = os.environ if env is None else env

    # Check explanation mode
    mode = _get_explanation_mode(env)
    if mode == MODE_DETERMINISTIC:
        # Deterministic-only mode: skip LLM providers
        fallback_text = _deterministic_explanation(packet)
        return EnhancedExplanation(text=fallback_text, source="Deterministic fallback")

    # Auto mode: try enabled providers in order
    # Try Bedrock
    text = _call_bedrock(packet, env)
    if text:
        valid, error = _validate_explanation_output(text)
        if valid:
            return EnhancedExplanation(text=text, source="Amazon Bedrock")
        # Invalid output; fall through to Groq

    # Try Groq
    text = _call_groq(packet, env)
    if text:
        valid, error = _validate_explanation_output(text)
        if valid:
            return EnhancedExplanation(text=text, source="Groq")
        # Invalid output; fall through to deterministic

    # Deterministic fallback: synthesize from packet
    fallback_text = _deterministic_explanation(packet)
    return EnhancedExplanation(text=fallback_text, source="Deterministic fallback")


def _deterministic_explanation(packet: ReviewSupportPacket) -> str:
    """Generate a deterministic explanation from packet contents.

    Args:
        packet: Review support packet

    Returns:
        Plain-text explanation
    """
    lines = [
        "Evidence Review Summary:",
        packet.evidence_summary,
        "",
        "Critic Findings:",
    ]
    lines.extend(f"• {f}" for f in packet.critic_findings[:3])
    lines.append("")
    lines.append("Reviewer Verification Steps:")
    lines.extend(f"• {c}" for c in packet.reviewer_checks[:3])
    lines.append("")
    lines.append("A named human reviewer must evaluate the source images and make the final decision.")

    return "\n".join(lines)
