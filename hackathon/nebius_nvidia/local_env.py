"""Load Nebius configuration from .env file in repository root.

Environment variables take precedence over .env file values.
Secrets are never printed or exposed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Mapping


def load_env_file(env_path: Optional[str] = None) -> dict[str, str]:
    """Load key=value pairs from .env file; environment takes precedence."""
    env_vars = {}

    if env_path is None:
        root = Path(__file__).parent.parent.parent
        env_path = root / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        return env_vars

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                env_vars[key] = value

    return env_vars


def get_nebius_config(environ: Optional[Mapping[str, str]] = None) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Get Nebius config (api_key, base_url, model) from environment or .env.

    When environ is explicitly provided, ONLY that mapping is used (for test isolation).
    When environ is None, process environment is checked, falling back to .env file.
    Returns (api_key, base_url, model) tuple with None for missing values.
    Secrets are never printed.
    """
    if environ is not None:
        api_key = environ.get("NEBIUS_API_KEY") or ""
        base_url = environ.get("NEBIUS_BASE_URL") or ""
        model = environ.get("NEBIUS_MODEL") or ""
    else:
        env = os.environ
        env_file_vars = load_env_file()

        api_key = env.get("NEBIUS_API_KEY") or env_file_vars.get("NEBIUS_API_KEY") or ""
        base_url = env.get("NEBIUS_BASE_URL") or env_file_vars.get("NEBIUS_BASE_URL") or ""
        model = env.get("NEBIUS_MODEL") or env_file_vars.get("NEBIUS_MODEL") or ""

    return (api_key.strip() or None,
            base_url.strip() or None,
            model.strip() or None)


def get_nebius_config_from_env_file() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Get Nebius config ONLY from local .env file, never from process environment.

    This is the Streamlit mode: load configuration from repository-root .env only.
    Returns (api_key, base_url, model) tuple with None for missing values.
    Secrets are never printed.
    """
    env_file_vars = load_env_file()

    api_key = env_file_vars.get("NEBIUS_API_KEY") or ""
    base_url = env_file_vars.get("NEBIUS_BASE_URL") or ""
    model = env_file_vars.get("NEBIUS_MODEL") or ""

    return (api_key.strip() or None,
            base_url.strip() or None,
            model.strip() or None)
