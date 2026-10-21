"""Tests for .env configuration loading."""

import os
from pathlib import Path


def test_env_file_exists():
    """Verify .env file exists at repo root."""
    env_file = Path(__file__).parent.parent / ".env"
    assert env_file.exists(), ".env file not found at repo root"


def test_env_contains_required_keys():
    """Verify .env contains all required configuration keys."""
    env_file = Path(__file__).parent.parent / ".env"
    env_content = env_file.read_text()

    required_keys = [
        "LOCAL_IMAGE_EVIDENCE_ENABLED",
        "EXPLANATION_MODE",
        "BEDROCK_ENABLED",
        "GROQ_ENABLED",
    ]

    for key in required_keys:
        assert key in env_content, f"{key} not found in .env"


def test_env_example_exists_and_is_not_ignored():
    """Verify .env.example exists (should not be ignored by .gitignore)."""
    example_file = Path(__file__).parent.parent / ".env.example"
    assert example_file.exists(), ".env.example file not found at repo root"

    # Verify .env.example is explicitly NOT ignored (marked with !)
    gitignore = Path(__file__).parent.parent / ".gitignore"
    gitignore_content = gitignore.read_text()
    assert "!.env.example" in gitignore_content, ".env.example not marked to be tracked in .gitignore"


def test_env_is_ignored_by_git():
    """Verify .env is in .gitignore."""
    gitignore = Path(__file__).parent.parent / ".gitignore"
    gitignore_content = gitignore.read_text()
    assert ".env" in gitignore_content, ".env not found in .gitignore"


def test_env_configuration_values():
    """Verify .env.example template has expected default configuration."""
    env_file = Path(__file__).parent.parent / ".env.example"
    env_vars = {}

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env_vars[key] = value

    assert env_vars.get("LOCAL_IMAGE_EVIDENCE_ENABLED") == "true"
    assert env_vars.get("EXPLANATION_MODE") == "deterministic"
    assert env_vars.get("BEDROCK_ENABLED") == "false"
    assert env_vars.get("GROQ_ENABLED") == "false"


def test_env_loading_in_streamlit_app():
    """Verify streamlit_app.py loads .env at startup."""
    app_file = Path(__file__).parent.parent / "hackathon" / "amazon_alexa" / "streamlit_app.py"
    app_content = app_file.read_text(encoding="utf-8")

    # Verify the .env loading code is present
    assert ".env" in app_content, ".env loading code not found in streamlit_app.py"
    assert "_env_file = _repo_root / \".env\"" in app_content or "_env_file" in app_content
    assert "os.environ" in app_content, "os.environ assignment not found in streamlit_app.py"
