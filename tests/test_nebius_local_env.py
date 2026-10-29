"""Tests for Nebius .env file loader and configuration."""

import tempfile
from pathlib import Path

import pytest

from hackathon.nebius_nvidia.local_env import load_env_file, get_nebius_config, get_nebius_config_from_env_file


def test_load_env_file_empty_path_not_found():
    """load_env_file with no args returns empty dict if .env doesn't exist."""
    result = load_env_file()
    assert result == {} or isinstance(result, dict)


def test_load_env_file_from_explicit_path():
    """load_env_file reads key=value pairs from specified file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("KEY1=value1\n")
        f.write("KEY2=value2\n")
        f.write("# comment\n")
        f.write("KEY3=value3\n")
        f.write("\n")
        temp_path = f.name

    try:
        result = load_env_file(temp_path)
        assert result == {"KEY1": "value1", "KEY2": "value2", "KEY3": "value3"}
    finally:
        Path(temp_path).unlink()


def test_load_env_file_strips_whitespace():
    """load_env_file strips whitespace around keys and values."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("  KEY1  =  value1  \n")
        f.write("KEY2=  value2\n")
        temp_path = f.name

    try:
        result = load_env_file(temp_path)
        assert result["KEY1"] == "value1"
        assert result["KEY2"] == "value2"
    finally:
        Path(temp_path).unlink()


def test_load_env_file_ignores_comments_and_empty_lines():
    """load_env_file skips comments and empty lines."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("# this is a comment\n")
        f.write("\n")
        f.write("KEY1=value1\n")
        f.write("  \n")
        f.write("# another comment\n")
        f.write("KEY2=value2\n")
        temp_path = f.name

    try:
        result = load_env_file(temp_path)
        assert result == {"KEY1": "value1", "KEY2": "value2"}
    finally:
        Path(temp_path).unlink()


def test_load_env_file_handles_equals_in_value():
    """load_env_file correctly splits on first = only."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("URL=https://api.example.com?param=value\n")
        f.write("FORMULA=a=b+c\n")
        temp_path = f.name

    try:
        result = load_env_file(temp_path)
        assert result["URL"] == "https://api.example.com?param=value"
        assert result["FORMULA"] == "a=b+c"
    finally:
        Path(temp_path).unlink()


def test_get_nebius_config_from_env_only():
    """get_nebius_config returns config when all three vars are in process environ."""
    env = {
        "NEBIUS_API_KEY": "test-key",
        "NEBIUS_BASE_URL": "https://api.test",
        "NEBIUS_MODEL": "test-model",
    }
    api_key, base_url, model = get_nebius_config(environ=env)
    assert api_key == "test-key"
    assert base_url == "https://api.test"
    assert model == "test-model"


def test_get_nebius_config_missing_returns_none():
    """get_nebius_config returns None for missing values."""
    env = {"NEBIUS_API_KEY": "key"}
    api_key, base_url, model = get_nebius_config(environ=env)
    assert api_key == "key"
    assert base_url is None
    assert model is None


def test_get_nebius_config_empty_string_treated_as_missing():
    """get_nebius_config treats empty strings as missing."""
    env = {
        "NEBIUS_API_KEY": "",
        "NEBIUS_BASE_URL": "https://api.test",
        "NEBIUS_MODEL": "model",
    }
    api_key, base_url, model = get_nebius_config(environ=env)
    assert api_key is None
    assert base_url == "https://api.test"
    assert model == "model"


def test_get_nebius_config_whitespace_only_treated_as_missing():
    """get_nebius_config treats whitespace-only values as missing."""
    env = {
        "NEBIUS_API_KEY": "   ",
        "NEBIUS_BASE_URL": "https://api.test",
        "NEBIUS_MODEL": "model",
    }
    api_key, base_url, model = get_nebius_config(environ=env)
    assert api_key is None
    assert base_url == "https://api.test"
    assert model == "model"


def test_get_nebius_config_process_env_takes_precedence():
    """get_nebius_config prefers process environment over .env."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("NEBIUS_API_KEY=file-key\n")
        f.write("NEBIUS_BASE_URL=https://file.api\n")
        f.write("NEBIUS_MODEL=file-model\n")
        temp_path = f.name

    try:
        env_vars = {
            "NEBIUS_API_KEY": "process-key",
            "NEBIUS_BASE_URL": "https://file.api",
        }
        import os
        original_env = os.environ.copy()
        try:
            os.environ.update(env_vars)

            api_key, base_url, model = get_nebius_config(environ=env_vars)
            assert api_key == "process-key"
            assert base_url == "https://file.api"
        finally:
            os.environ.clear()
            os.environ.update(original_env)
    finally:
        Path(temp_path).unlink()


def test_get_nebius_config_no_secrets_exposed():
    """get_nebius_config returns values, never prints secrets."""
    env = {
        "NEBIUS_API_KEY": "secret-key-12345",
        "NEBIUS_BASE_URL": "https://api.test",
        "NEBIUS_MODEL": "model",
    }
    result = get_nebius_config(environ=env)
    assert result[0] == "secret-key-12345"
    assert "secret-key-12345" not in str(result) or result[0] in str(result)


def test_get_nebius_config_from_env_file_loads_only_from_file():
    """get_nebius_config_from_env_file loads only from .env, not process environ."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("NEBIUS_API_KEY=file-key\n")
        f.write("NEBIUS_BASE_URL=https://file.api\n")
        f.write("NEBIUS_MODEL=file-model\n")
        temp_path = f.name

    try:
        import os
        from hackathon.nebius_nvidia.local_env import load_env_file

        env_vars = load_env_file(temp_path)
        assert env_vars["NEBIUS_API_KEY"] == "file-key"
        assert env_vars["NEBIUS_BASE_URL"] == "https://file.api"
        assert env_vars["NEBIUS_MODEL"] == "file-model"
    finally:
        Path(temp_path).unlink()


def test_get_nebius_config_from_env_file_ignores_process_environ():
    """Prove that get_nebius_config_from_env_file ignores process environ."""
    import os

    original_env = os.environ.copy()
    try:
        os.environ["NEBIUS_API_KEY"] = "process-key-should-be-ignored"
        os.environ["NEBIUS_BASE_URL"] = "https://process.api"
        os.environ["NEBIUS_MODEL"] = "process-model"

        api_key, base_url, model = get_nebius_config_from_env_file()

        assert api_key != "process-key-should-be-ignored"
        assert base_url != "https://process.api"
        assert model != "process-model"

    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_get_nebius_config_from_env_file_returns_none_for_missing():
    """get_nebius_config_from_env_file returns None for missing values."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("NEBIUS_API_KEY=only-key\n")
        temp_path = f.name

    try:
        from unittest.mock import patch

        with patch("hackathon.nebius_nvidia.local_env.load_env_file") as mock_load:
            mock_load.return_value = {"NEBIUS_API_KEY": "only-key"}
            api_key, base_url, model = get_nebius_config_from_env_file()
            assert api_key == "only-key"
            assert base_url is None
            assert model is None
    finally:
        Path(temp_path).unlink()


def test_streamlit_mode_cannot_be_overridden_by_process_environ():
    """Streamlit mode enforces .env-only configuration isolation."""
    import os

    original_env = os.environ.copy()
    try:
        os.environ["NEBIUS_API_KEY"] = "should-not-be-used"
        os.environ["NEBIUS_BASE_URL"] = "https://should-not-be-used.api"
        os.environ["NEBIUS_MODEL"] = "should-not-be-used"

        api_key, base_url, model = get_nebius_config_from_env_file()

        assert api_key is None or api_key != "should-not-be-used"
        assert base_url is None or base_url != "https://should-not-be-used.api"
        assert model is None or model != "should-not-be-used"

    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_explicit_environ_mapping_never_falls_through_to_env_file():
    """Regression: explicit environ mapping must NOT fall back to .env file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("NEBIUS_API_KEY=file-secret-key\n")
        f.write("NEBIUS_BASE_URL=https://file.api\n")
        f.write("NEBIUS_MODEL=file-model\n")
        temp_path = f.name

    try:
        from unittest.mock import patch

        test_mapping = {"NEBIUS_API_KEY": "test-key"}

        with patch("hackathon.nebius_nvidia.local_env.load_env_file") as mock_load:
            mock_load.return_value = {"NEBIUS_API_KEY": "file-secret-key", "NEBIUS_BASE_URL": "https://file.api", "NEBIUS_MODEL": "file-model"}

            api_key, base_url, model = get_nebius_config(environ=test_mapping)

            assert api_key == "test-key"
            assert base_url is None
            assert model is None
    finally:
        Path(temp_path).unlink()


def test_explicit_environ_with_empty_strings_not_overridden_by_env_file():
    """Regression: empty string in explicit mapping should NOT fall back to .env."""
    test_mapping = {
        "NEBIUS_API_KEY": "",
        "NEBIUS_BASE_URL": "https://test.api",
        "NEBIUS_MODEL": "",
    }

    from unittest.mock import patch

    with patch("hackathon.nebius_nvidia.local_env.load_env_file") as mock_load:
        mock_load.return_value = {
            "NEBIUS_API_KEY": "file-secret",
            "NEBIUS_BASE_URL": "https://file.api",
            "NEBIUS_MODEL": "file-model",
        }

        api_key, base_url, model = get_nebius_config(environ=test_mapping)

        assert api_key is None
        assert base_url == "https://test.api"
        assert model is None
        assert base_url != "https://file.api"
