"""Tests for hackathon/amd/data_config.py: environment/.env-driven AMD_KAGGLE_RSNA_ROOT /
AMD_KAGGLE_CHECKPOINT resolution and the cheap, structural path validation used before the
Streamlit app ever lists studies or loads a checkpoint.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from hackathon.amd.data_config import (
    EXPECTED_CHECKPOINT_SUFFIX,
    resolve_data_paths,
    validate_checkpoint,
    validate_data_paths,
    validate_data_root,
)

_ENV_VARS = ("AMD_KAGGLE_RSNA_ROOT", "AMD_KAGGLE_CHECKPOINT")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _make_data_root(tmp_path, with_metadata=True):
    root = tmp_path / "rsna-knee"
    root.mkdir()
    if with_metadata:
        (root / "train.csv").write_text("StudyInstanceUID\n", encoding="utf-8")
    return root


def _make_checkpoint(tmp_path, name="fold0-knee-mil-best.ckpt"):
    path = tmp_path / name
    path.write_bytes(b"not a real checkpoint, just bytes for existence/extension checks")
    return path


class TestResolveDataPaths:
    def test_missing_variables_resolve_to_none(self):
        config = resolve_data_paths(use_dotenv=False)
        assert config.data_root is None
        assert config.checkpoint is None

    def test_windows_style_path_is_resolved(self, monkeypatch):
        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", r"D:\Work\Springboard\data\rsna-knee")
        monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", r"D:\Work\Springboard\models\fold0-knee-mil-best.ckpt")
        config = resolve_data_paths(use_dotenv=False)
        assert str(config.data_root).endswith("rsna-knee")
        assert config.checkpoint.suffix == ".ckpt"

    def test_linux_style_path_is_resolved(self, monkeypatch):
        """Illustrative Railway-style paths -- resolution must not assume the file exists,
        only parse the string into a Path; existence is validate_*'s job, not resolve's."""
        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", "/app/data/rsna-knee")
        monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", "/app/models/fold0-knee-mil-best.ckpt")
        config = resolve_data_paths(use_dotenv=False)
        assert config.data_root is not None
        assert config.checkpoint is not None
        assert config.checkpoint.suffix == ".ckpt"
        # Resolution alone never raises or claims existence for an illustrative path.
        assert validate_data_root(config.data_root) == [f"Data directory not found: {config.data_root}"]

    def test_real_environment_wins_over_dotenv_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", "from-real-environment")
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("AMD_KAGGLE_RSNA_ROOT=from-dotenv-file\n", encoding="utf-8")

        config = resolve_data_paths(use_dotenv=True, dotenv_path=dotenv_path)
        assert str(config.data_root) == "from-real-environment"
        assert config.dotenv_loaded is True

    def test_dotenv_file_fills_in_when_real_environment_is_unset(self, tmp_path):
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text(
            "AMD_KAGGLE_RSNA_ROOT=from-dotenv-file\nAMD_KAGGLE_CHECKPOINT=/dotenv/fold0.ckpt\n", encoding="utf-8"
        )
        config = resolve_data_paths(use_dotenv=True, dotenv_path=dotenv_path)
        assert str(config.data_root) == "from-dotenv-file"
        assert config.checkpoint.name == "fold0.ckpt"


class TestValidateDataRoot:
    def test_valid_directory_with_metadata_has_no_problems(self, tmp_path):
        root = _make_data_root(tmp_path)
        assert validate_data_root(root) == []

    def test_missing_directory_is_reported(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        problems = validate_data_root(missing)
        assert len(problems) == 1
        assert "not found" in problems[0]

    def test_directory_without_required_metadata_is_reported(self, tmp_path):
        root = _make_data_root(tmp_path, with_metadata=False)
        problems = validate_data_root(root)
        assert len(problems) == 1
        assert "train.csv" in problems[0]

    def test_never_parses_the_metadata_csv_contents(self, tmp_path, monkeypatch):
        """Validation must be cheap: presence-only, no pandas/CSV parsing."""
        root = _make_data_root(tmp_path)
        import pandas

        def _blocked(*args, **kwargs):
            raise AssertionError("validate_data_root must never parse the CSV, only check its presence.")

        monkeypatch.setattr(pandas, "read_csv", _blocked)
        assert validate_data_root(root) == []


class TestValidateCheckpoint:
    def test_valid_checkpoint_has_no_problems(self, tmp_path):
        checkpoint = _make_checkpoint(tmp_path)
        assert validate_checkpoint(checkpoint) == []

    def test_missing_checkpoint_is_reported(self, tmp_path):
        missing = tmp_path / "does-not-exist.ckpt"
        problems = validate_checkpoint(missing)
        assert len(problems) == 1
        assert "not found" in problems[0]

    def test_wrong_extension_is_reported(self, tmp_path):
        wrong = tmp_path / "checkpoint.pt"
        wrong.write_bytes(b"bytes")
        problems = validate_checkpoint(wrong)
        assert len(problems) == 1
        assert EXPECTED_CHECKPOINT_SUFFIX in problems[0]

    def test_never_loads_the_checkpoint_weights(self, tmp_path, monkeypatch):
        checkpoint = _make_checkpoint(tmp_path)
        import torch

        def _blocked(*args, **kwargs):
            raise AssertionError("validate_checkpoint must never call torch.load.")

        monkeypatch.setattr(torch, "load", _blocked)
        assert validate_checkpoint(checkpoint) == []


class TestValidateDataPaths:
    def test_combines_both_problem_lists(self, tmp_path):
        missing_root = tmp_path / "no-such-dir"
        missing_checkpoint = tmp_path / "no-such.ckpt"
        problems = validate_data_paths(missing_root, missing_checkpoint)
        assert len(problems) == 2

    def test_all_valid_is_empty(self, tmp_path):
        root = _make_data_root(tmp_path)
        checkpoint = _make_checkpoint(tmp_path)
        assert validate_data_paths(root, checkpoint) == []
