"""Phase 19B: the prepared (unlaunched) pretrained-weights driver and GPU smoke script are consistent and fail closed."""
import ast
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHA = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
SCRIPTS = [ROOT / "scripts" / "kaggle_phase19_driver.py", ROOT / "scripts" / "kaggle_phase19_smoke.py"]


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_scripts_parse_pin_the_weights_hash_and_require_the_weights_package(path):
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert SHA in source and "phase19_weights" in source
    assert "kaggle competitions submit" not in source and "submit" not in re.findall(r"subprocess\.run\(\[[^\]]*", source)


def test_driver_defaults_to_pretrained_and_only_an_explicit_switch_allows_random_init():
    source = SCRIPTS[0].read_text(encoding="utf-8")
    assert 'os.environ.get("P19_RANDOM_INIT") != "1"' in source
    assert '"pretrained_weights": None if WEIGHTS is None else str(WEIGHTS)' in source


def test_smoke_script_is_training_only_bounded_and_measures_what_was_requested():
    source = SCRIPTS[1].read_text(encoding="utf-8")
    assert "[:8]" in source and "assert not set(smoke_ids) & (set(val_ids) | gold | test)" in source  # <= 8 training studies, no overlap with validation/gold/test
    for needle in ("max_memory_allocated", "isfinite", "save_checkpoint", "load_checkpoint", "OutOfMemoryError", "GradScaler", "scheduler", "loader_generator"):
        assert needle in source, needle


def test_local_weights_package_provenance_is_consistent_when_present():
    prov_path = ROOT / "outputs" / "pretrained" / "phase19_weights" / "PROVENANCE.json"
    if not prov_path.is_file():
        pytest.skip("local weights package not built")
    prov = json.loads(prov_path.read_text())
    assert prov["kind"] == "phase19_weights" and prov["files"]["convnext_tiny-983f1562.pth"]["sha256"] == SHA
    assert prov["source_url"] == "https://download.pytorch.org/models/convnext_tiny-983f1562.pth" and "NOT stated" in prov["license_note"]
