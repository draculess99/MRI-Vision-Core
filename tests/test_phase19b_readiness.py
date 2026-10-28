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
def test_scripts_parse_and_pin_the_weights_hash(path):
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert SHA in source
    assert "kaggle competitions submit" not in source and "submit" not in re.findall(r"subprocess\.run\(\[[^\]]*", source)


def test_driver_defaults_to_the_verified_official_download_and_random_init_needs_an_explicit_full_mode_switch():
    source = SCRIPTS[0].read_text(encoding="utf-8")
    assert 'os.environ.get("P19_RANDOM_INIT") == "1" and MODE == "full"' in source
    assert "download_verified(WEIGHTS_URL, WEIGHTS, EXPECTED_WEIGHTS_SHA256" in source and "no random-initialisation fallback" in source and "phase19_weights" not in source
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


def test_smoke_downloads_only_the_official_https_url_with_the_full_hash_and_never_falls_back_to_random_init():
    source = SCRIPTS[1].read_text(encoding="utf-8")
    assert 'WEIGHTS_URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"' in source
    assert "download_verified(WEIGHTS_URL, WEIGHTS, EXPECTED_WEIGHTS_SHA256" in source and "no random-initialisation fallback" in source
    assert "phase19_weights" not in source and "pretrained_weights\": None}" not in source.split("# ---- model")[1].split("fresh =")[0]


def test_smoke_kernel_metadata_is_private_gpu_internet_on_with_no_weights_dataset():
    meta = ROOT / "outputs" / "kaggle_smoke" / "p19b_smoke_DO_NOT_PUSH_without_approval" / "kernel-metadata.json"
    if not meta.is_file():
        pytest.skip("local kernel folder not built")
    m = json.loads(meta.read_text())
    assert m["is_private"] and m["enable_gpu"] and m["enable_internet"] is True
    assert m["dataset_sources"] == ["draculess/rsna-weak-label-code", "draculess/rsna-weak-label-phase17a-inputs"] and m["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert (meta.parent / "kaggle_phase19_smoke.py").read_text(encoding="utf-8") == SCRIPTS[1].read_text(encoding="utf-8")


class _Resp:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        chunk, self.pos = self.data[self.pos:self.pos + n], self.pos + n
        return chunk


def test_download_verified_accepts_only_matching_bytes_over_https_and_fails_closed(tmp_path):
    import hashlib
    from mri_core.kaggle_inputs import InputsError, download_verified
    data = b"x" * 3_000_000
    good = hashlib.sha256(data).hexdigest()
    dest = tmp_path / "w.pth"
    assert download_verified("https://example.org/w.pth", dest, good, opener=lambda u: _Resp(data))["downloaded"] and dest.read_bytes() == data
    assert download_verified("https://example.org/w.pth", dest, good, opener=lambda u: pytest.fail("must not re-download a verified file"))["downloaded"] is False
    for url, op, code, exp in [("http://example.org/w.pth", lambda u: _Resp(data), "insecure_url", good), ("https://example.org/w.pth", lambda u: _Resp(data + b"!"), "hash_mismatch", good),
                               ("https://example.org/w.pth", lambda u: _Resp(data), "too_large", good)]:
        bad = tmp_path / code
        with pytest.raises(InputsError) as e:
            download_verified(url, bad, exp, opener=op, **({"max_bytes": 1000} if code == "too_large" else {}))
        assert e.value.code == code and not bad.exists() and not list(tmp_path.glob(f"{code}*"))
    def boom(u):
        raise OSError("network down")
    with pytest.raises(InputsError) as e:
        download_verified("https://example.org/w.pth", tmp_path / "n.pth", good, opener=boom)
    assert e.value.code == "download_failed" and not (tmp_path / "n.pth").exists()
    dest.write_bytes(b"corrupt")  # a stale or corrupt file is never trusted
    with pytest.raises(InputsError):
        download_verified("https://example.org/w.pth", dest, good, opener=lambda u: _Resp(b"other"))
    assert not dest.exists()


def test_stability_kernel_is_the_driver_in_stability_mode_and_is_private_gpu_with_no_extra_datasets():
    d = ROOT / "outputs" / "kaggle_smoke" / "p19c_stability_DO_NOT_PUSH_without_approval"
    if not d.is_dir():
        pytest.skip("local kernel folder not built")
    body = (d / "kaggle_phase19c_stability.py").read_text(encoding="utf-8")
    assert body.startswith("import os as _os") and 'if MODE == "stability":' in body  # a record of the launched rehearsal; the driver has since gained full-run resume wiring and '"P19_MODE"] = "stability"' in body.split("\n")[1]
    m = json.loads((d / "kernel-metadata.json").read_text())
    assert m["is_private"] and m["enable_gpu"] and m["enable_internet"] is True and m["id"] == "draculess/rsna-phase19c-stability"
    assert m["dataset_sources"] == ["draculess/rsna-weak-label-code", "draculess/rsna-weak-label-phase17a-inputs"] and m["competition_sources"] == ["rsna-knee-abnormality-detection"]


def test_driver_stability_mode_is_training_only_unscored_and_full_mode_keeps_the_registered_partition():
    source = SCRIPTS[0].read_text(encoding="utf-8")
    branch = source.split('if MODE == "stability":')[1].split('SUMMARY["stage"] = "cache"')[0]
    assert "set(val_ids)" in branch and "complete_label_train" in branch and "test_ids" in branch and "score-validation" not in branch and "metadata" not in branch.replace("metadata.", "")
    assert 'VAL_FRACTION = 0.2 if MODE == "full" else 0.4' in source and '"grad_accum_steps": ACCUM' in source and '"amp_init_scale": 1024.0' in source
    assert 'os.environ.get("P19_ACCUM", "2")' in source and 'int(os.environ.get("P19_BATCH", "2"))' in source
