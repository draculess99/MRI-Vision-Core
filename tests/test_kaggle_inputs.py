"""Kaggle input discovery: standard and nested mounts, missing/duplicate/corrupt inputs packages, no scanning of the DICOM trees."""

import ast
import hashlib
import json
import os
import textwrap
from pathlib import Path

import pytest

from mri_core.kaggle_inputs import InputsError, find_named_files, locate_inputs_package, mount_listing
from scripts import kaggle_rsna_runner as runner

KIND = "phase17a_inputs"


def package(folder, kind=KIND, corrupt=False, drop=None):
    folder.mkdir(parents=True)
    files = {"a.json": b'{"x": 1}', "b.csv": b"c1,c2\n1,2\n"}
    for name, data in files.items():
        (folder / name).write_bytes(data + (b"!" if corrupt and name == "a.json" else b""))
    prov = {"kind": kind, "files": {n: {"sha256": hashlib.sha256(d).hexdigest()} for n, d in files.items()}}
    (folder / "PROVENANCE.json").write_text(json.dumps(prov))
    if drop:
        (folder / drop).unlink()
    return folder


def test_standard_mount_is_found_and_verified(tmp_path):
    package(tmp_path / "rsna-weak-label-phase17a-inputs")
    folder, prov = locate_inputs_package(tmp_path, KIND)
    assert folder.name == "rsna-weak-label-phase17a-inputs" and prov["kind"] == KIND


def test_nested_mount_is_found(tmp_path):
    package(tmp_path / "datasets" / "draculess" / "rsna-weak-label-phase17a-inputs")
    folder, _ = locate_inputs_package(tmp_path, KIND)
    assert folder.parts[-3:] == ("datasets", "draculess", "rsna-weak-label-phase17a-inputs") or folder.name == "rsna-weak-label-phase17a-inputs"


def test_missing_inputs_report_mount_directories_without_contents(tmp_path):
    (tmp_path / "rsna-weak-label-code").mkdir()
    (tmp_path / "rsna-weak-label-code" / "BUNDLE_FILES.json").write_text("{}")
    (tmp_path / "competition").mkdir()
    with pytest.raises(InputsError) as e:
        locate_inputs_package(tmp_path, KIND)
    assert e.value.code == "missing"
    diag = e.value.diagnostics
    assert "rsna-weak-label-code" in diag["mounts"] and diag["mounts"]["rsna-weak-label-code"] == ["BUNDLE_FILES.json"] and "competition" in diag["mounts"]
    assert "{}" not in json.dumps(diag)  # names only, never contents


def test_other_kinds_are_ignored_and_duplicates_of_the_right_kind_are_refused(tmp_path):
    package(tmp_path / "phase16x-inputs", kind="phase16x_inputs")
    package(tmp_path / "one", kind=KIND)
    assert locate_inputs_package(tmp_path, KIND)[0].name == "one"
    package(tmp_path / "datasets" / "o" / "two", kind=KIND)
    with pytest.raises(InputsError) as e:
        locate_inputs_package(tmp_path, KIND)
    assert e.value.code == "duplicate"


def test_incorrect_hash_and_absent_listed_file_are_refused(tmp_path):
    package(tmp_path / "bad", corrupt=True)
    with pytest.raises(InputsError) as e:
        locate_inputs_package(tmp_path, KIND)
    assert e.value.code == "hash_mismatch" and e.value.diagnostics["required_file_presence"] == {"a.json": True, "b.csv": True}
    other = tmp_path / "second"
    package(other / "gone", drop="b.csv")
    with pytest.raises(InputsError) as e2:
        locate_inputs_package(other, KIND)
    assert e2.value.code == "file_missing" and e2.value.diagnostics["required_file_presence"]["b.csv"] is False


def test_override_path_is_used_for_local_runs(tmp_path):
    package(tmp_path / "x")
    assert locate_inputs_package(tmp_path / "unrelated", KIND, override=tmp_path / "x")[0] == tmp_path / "x"


def test_the_dicom_trees_are_never_scanned(tmp_path, monkeypatch):
    comp = tmp_path / "rsna-knee-abnormality-detection"
    for i in range(30):
        (comp / "train_series" / f"study{i}" / "series").mkdir(parents=True)
        (comp / "train_series" / f"study{i}" / "series" / "PROVENANCE.json").write_text("{}")  # trap files inside the pruned tree
    (comp / "train.csv").write_text("x")
    visited = []
    real = os.scandir
    monkeypatch.setattr(os, "scandir", lambda p: (visited.append(str(p)), real(p))[1])
    assert find_named_files(tmp_path, "PROVENANCE.json") == []
    assert not any("train_series" in v for v in visited)
    assert mount_listing(tmp_path) == {"rsna-knee-abnormality-detection": ["train.csv", "train_series"]}  # names only; the tree itself is not entered


def test_discover_data_does_not_walk_the_dicom_trees(tmp_path, monkeypatch):
    comp = tmp_path / "rsna-knee-abnormality-detection"
    for name in runner.REQUIRED_CSVS:
        comp.mkdir(exist_ok=True)
        (comp / name).write_text("x")
    for i in range(20):
        (comp / "train_series" / f"s{i}" / "ser").mkdir(parents=True)
    visited = []
    real = os.walk

    def tracing(top, *a, **k):
        for current, dirs, files in real(top, *a, **k):
            visited.append(current)
            yield current, dirs, files
    monkeypatch.setattr(os, "walk", tracing)
    meta, dicom = runner.discover_data(tmp_path)
    assert meta == comp and dicom == comp and not any("train_series" in v for v in visited)


def test_driver_bundle_finder_is_depth_limited_and_prunes_dicom_trees(tmp_path):
    source = Path("scripts/kaggle_phase17a_driver.py").read_text(encoding="utf-8")
    func = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_find_bundle")
    namespace = {"os": os}
    exec(textwrap.dedent(ast.get_source_segment(source, func)), namespace)
    find = namespace["_find_bundle"]
    (tmp_path / "rsna-weak-label-code").mkdir()
    (tmp_path / "rsna-weak-label-code" / "BUNDLE_FILES.json").write_text("{}")
    (tmp_path / "datasets" / "o" / "code").mkdir(parents=True)
    (tmp_path / "datasets" / "o" / "code" / "BUNDLE_FILES.json").write_text("{}")
    (tmp_path / "comp" / "train_series" / "s").mkdir(parents=True)
    (tmp_path / "comp" / "train_series" / "s" / "BUNDLE_FILES.json").write_text("{}")
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "BUNDLE_FILES.json").write_text("{}")
    found = [Path(p).relative_to(tmp_path).as_posix() for p in find(tmp_path)]
    assert found == ["datasets/o/code/BUNDLE_FILES.json", "rsna-weak-label-code/BUNDLE_FILES.json"]
