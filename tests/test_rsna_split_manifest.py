"""Smoke tests for the reproducible RSNA CV split manifest, and a non-regression check
that generating it never touches the MRNet dataset/split machinery.
"""

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

from mri_core.rsna_knee_dataset import TARGET_COLUMNS, load_rsna_metadata
from mri_core.rsna_split_manifest import (
    SplitIntegrityError,
    build_rsna_cv_manifest,
    library_versions,
    verify_cv_manifest_integrity,
    write_cv_manifest,
)

from .test_rsna_knee import make_metadata


@pytest.fixture
def metadata(tmp_path):
    root, _, _ = make_metadata(tmp_path, labeled_rows=10)
    return load_rsna_metadata(root)


class TestManifestGeneration:
    def test_manifest_has_one_entry_per_fold(self, metadata):
        manifest = build_rsna_cv_manifest(metadata, n_splits=5, seed=42)
        assert len(manifest["folds"]) == 5
        assert manifest["n_splits"] == 5
        assert manifest["seed"] == 42
        assert manifest["target_columns"] == list(TARGET_COLUMNS)

    def test_manifest_records_study_ids_and_label_distributions(self, metadata):
        manifest = build_rsna_cv_manifest(metadata, n_splits=5, seed=42)
        for fold in manifest["folds"]:
            assert fold["train_study_ids"] == sorted(fold["train_study_ids"])
            assert fold["valid_study_ids"] == sorted(fold["valid_study_ids"])
            assert set(fold["train_label_counts"]) == set(TARGET_COLUMNS)
            assert set(fold["valid_label_counts"]) == set(TARGET_COLUMNS)
            assert fold["train_size"] == len(fold["train_study_ids"])
            assert fold["valid_size"] == len(fold["valid_study_ids"])

    def test_manifest_records_library_versions(self, metadata):
        manifest = build_rsna_cv_manifest(metadata, n_splits=5, seed=42)
        versions = manifest["library_versions"]
        assert "torch" in versions and "numpy" in versions
        assert "scikit_learn" in versions and "iterative_stratification" in versions

    def test_same_seed_reproduces_identical_manifest(self, metadata):
        first = build_rsna_cv_manifest(metadata, n_splits=5, seed=42)
        second = build_rsna_cv_manifest(metadata, n_splits=5, seed=42)
        for f1, f2 in zip(first["folds"], second["folds"]):
            assert f1["train_study_ids"] == f2["train_study_ids"]
            assert f1["valid_study_ids"] == f2["valid_study_ids"]

    def test_different_seed_can_produce_a_different_manifest(self, metadata):
        first = build_rsna_cv_manifest(metadata, n_splits=5, seed=42)
        second = build_rsna_cv_manifest(metadata, n_splits=5, seed=1)
        all_first = [f["valid_study_ids"] for f in first["folds"]]
        all_second = [f["valid_study_ids"] for f in second["folds"]]
        assert all_first != all_second

    def test_benchmark_reproducibility_note_does_not_claim_auc_verified(self, metadata):
        manifest = build_rsna_cv_manifest(metadata, n_splits=5, seed=42)
        note = manifest["benchmark_reproducibility_note"].lower()
        assert "does not" in note or "not independently" in note


class TestIntegrityVerification:
    def test_valid_manifest_passes_verification(self, metadata):
        manifest = build_rsna_cv_manifest(metadata, n_splits=5, seed=42)
        verify_cv_manifest_integrity(manifest)  # must not raise

    def test_detects_train_valid_overlap_within_a_fold(self, metadata):
        manifest = build_rsna_cv_manifest(metadata, n_splits=5, seed=42)
        leaked_uid = manifest["folds"][0]["valid_study_ids"][0]
        manifest["folds"][0]["train_study_ids"].append(leaked_uid)
        manifest["folds"][0]["train_size"] += 1
        with pytest.raises(SplitIntegrityError):
            verify_cv_manifest_integrity(manifest)

    def test_detects_cross_fold_validation_overlap(self, metadata):
        manifest = build_rsna_cv_manifest(metadata, n_splits=5, seed=42)
        duplicate_uid = manifest["folds"][0]["valid_study_ids"][0]
        manifest["folds"][1]["valid_study_ids"].append(duplicate_uid)
        with pytest.raises(SplitIntegrityError):
            verify_cv_manifest_integrity(manifest)

    def test_detects_incomplete_coverage_of_labeled_studies(self, metadata):
        manifest = build_rsna_cv_manifest(metadata, n_splits=5, seed=42)
        manifest["folds"][0]["valid_study_ids"] = manifest["folds"][0]["valid_study_ids"][:-1]
        manifest["folds"][0]["valid_size"] -= 1
        with pytest.raises(SplitIntegrityError):
            verify_cv_manifest_integrity(manifest)


class TestWriteManifest:
    def test_write_is_atomic_and_readable_back(self, tmp_path, metadata):
        import json

        manifest = build_rsna_cv_manifest(metadata, n_splits=5, seed=42)
        path = tmp_path / "cv_manifest.json"
        write_cv_manifest(manifest, path)
        assert path.is_file()
        assert not path.with_suffix(path.suffix + ".tmp").exists()
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        assert reloaded["seed"] == 42
        assert len(reloaded["folds"]) == 5


class TestMrnetSplitIsUntouched:
    """Non-regression: building an RSNA CV manifest must not import or alter anything
    in the MRNet dataset/manifest/label modules, and the original MRNet train/valid
    label files on disk are never read by this module."""

    def test_rsna_split_manifest_module_does_not_import_mrnet_modules(self):
        import mri_core.rsna_split_manifest as module

        source = module.__file__
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        assert "cnn_dataset" not in text
        assert "manifest.py" not in text
        assert "from .manifest" not in text
        assert "from .labels" not in text
