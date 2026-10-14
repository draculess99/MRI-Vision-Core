"""Real-data smoke tests for the inference engine, using the actual Phase 5 random-init
checkpoints and whichever local RSNA studies are fully downloaded. Skips cleanly if the
local RSNA dataset or Phase 5 checkpoints are not present -- no download is triggered.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mri_core.rsna_inference_engine import (
    STATUS_OK,
    sha256_of_file,
    run_ensemble_model_inference,
    run_single_model_inference,
    verify_ensemble_compatibility,
)
from mri_core.rsna_integration import discover_available_studies, load_rsna_metadata_safe
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

pytestmark = pytest.mark.rsna

ROOT = Path(__file__).resolve().parents[1]
RSNA_ROOT = ROOT / "data" / "rsna-knee"
RANDOM_FOLD_PATHS = [ROOT / "outputs" / "rsna-fivefold-benchmark" / "random" / f"fold{i}" / "last.pt" for i in range(5)]


def _require_real_data():
    if not (RSNA_ROOT / "train.csv").is_file():
        pytest.skip("Local RSNA data not available")
    if not all(p.is_file() for p in RANDOM_FOLD_PATHS):
        pytest.skip("Phase 5 random-init fold checkpoints not available")


@pytest.fixture(scope="module")
def real_study():
    _require_real_data()
    metadata = load_rsna_metadata_safe(RSNA_ROOT)
    for study_uid in discover_available_studies(RSNA_ROOT, metadata):
        result = run_single_model_inference(metadata, RANDOM_FOLD_PATHS[0], study_uid)
        if result.status == STATUS_OK:
            return metadata, study_uid
    pytest.skip("No fully-downloaded RSNA study produced a valid result")


class TestRealStudyLoading:
    def test_study_loads_and_produces_twelve_probabilities(self, real_study):
        metadata, study_uid = real_study
        result = run_single_model_inference(metadata, RANDOM_FOLD_PATHS[0], study_uid)
        assert result.status == STATUS_OK
        assert list(result.probabilities.keys()) == list(TARGET_COLUMNS)
        assert len(result.probabilities) == 12
        assert all(0.0 <= v <= 1.0 for v in result.probabilities.values())

    def test_result_is_reproducible_across_calls(self, real_study):
        metadata, study_uid = real_study
        first = run_single_model_inference(metadata, RANDOM_FOLD_PATHS[0], study_uid)
        second = run_single_model_inference(metadata, RANDOM_FOLD_PATHS[0], study_uid)
        assert first.probabilities == second.probabilities

    def test_latency_is_measured(self, real_study):
        metadata, study_uid = real_study
        result = run_single_model_inference(metadata, RANDOM_FOLD_PATHS[0], study_uid)
        assert 0.0 < result.latency_seconds < 60.0


class TestRealEnsembleInference:
    def test_ensemble_compatibility_verified_against_real_checkpoints(self):
        _require_real_data()
        report = verify_ensemble_compatibility(RANDOM_FOLD_PATHS)
        assert report.compatible

    def test_ensemble_produces_valid_aggregated_output(self, real_study):
        metadata, study_uid = real_study
        result = run_ensemble_model_inference(metadata, RANDOM_FOLD_PATHS, study_uid)
        if result.status != STATUS_OK:
            pytest.skip(f"This study is not compatible with the ensemble's fixed config: {result.warnings}")
        assert result.is_ensemble
        assert len(result.model_provenance) == 5
        assert list(result.probabilities.keys()) == list(TARGET_COLUMNS)

    def test_ensemble_aggregation_is_deterministic(self, real_study):
        metadata, study_uid = real_study
        first = run_ensemble_model_inference(metadata, RANDOM_FOLD_PATHS, study_uid)
        if first.status != STATUS_OK:
            pytest.skip("Ensemble not applicable to this study")
        second = run_ensemble_model_inference(metadata, RANDOM_FOLD_PATHS, study_uid)
        assert first.probabilities == second.probabilities


class TestDecisionReportIntegrationContract:
    """Confirms the engine's probabilities dict is accepted as-is by the existing,
    unmodified mri_core.decision.generate_decision_report() -- the integration contract
    any hackathon Streamlit app should rely on instead of touching model internals."""

    def test_engine_probabilities_are_accepted_by_generate_decision_report(self, real_study):
        from mri_core.decision import generate_decision_report
        from mri_core.pipeline import process_mri_image
        from mri_core.rsna_integration import get_available_planes, load_rsna_study_series

        metadata, study_uid = real_study
        plane = get_available_planes(RSNA_ROOT, metadata, study_uid)[0]
        volume, meta = load_rsna_study_series(RSNA_ROOT, metadata, study_uid, plane)
        results = process_mri_image(
            volume.get_display_slice(volume.default_slice_index),
            segmentation_method="otsu", is_mri=True, is_inverted=volume.is_inverted,
        )

        inference_result = run_single_model_inference(metadata, RANDOM_FOLD_PATHS[0], study_uid)
        assert inference_result.status == STATUS_OK

        report = generate_decision_report(
            study_uid=meta["study_uid"], series_uid=meta["series_uid"], plane=meta["plane"],
            volume_data=volume.raw_data, preprocessed_data=results["preprocessed"],
            mask=results["mask"], features=results["features"], num_slices=meta["num_slices"],
            model_predictions=inference_result.probabilities,
        )
        assert report.model_predictions == inference_result.probabilities
        assert report.quality_status is not None  # deterministic quality path is independent and still runs


class TestCheckpointHashesUnchanged:
    def test_running_inference_never_modifies_checkpoint_files(self, real_study):
        _require_real_data()
        before = {p: sha256_of_file(p) for p in RANDOM_FOLD_PATHS}
        metadata, study_uid = real_study
        run_single_model_inference(metadata, RANDOM_FOLD_PATHS[0], study_uid)
        run_ensemble_model_inference(metadata, RANDOM_FOLD_PATHS, study_uid)
        after = {p: sha256_of_file(p) for p in RANDOM_FOLD_PATHS}
        assert before == after
