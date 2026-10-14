"""Tests for hackathon/opencv_aws/streamlit_app.py's two modes: the pre-existing OpenCV
image-processing demo (must remain unaffected) and the new "Trained MRI Inference —
Research Only" mode built on mri_core.rsna_inference_engine.

Uses Streamlit's headless AppTest, matching the pattern already used by
tests/test_app_rsna_smoke_ui.py for the main app. Real-data cases skip cleanly when the
local RSNA dataset or Phase 5 fold checkpoints are not present.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
torch = pytest.importorskip("torch")

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "hackathon" / "opencv_aws" / "streamlit_app.py"
RSNA_ROOT = ROOT / "data" / "rsna-knee"
RANDOM_FOLD_PATHS = [ROOT / "outputs" / "rsna-fivefold-benchmark" / "random" / f"fold{i}" / "last.pt" for i in range(5)]


def real_data_available():
    return (RSNA_ROOT / "train.csv").is_file() and all(p.is_file() for p in RANDOM_FOLD_PATHS)


def run_app():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    return at


# ------------------------------------------------------------------------------ existing OpenCV mode

class TestExistingOpenCVModeUnaffected:
    def test_opencv_mode_is_the_default_and_loads_without_error(self):
        at = run_app()
        assert not at.exception
        assert at.sidebar.radio(key="demo_mode").value == "OpenCV Image Processing"

    def test_opencv_processing_still_runs_end_to_end(self):
        at = run_app()
        at.sidebar.radio(key="demo_mode").set_value("OpenCV Image Processing")
        at.run()
        at.button(key="process_btn").click().run()  # "Run OpenCV Processing" on the bundled demo image
        assert not at.exception
        assert any("Processing complete" in s.value for s in at.success)

    def test_switching_to_inference_mode_and_back_preserves_opencv_mode(self):
        at = run_app()
        at.sidebar.radio(key="demo_mode").set_value("Trained MRI Inference — Research Only")
        at.run()
        at.sidebar.radio(key="demo_mode").set_value("OpenCV Image Processing")
        at.run()
        assert not at.exception
        assert at.sidebar.radio(key="demo_mode").value == "OpenCV Image Processing"


# ------------------------------------------------------------------------------ trained inference mode: structure

class TestTrainedInferenceModeStructure:
    def test_switching_to_inference_mode_shows_research_warning(self):
        at = run_app()
        at.sidebar.radio(key="demo_mode").set_value("Trained MRI Inference — Research Only")
        at.run()
        assert not at.exception
        assert any("not diagnostic" in w.value for w in at.warning)

    def test_no_local_rsna_data_shows_clear_status_not_a_crash(self, monkeypatch):
        from mri_core.rsna_integration import RSNADiscoveryError

        def always_fail(*args, **kwargs):
            raise RSNADiscoveryError("no local RSNA root for this test")

        monkeypatch.setattr("mri_core.rsna_integration.discover_rsna_root", always_fail)
        at = run_app()
        at.sidebar.radio(key="demo_mode").set_value("Trained MRI Inference — Research Only")
        at.run()
        assert not at.exception
        assert any("not found" in e.value for e in at.error)


@pytest.mark.skipif(not real_data_available(), reason="Local RSNA data or Phase 5 checkpoints not available")
class TestTrainedInferenceModeRealData:
    def _enter_inference_mode(self, at):
        at.sidebar.radio(key="demo_mode").set_value("Trained MRI Inference — Research Only")
        at.run()
        return at

    def test_study_selection_widget_is_populated(self):
        at = self._enter_inference_mode(run_app())
        selectboxes = [s for s in at.sidebar.selectbox if "study" in s.label.lower()]
        assert selectboxes
        assert len(selectboxes[0].options) > 0

    def test_single_model_inference_shows_twelve_targets_in_exact_order(self):
        at = self._enter_inference_mode(run_app())
        at.sidebar.radio(key="inference_mode").set_value("Single model (fold 0)")
        at.run()
        at.sidebar.button(key="run_trained_inference").click().run()
        assert not at.exception

        from mri_core.rsna_knee_dataset import TARGET_COLUMNS
        markdown_targets = [m.value.strip("*") for m in at.markdown if m.value.strip("*") in TARGET_COLUMNS]
        assert markdown_targets == list(TARGET_COLUMNS)

    def test_ensemble_inference_runs_and_reports_five_models(self):
        at = self._enter_inference_mode(run_app())
        at.sidebar.radio(key="inference_mode").set_value("5-fold ensemble")
        at.run()
        at.sidebar.button(key="run_trained_inference").click().run()
        assert not at.exception
        assert any("Models used:** 5" in w.value for w in at.markdown)

    def test_model_provenance_and_disclaimer_are_shown(self):
        at = self._enter_inference_mode(run_app())
        at.sidebar.button(key="run_trained_inference").click().run()
        assert any("not calibrated confidence" in i.value for i in at.info)
        codes = [c.value for c in at.get("code")]
        assert any("sha256:" in c for c in codes)

    def test_review_state_defaults_to_pending_and_can_be_changed(self):
        at = self._enter_inference_mode(run_app())
        at.sidebar.button(key="run_trained_inference").click().run()
        review_radios = [r for r in at.radio if r.label == "Review state"]
        assert review_radios
        assert review_radios[0].value == "Pending review"
        review_radios[0].set_value("Reviewed")
        at.run()
        review_radios = [r for r in at.radio if r.label == "Review state"]
        assert review_radios[0].value == "Reviewed"

    def test_review_state_change_does_not_alter_probabilities(self):
        at = self._enter_inference_mode(run_app())
        at.sidebar.button(key="run_trained_inference").click().run()
        before = [c.value for c in at.caption if "." in c.value and c.value.replace(".", "").isdigit()]
        review_radios = [r for r in at.radio if r.label == "Review state"]
        review_radios[0].set_value("More evidence requested")
        at.run()
        after = [c.value for c in at.caption if "." in c.value and c.value.replace(".", "").isdigit()]
        assert before == after

    def test_missing_checkpoint_is_handled_without_crash(self, monkeypatch, tmp_path):
        # RANDOM_FOLD_PATHS is relative to the launch directory; chdir somewhere with no
        # outputs/ tree makes all five checkpoints genuinely absent, without touching the
        # real protected checkpoint files.
        monkeypatch.chdir(tmp_path)
        at = self._enter_inference_mode(run_app())
        assert not at.exception
        assert any("missing" in e.value.lower() for e in at.sidebar.error)
