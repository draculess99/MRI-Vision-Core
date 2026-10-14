"""Phase 7B tests: MRI study preview, stale-result invalidation, and the scoped
human-review workflow in hackathon/opencv_aws/streamlit_app.py. Real-data cases skip
cleanly when the local RSNA dataset or Phase 5 fold checkpoints are not present.
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


def enter_inference_mode(at):
    at.sidebar.radio(key="demo_mode").set_value("Trained MRI Inference — Research Only")
    at.run()
    return at


def probability_captions(at):
    return [c.value for c in at.caption if "." in c.value and c.value.replace(".", "", 1).isdigit()]


pytestmark = pytest.mark.skipif(not real_data_available(), reason="Local RSNA data or Phase 5 checkpoints not available")


class TestMriStudyPreview:
    def test_preview_image_is_shown_for_the_selected_study_with_plane_and_slice_caption(self):
        at = enter_inference_mode(run_app())
        images = at.get("image")
        assert images, "Expected at least one preview image"
        captions = [c for img in images for c in img.captions if c]
        assert any("slice" in c.lower() for c in captions)

    def test_changing_plane_updates_the_preview_without_crashing(self):
        at = enter_inference_mode(run_app())
        plane_select = at.selectbox(key="preview_plane")
        if len(plane_select.options) < 2:
            pytest.skip("Only one plane available locally for this study")
        other_plane = next(o for o in plane_select.options if o != plane_select.value)
        plane_select.set_value(other_plane)
        at.run()
        assert not at.exception
        images = at.get("image")
        captions = [c for img in images for c in img.captions if c]
        assert any(other_plane in c for c in captions)

    def test_changing_slice_updates_the_caption(self):
        at = enter_inference_mode(run_app())
        slider = at.slider(key="preview_slice")
        if slider.max == slider.min:
            pytest.skip("Study has only one slice locally")
        new_value = int(slider.max)
        slider.set_value(new_value)
        at.run()
        assert not at.exception
        images = at.get("image")
        captions = [c for img in images for c in img.captions if c]
        assert any(f"slice {new_value + 1}" in c for c in captions)

    def test_no_available_planes_shows_explicit_error_not_blank_preview(self, monkeypatch):
        # The app script re-imports fresh from mri_core.rsna_integration on every AppTest
        # run, so the patch must target that cached module, not the app's own module object.
        monkeypatch.setattr("mri_core.rsna_integration.get_available_planes", lambda *a, **k: [])
        at = enter_inference_mode(run_app())
        assert not at.exception
        assert any("No viewable series" in e.value for e in at.error)


class TestStaleResultInvalidation:
    def test_changing_study_hides_previous_prediction(self):
        at = enter_inference_mode(run_app())
        study_select = next(s for s in at.sidebar.selectbox if "study" in s.label.lower())
        if len(study_select.options) < 2:
            pytest.skip("Only one local study available; cannot test cross-study invalidation")
        first_study, second_study = study_select.options[0], study_select.options[1]

        study_select.set_value(first_study)
        at.run()
        at.sidebar.button(key="run_trained_inference").click().run()
        first_probs = probability_captions(at)
        assert first_probs  # a result rendered for the first study

        # AppTest widget handles go stale after a .run(); re-fetch before the next interaction.
        study_select = next(s for s in at.sidebar.selectbox if "study" in s.label.lower())
        study_select.set_value(second_study)
        at.run()
        # Without clicking "Run" again, the previous study's result must not still be shown.
        assert any("Previous predictions are hidden" in i.value for i in at.info)
        assert probability_captions(at) == []

    def test_changing_inference_mode_hides_previous_prediction(self):
        at = enter_inference_mode(run_app())
        at.sidebar.radio(key="inference_mode").set_value("Single model (fold 0)")
        at.run()
        at.sidebar.button(key="run_trained_inference").click().run()
        assert probability_captions(at)

        at.sidebar.radio(key="inference_mode").set_value("5-fold ensemble")
        at.run()
        assert any("Previous predictions are hidden" in i.value for i in at.info)
        assert probability_captions(at) == []

    def test_checkpoint_hash_change_invalidates_cached_prediction(self, tmp_path):
        at = enter_inference_mode(run_app())
        at.sidebar.radio(key="inference_mode").set_value("Single model (fold 0)")
        at.run()
        at.sidebar.button(key="run_trained_inference").click().run()
        assert probability_captions(at)

        # Mutate the real checkpoint's bytes on disk is forbidden (protected artifact); instead
        # verify the key-building function itself is sensitive to hash changes directly.
        from hackathon.opencv_aws.streamlit_app import build_inference_result_key

        key_a = build_inference_result_key("study1", "Single model (fold 0)", ("hash_a",))
        key_b = build_inference_result_key("study1", "Single model (fold 0)", ("hash_b",))
        assert key_a != key_b

    def test_rerunning_with_identical_selection_still_shows_result(self):
        at = enter_inference_mode(run_app())
        at.sidebar.button(key="run_trained_inference").click().run()
        assert probability_captions(at)
        at.run()  # a harmless rerun with no widget change
        assert probability_captions(at)


class TestReviewWorkflowPreservationAndScoping:
    def test_expanding_provenance_preserves_predictions(self):
        at = enter_inference_mode(run_app())
        at.sidebar.button(key="run_trained_inference").click().run()
        before = probability_captions(at)
        at.run()  # re-render; expanding an expander in real usage does not change any widget value
        assert probability_captions(at) == before

    def test_review_state_transition_preserves_predictions(self):
        at = enter_inference_mode(run_app())
        at.sidebar.button(key="run_trained_inference").click().run()
        before = probability_captions(at)

        review_radio = next(r for r in at.radio if r.label == "Review state")
        review_radio.set_value("More evidence requested")
        at.run()

        after = probability_captions(at)
        assert after == before

    def test_review_state_is_scoped_per_inference_result_not_shared_globally(self):
        at = enter_inference_mode(run_app())
        at.sidebar.radio(key="inference_mode").set_value("Single model (fold 0)")
        at.run()
        at.sidebar.button(key="run_trained_inference").click().run()
        review_radio = next(r for r in at.radio if r.label == "Review state")
        review_radio.set_value("Reviewed")
        at.run()

        at.sidebar.radio(key="inference_mode").set_value("5-fold ensemble")
        at.run()
        at.sidebar.button(key="run_trained_inference").click().run()
        review_radio_new = next(r for r in at.radio if r.label == "Review state")
        # A different (study, mode, checkpoints) combination must start its own review state,
        # not inherit "Reviewed" from the single-model run above.
        assert review_radio_new.value == "Pending review"

    def test_simulated_label_and_no_safety_clearance_language_present(self):
        at = enter_inference_mode(run_app())
        at.sidebar.button(key="run_trained_inference").click().run()
        assert any("Simulated workflow" in c.value for c in at.caption)
        assert any("not a clinical safety clearance" in c.value for c in at.caption)
        assert any("Not assessed" in m.value for m in at.markdown)

    def test_quality_assessment_shows_not_assessed_not_no_problems(self):
        at = enter_inference_mode(run_app())
        at.sidebar.button(key="run_trained_inference").click().run()
        full_text = " ".join(getattr(el, "value", "") for el in at.get("markdown"))
        assert "Not assessed" in full_text
        assert "No quality problems" not in full_text
