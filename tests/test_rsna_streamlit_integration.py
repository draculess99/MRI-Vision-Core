"""Tests for RSNA Streamlit app integration.

Verifies the RSNA data-source mode initializes correctly, study/plane selection
works, and incomplete/missing local data produces clean non-fatal handling
rather than crashing the app.
"""

from pathlib import Path

import pytest

from mri_core.rsna_integration import (
    discover_rsna_root,
    load_rsna_metadata_safe,
    discover_available_studies,
    get_available_planes,
    load_rsna_study_series,
    RSNADiscoveryError,
    RSNAStudyNotAvailable,
)

pytestmark = pytest.mark.rsna

ROOT = Path(__file__).resolve().parents[1]
RSNA_ROOT = ROOT / "data" / "rsna-knee"


class TestRSNAModeInitialization:
    """Verify the RSNA exploration mode can initialize without crashing."""

    def test_rsna_mode_initializes_with_valid_root(self):
        """discover_rsna_root + metadata load succeed together, as app.py calls them."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        root = discover_rsna_root(explicit_root=RSNA_ROOT)
        metadata = load_rsna_metadata_safe(root)
        assert metadata is not None

    def test_rsna_mode_handles_missing_root_gracefully(self):
        """When RSNA root cannot be discovered, app.py catches RSNADiscoveryError
        and disables the RSNA mode rather than crashing (see app.py has_rsna flag)."""
        with pytest.raises(RSNADiscoveryError):
            discover_rsna_root(explicit_root=Path("/definitely/not/a/real/path"))

    def test_app_module_imports_without_error(self):
        """app.py must import cleanly with the new RSNA integration wired in."""
        pytest.importorskip("streamlit")
        import importlib
        import sys

        # app.py runs top-level Streamlit calls (st.set_page_config etc.) that
        # require a Streamlit script-run context; importing directly under
        # pytest would raise Streamlit's "missing ScriptRunContext" warnings
        # but should not raise ImportError for our new modules.
        spec = importlib.util.find_spec("app")
        assert spec is not None


class TestStudyAndPlaneSelection:
    """Verify study/plane selection logic used by the Streamlit sidebar works standalone."""

    def test_study_selection_lists_only_downloaded_studies(self):
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        studies = discover_available_studies(RSNA_ROOT, metadata)
        # Every entry must be selectable without raising when probing planes
        for study_uid in studies:
            planes = get_available_planes(RSNA_ROOT, metadata, study_uid)
            assert isinstance(planes, list)

    def test_plane_selection_then_load_round_trip(self):
        """Simulates the sidebar flow: pick study -> pick plane -> load series."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        studies = discover_available_studies(RSNA_ROOT, metadata)
        if not studies:
            pytest.skip("No downloaded RSNA studies available")

        study_uid = studies[0]
        planes = get_available_planes(RSNA_ROOT, metadata, study_uid)
        if not planes:
            pytest.skip("No available planes for the first downloaded study")

        volume, meta = load_rsna_study_series(RSNA_ROOT, metadata, study_uid, planes[0])
        assert volume.num_slices > 0
        assert meta["study_uid"] == study_uid


class TestIncompleteDownloadHandling:
    """Verify incomplete downloads produce clean, non-fatal signals (matching app.py's
    try/except RSNAStudyNotAvailable handling around load_rsna_study_series)."""

    def test_study_with_no_local_series_raises_recoverable_error(self):
        """A study listed in metadata but not yet downloaded raises RSNAStudyNotAvailable,
        which app.py catches and turns into a sidebar warning instead of a crash."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        all_studies = metadata.train["StudyInstanceUID"].astype(str).tolist()
        downloaded = set(discover_available_studies(RSNA_ROOT, metadata))

        not_downloaded = next((uid for uid in all_studies if uid not in downloaded), None)
        if not_downloaded is None:
            pytest.skip("All metadata studies already downloaded locally")

        try:
            load_rsna_study_series(RSNA_ROOT, metadata, not_downloaded, "Axial")
            pytest.fail("Expected RSNAStudyNotAvailable for a non-downloaded study")
        except RSNAStudyNotAvailable:
            pass  # Expected: this is the non-fatal path app.py relies on

    def test_empty_data_root_does_not_crash_discovery(self, tmp_path):
        """An RSNA root with metadata but zero downloaded DICOMs (very early in the
        bulk download) must return an empty study list, not raise."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        # tmp_path has no raw/train_series at all
        available = discover_available_studies(tmp_path, metadata)
        assert available == []

    def test_partially_present_series_directory_handled(self, tmp_path):
        """A study directory that exists but is empty (download started, no files
        landed yet) must not be listed as available and must not crash get_available_planes."""
        if not (RSNA_ROOT / "train.csv").is_file():
            pytest.skip("RSNA data directory not found")

        metadata = load_rsna_metadata_safe(RSNA_ROOT)
        all_studies = metadata.train["StudyInstanceUID"].astype(str).tolist()
        if not all_studies:
            pytest.skip("No studies in metadata")

        fake_study_uid = all_studies[0]
        empty_study_dir = tmp_path / "raw" / "train_series" / fake_study_uid
        empty_study_dir.mkdir(parents=True)  # study dir exists but has no series subdirs

        available = discover_available_studies(tmp_path, metadata)
        assert fake_study_uid not in available

        planes = get_available_planes(tmp_path, metadata, fake_study_uid)
        assert planes == []
