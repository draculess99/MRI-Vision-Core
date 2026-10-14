import hashlib
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Add repo root to path so imports work from any directory
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hackathon.opencv_aws.opencv_mri import process_mri_slice
from mri_core.dicom_series import DicomSeriesError
from mri_core.rsna_integration import (
    RSNADiscoveryError,
    RSNAStudyNotAvailable,
    discover_available_studies,
    discover_rsna_root,
    get_available_planes,
    load_rsna_metadata_safe,
    load_rsna_study_series,
)
from mri_core.rsna_inference_engine import (
    STATUS_OK,
    run_ensemble_model_inference,
    run_single_model_inference,
)
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

# Relative to the working directory the app is launched from (repo root), matching the
# existing main app.py's checkpoint-path convention -- this also makes "missing checkpoint"
# testable via monkeypatch.chdir rather than needing to touch real files.
RANDOM_FOLD_PATHS = [
    Path(f"outputs/rsna-fivefold-benchmark/random/fold{i}/last.pt") for i in range(5)
]
REVIEW_STATES = ("Pending review", "Reviewed", "More evidence requested")

# Page configuration
st.set_page_config(
    page_title="MRI Vision Core — OpenCV Demo",
    layout="wide",
)

st.title("MRI Vision Core — OpenCV Demo")

st.warning(
    "**Research and educational demonstration.** "
    "This is a computer-vision image-processing tool for exploring MRI slice structure. "
    "It is not a diagnostic medical tool and does not provide clinical interpretation."
)


@st.cache_data(show_spinner=False)
def cached_single_inference(checkpoint_path_str: str, checkpoint_sha256: str, rsna_root_str: str, study_uid: str):
    """Cached on (checkpoint path, checkpoint hash, data root, study). checkpoint_sha256 is
    not used inside, but is part of the cache key so a changed checkpoint file invalidates
    the cache even if its path stayed the same."""
    del checkpoint_sha256  # part of the cache key only
    metadata = load_rsna_metadata_safe(Path(rsna_root_str))
    return run_single_model_inference(metadata, Path(checkpoint_path_str), study_uid)


@st.cache_data(show_spinner=False)
def cached_ensemble_inference(checkpoint_paths: tuple, checkpoint_hashes: tuple, rsna_root_str: str, study_uid: str):
    del checkpoint_hashes  # part of the cache key only
    metadata = load_rsna_metadata_safe(Path(rsna_root_str))
    return run_ensemble_model_inference(metadata, [Path(p) for p in checkpoint_paths], study_uid)


@st.cache_data(show_spinner=False)
def cached_checkpoint_sha256(checkpoint_path_str: str) -> str:
    from mri_core.rsna_inference_engine import sha256_of_file
    return sha256_of_file(Path(checkpoint_path_str))


def build_inference_result_key(study_uid: str, inference_mode: str, checkpoint_hashes: tuple) -> str:
    """Stable identity for one (study, mode, exact-checkpoint-set) combination. Displayed
    results are only ever shown when this key matches the currently selected widgets --
    changing the study, the mode, or the on-disk checkpoint content (hashes) invalidates
    any previously stored result instead of silently keeping it on screen."""
    payload = f"{study_uid}|{inference_mode}|{'|'.join(checkpoint_hashes)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


demo_mode = st.sidebar.radio(
    "Select demo mode:",
    ["OpenCV Image Processing", "Trained MRI Inference — Research Only"],
    key="demo_mode",
)

# ============================================================================
# MODE 1: Existing OpenCV image-processing demo (unchanged)
# ============================================================================
if demo_mode == "OpenCV Image Processing":
    st.sidebar.header("Input")
    input_method = st.sidebar.radio(
        "Select input method:",
        ["Use bundled demo image", "Upload an image"],
    )

    image_array = None
    image_source = ""

    if input_method == "Use bundled demo image":
        demo_image_path = ROOT / "docs" / "images" / "opencv_aws_demo" / "real_knee_mri_original.png"

        if demo_image_path.exists():
            try:
                demo_image = cv2.imread(str(demo_image_path), cv2.IMREAD_GRAYSCALE)
                if demo_image is not None:
                    image_array = demo_image.astype(np.float32)
                    image_source = f"Demo image: {demo_image_path.name}"
                    st.sidebar.success(f"Loaded: {demo_image_path.name}")
                else:
                    st.sidebar.error("Failed to read demo image")
            except Exception as e:
                st.sidebar.error(f"Error loading demo image: {e}")
        else:
            st.sidebar.warning(f"Demo image not found at {demo_image_path}")

    else:  # Upload an image
        uploaded_file = st.sidebar.file_uploader(
            "Upload a PNG or JPG image:",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file is not None:
            try:
                pil_image = Image.open(uploaded_file)
                if pil_image.mode != "L":
                    pil_image = pil_image.convert("L")
                image_array = np.array(pil_image, dtype=np.float32)
                image_source = f"Uploaded: {uploaded_file.name}"
                st.sidebar.success(f"Loaded: {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error loading uploaded image: {e}")

    st.subheader("OpenCV Image Processing Pipeline")
    st.markdown(
        """
        This pipeline demonstrates deterministic image-processing techniques for MRI slices:

        1. **Original**: Normalized grayscale image
        2. **Enhanced**: Contrast-enhanced using CLAHE
        3. **Detected Edges**: Structural edges via Gaussian blur + Canny
        4. **Overlay**: Edge visualization on enhanced base
        """
    )

    if image_array is not None:
        if st.button("Run OpenCV Processing", key="process_btn"):
            with st.spinner("Processing..."):
                try:
                    result = process_mri_slice(image_array)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original")
                        st.image(result.original, use_container_width=True, caption="Normalized grayscale")
                    with col2:
                        st.subheader("Enhanced")
                        st.image(result.enhanced, use_container_width=True, caption="CLAHE contrast enhancement")

                    col3, col4 = st.columns(2)
                    with col3:
                        st.subheader("Detected Edges")
                        st.image(result.edges, use_container_width=True, caption="Canny edge detection")
                    with col4:
                        st.subheader("Overlay")
                        overlay_rgb = cv2.cvtColor(result.overlay, cv2.COLOR_BGR2RGB)
                        st.image(overlay_rgb, use_container_width=True, caption="Edge overlay (green)")

                    st.success("✓ Processing complete")

                    st.markdown("---")
                    st.subheader("Image Information")
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("Input shape", f"{image_array.shape[0]} × {image_array.shape[1]}")
                    with col_info2:
                        st.metric("Input dtype", str(image_array.dtype))
                    with col_info3:
                        st.metric("Source", image_source.split(":")[0])

                except Exception as e:
                    st.error(f"Processing error: {e}")
    else:
        st.info("Select an input method and provide an image to begin.")

    st.markdown("---")
    st.markdown(
        """
        **About**: This demo uses deterministic OpenCV functions for structural analysis.
        No machine learning or clinical interpretation is performed.
        """
    )

# ============================================================================
# MODE 2: Trained MRI inference — research only
# ============================================================================
else:
    # Scoped CSS for exactly the two study/plane selectboxes below (identified by their
    # own `key=`, which Streamlit renders as a stable `.st-key-<key>` wrapper class --
    # the same mechanism already used for `.st-key-image_grid` in the main app.py).
    # Deliberately does not touch stRadio/stSlider/stButton or any other widget type.
    #
    # Streamlit 1.64's selectbox control is a React Aria ComboBox, not a BaseWeb select
    # (confirmed via Chrome DevTools: the live DOM node is `div.react-aria-ComboBox`, with
    # the visible interactive control as its child `div`). The previous iteration of this
    # CSS targeted `[data-baseweb="select"]`, which does not exist in this DOM and had no
    # visible effect -- those rules are removed below in favor of `.react-aria-ComboBox`.
    st.markdown(
        """
        <style>
        .st-key-study_select .react-aria-ComboBox > div,
        .st-key-preview_plane .react-aria-ComboBox > div {
            min-height: 52px !important;
            background-color: #133C29 !important;
            border: 2px solid #36D67A !important;
            border-radius: 8px !important;
            color: #FFFFFF !important;
            box-sizing: border-box !important;
            display: flex;
            align-items: center;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }
        /* Selected-value text and any nested spans/inputs inside the control. */
        .st-key-study_select .react-aria-ComboBox > div *,
        .st-key-preview_plane .react-aria-ComboBox > div * {
            color: #FFFFFF !important;
        }
        /* Chevron icon -- React Aria Components render it as an inline svg/button inside
        the ComboBox; targeted broadly within the component so it stays visible regardless
        of the exact wrapper markup, without reaching outside .react-aria-ComboBox. */
        .st-key-study_select .react-aria-ComboBox svg,
        .st-key-preview_plane .react-aria-ComboBox svg {
            fill: #36D67A !important;
            stroke: #36D67A !important;
            width: 22px !important;
            height: 22px !important;
        }
        /* Hover: plain CSS :hover plus React Aria's own data-hovered attribute, which the
        library sets on the interactive element itself when hovered via pointer. */
        .st-key-study_select .react-aria-ComboBox > div:hover,
        .st-key-preview_plane .react-aria-ComboBox > div:hover,
        .st-key-study_select .react-aria-ComboBox > div[data-hovered],
        .st-key-preview_plane .react-aria-ComboBox > div[data-hovered] {
            border-color: #36D67A !important;
            box-shadow: 0 0 6px rgba(54, 214, 122, 0.6) !important;
        }
        /* Focus: :focus-within plus React Aria's data-focused/data-focus-visible attributes. */
        .st-key-study_select .react-aria-ComboBox:focus-within > div,
        .st-key-preview_plane .react-aria-ComboBox:focus-within > div,
        .st-key-study_select .react-aria-ComboBox > div[data-focused],
        .st-key-preview_plane .react-aria-ComboBox > div[data-focused],
        .st-key-study_select .react-aria-ComboBox > div[data-focus-visible],
        .st-key-preview_plane .react-aria-ComboBox > div[data-focus-visible] {
            border-color: #36D67A !important;
            box-shadow: 0 0 0 2px rgba(54, 214, 122, 0.55) !important;
        }
        /* The options popover/listbox renders in a portal outside these widgets' own DOM
        subtree, so it cannot be reached by a .st-key-* descendant selector. React Aria
        Components assign fixed default class names to these portal elements
        (react-aria-Popover / react-aria-ListBox / react-aria-ListBoxItem) when no custom
        className is supplied, which is the case here -- these are stable, not
        autogenerated. There are only two selectboxes in this app, so styling every
        instance of these classes stays scoped to selectbox options only (never radios,
        sliders, or buttons, none of which use ComboBox/ListBox at all). The previous
        BaseWeb-era popover rule is kept as a harmless fallback in case any BaseWeb-based
        popover markup is still present in this Streamlit version.
        */
        .react-aria-Popover,
        .react-aria-ListBox {
            background-color: #133C29 !important;
            border: 2px solid #36D67A !important;
            border-radius: 8px !important;
        }
        .react-aria-ListBoxItem,
        .react-aria-Option {
            background-color: #133C29 !important;
            color: #FFFFFF !important;
        }
        .react-aria-ListBoxItem:hover,
        .react-aria-ListBoxItem[data-hovered],
        .react-aria-ListBoxItem[data-focused],
        .react-aria-ListBoxItem[aria-selected="true"],
        .react-aria-Option:hover,
        .react-aria-Option[data-hovered],
        .react-aria-Option[data-focused],
        .react-aria-Option[aria-selected="true"] {
            background-color: #1F5A3E !important;
            color: #FFFFFF !important;
        }
        /* Fallback for any residual BaseWeb-based popover markup in this Streamlit version. */
        ul[data-testid="stSelectboxVirtualDropdown"] {
            background-color: #133C29 !important;
            border: 2px solid #36D67A !important;
            border-radius: 8px !important;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] li {
            background-color: #133C29 !important;
            color: #FFFFFF !important;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] li:hover,
        ul[data-testid="stSelectboxVirtualDropdown"] li[aria-selected="true"] {
            background-color: #1F5A3E !important;
            color: #FFFFFF !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "**Experimental research model — not diagnostic.** Trained on only 46–48 knee MRI "
        "studies per fold (58 fully labeled studies total, 5-fold cross-validated). "
        "Probabilities below are raw, uncalibrated sigmoid outputs, not clinical findings."
    )

    try:
        rsna_root = discover_rsna_root()
        metadata = load_rsna_metadata_safe(rsna_root)
        available_studies = discover_available_studies(rsna_root, metadata)
    except RSNADiscoveryError as e:
        st.error(f"Local RSNA dataset not found: {e}")
        available_studies, rsna_root, metadata = [], None, None

    if rsna_root is not None and not available_studies:
        st.info("No fully downloaded RSNA studies found locally. This mode requires real knee MRI study data.")

    if available_studies:
        st.sidebar.header("Trained Inference")
        selected_study = st.sidebar.selectbox(
            "Select a local knee MRI study", available_studies, key="study_select"
        )

        missing_checkpoints = [p for p in RANDOM_FOLD_PATHS if not p.is_file()]
        if missing_checkpoints:
            st.sidebar.error(
                f"{len(missing_checkpoints)} of 5 fold checkpoint(s) missing under "
                f"outputs/rsna-fivefold-benchmark/random/. Run the Phase 5 benchmark first."
            )

        inference_mode = st.sidebar.radio(
            "Inference mode",
            ["Single model (fold 0)", "5-fold ensemble"],
            key="inference_mode",
        )

        # ---------------------------------------------------------------- MRI study preview
        st.subheader("Selected MRI Study — Preview")
        available_planes = []
        try:
            available_planes = get_available_planes(rsna_root, metadata, selected_study)
        except Exception as e:
            st.error(f"Could not determine available planes for this study: {e}")

        if not available_planes:
            st.error("No viewable series available locally for this study. No preview can be shown.")
        else:
            preview_plane = st.selectbox("Plane", available_planes, key="preview_plane")
            try:
                preview_volume, preview_meta = load_rsna_study_series(rsna_root, metadata, selected_study, preview_plane)
                slice_idx = st.slider(
                    "Slice", min_value=0, max_value=preview_volume.num_slices - 1,
                    value=preview_volume.default_slice_index, key="preview_slice",
                )
                display_slice = preview_volume.get_display_slice(slice_idx)
                st.image(
                    display_slice,
                    channels="BGR" if (display_slice.ndim == 3 and display_slice.shape[2] == 3) else "GRAY",
                    caption=f"{preview_plane} — slice {slice_idx + 1} of {preview_volume.num_slices}",
                    width=360,
                )
            except (RSNAStudyNotAvailable, DicomSeriesError) as e:
                st.error(f"Could not load this study's {preview_plane} series: {e}")
            except Exception as e:
                st.error(f"Unexpected error loading MRI preview: {e}")

        st.markdown("---")

        run_clicked = st.sidebar.button("Run Trained Inference", key="run_trained_inference")

        st.subheader("Research Model Output — Twelve RSNA Knee Targets")

        # ---------------------------------------------------------------- stale-result guard
        # Every displayed result is tagged with the exact (study, mode, checkpoint-hashes) it
        # was computed from. If the current widget selections no longer match that tag -- a
        # different study, a different mode, or a changed checkpoint file -- the stored
        # result is never shown, regardless of what is still sitting in session_state.
        if inference_mode == "Single model (fold 0)":
            current_checkpoint_hashes = (cached_checkpoint_sha256(str(RANDOM_FOLD_PATHS[0])),) if RANDOM_FOLD_PATHS[0].is_file() else ()
        else:
            current_checkpoint_hashes = tuple(
                cached_checkpoint_sha256(str(p)) for p in RANDOM_FOLD_PATHS if p.is_file()
            )
        current_key = build_inference_result_key(selected_study, inference_mode, current_checkpoint_hashes)

        if run_clicked:
            with st.spinner("Running inference..."):
                try:
                    if inference_mode == "Single model (fold 0)":
                        checkpoint = RANDOM_FOLD_PATHS[0]
                        checkpoint_sha256 = cached_checkpoint_sha256(str(checkpoint))
                        result = cached_single_inference(str(checkpoint), checkpoint_sha256, str(rsna_root), selected_study)
                    else:
                        if missing_checkpoints:
                            raise RuntimeError("Cannot run the 5-fold ensemble: one or more fold checkpoints are missing.")
                        checkpoint_strs = tuple(str(p) for p in RANDOM_FOLD_PATHS)
                        checkpoint_hashes = tuple(cached_checkpoint_sha256(str(p)) for p in RANDOM_FOLD_PATHS)
                        result = cached_ensemble_inference(checkpoint_strs, checkpoint_hashes, str(rsna_root), selected_study)
                    st.session_state["inference_result"] = result
                    st.session_state["inference_result_key"] = current_key
                except Exception as e:
                    st.error(f"Inference could not be run: {e}")
                    st.session_state["inference_result"] = None
                    st.session_state["inference_result_key"] = None

        stored_key = st.session_state.get("inference_result_key")
        result = st.session_state.get("inference_result") if stored_key == current_key else None

        if stored_key is not None and stored_key != current_key:
            st.info(
                "Study, inference mode, or checkpoint files have changed since the last run. "
                "Previous predictions are hidden — click **Run Trained Inference** again."
            )

        if result is not None:
            if result.status != STATUS_OK:
                st.warning(
                    f"**Unable to evaluate this study.** {'; '.join(result.warnings) if result.warnings else 'Required input unavailable.'}"
                )
            else:
                # --- 3x4 grid of all twelve targets, in the specified fixed order ---
                st.caption(
                    "Raw model probabilities (sigmoid outputs). **Not calibrated confidence "
                    "scores** — do not interpret as diagnostic likelihood or severity."
                )
                targets_in_order = list(TARGET_COLUMNS)
                rows = [targets_in_order[i:i + 3] for i in range(0, 12, 3)]
                for row_targets in rows:
                    cols = st.columns(3)
                    for col, target in zip(cols, row_targets):
                        probability = result.probabilities[target]
                        with col:
                            st.markdown(f"**{target}**")
                            st.progress(probability)
                            st.caption(f"{probability:.3f}")

                st.markdown("---")

                # --- Research evidence panel ---
                st.subheader("Research Evidence Panel")
                evidence_col1, evidence_col2 = st.columns(2)
                with evidence_col1:
                    st.write(f"**Study identifier (de-identified):** `{result.study_uid[:24]}…`")
                    st.write(f"**Inference mode:** {'5-fold ensemble' if result.is_ensemble else 'Single model'}")
                    st.write(f"**Models used:** {len(result.model_provenance)}")
                    st.write(f"**Engine-flagged input review:** {'Yes' if result.review_required else 'No'}")
                    st.caption(
                        "This flag only means the engine could (or could not) evaluate the "
                        "input at all. A 'No' is not a clinical safety clearance."
                    )
                with evidence_col2:
                    st.write(f"**Inference latency:** {result.latency_seconds:.3f}s")
                    st.write(f"**Data quality warnings:** {'; '.join(result.warnings) if result.warnings else 'None'}")
                    st.write("**Deterministic image-quality assessment:** Not assessed")
                    st.caption(
                        "This app does not run the project's deterministic quality-assessment "
                        "pipeline (mri_core.decision). 'Not assessed' here is literal, never "
                        "an implied 'no quality problems.'"
                    )

                with st.expander("Model checkpoint provenance (SHA-256)"):
                    for provenance in result.model_provenance:
                        st.code(
                            f"fold={provenance.fold}  epoch={provenance.epoch}\n"
                            f"checkpoint: {provenance.checkpoint_path}\n"
                            f"sha256: {provenance.checkpoint_sha256}",
                            language=None,
                        )

                st.info(result.disclaimer)

                st.markdown("---")

                # --- Human-review workflow (demonstration only; no existing ChangeGuard found) ---
                st.subheader("Simulated Human-Review Workflow")
                st.caption(
                    "**Simulated workflow — no clinician has reviewed or approved this study.** "
                    "This control exists only to demonstrate a review-state UI and has no "
                    "effect on, and is not affected by, the model probabilities above."
                )
                # Scoped to the exact inference result (study + mode + checkpoint set), not
                # just the study, so a different mode/checkpoint run starts its own review state.
                review_key = f"review_state__{current_key}"
                if review_key not in st.session_state:
                    st.session_state[review_key] = REVIEW_STATES[0]
                st.radio(
                    "Review state",
                    REVIEW_STATES,
                    key=review_key,
                    help="Changing this has no effect on model predictions above.",
                )
        else:
            st.info("Select a study and click **Run Trained Inference** to see the twelve research-model outputs.")

    st.markdown("---")
    st.markdown(
        """
        **About this mode**: Uses the existing, previously-benchmarked RSNA inference engine
        (`mri_core/rsna_inference_engine.py`) against real local knee MRI DICOM studies. No model
        loading, preprocessing, or checkpoint validation is reimplemented here — all of it is
        reused from tested project code. Arbitrary 2D image uploads are not supported in this
        mode; the trained model requires a complete multi-plane knee MRI study.
        """
    )
