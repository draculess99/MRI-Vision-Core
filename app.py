from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import cv2

from mri_core.loader import load_mri
from mri_core.pipeline import process_mri_image
from mri_core.dataset_discovery import discover_mrnet_root, list_mrnet_exams, load_mrnet_exam
from mri_core.rsna_integration import (
    discover_rsna_root, load_rsna_metadata_safe, discover_available_studies,
    get_available_planes, load_rsna_study_series, RSNADiscoveryError, RSNAStudyNotAvailable
)
from mri_core.decision import generate_decision_report
from mri_core.rsna_knee_inference import run_inference, RSNAInferenceError

# Checkpoint location matches configs/rsna_knee.json's "checkpoint_path". No checkpoint
# is expected to exist yet; the RSNA viewer remains fully usable without one.
RSNA_CHECKPOINT_PATH = Path("outputs/rsna-knee/checkpoint_best.pt")

st.set_page_config(page_title="MRI Vision Core", layout="wide")

st.title("MRI Vision Core")
st.subheader("Medical MRI & OpenCV Image Processing & Feature Exploration")

st.warning("Research and educational prototype. Not for medical diagnosis or clinical decision-making.")

st.sidebar.header("Settings")

# Detect if local RSNA/MRNet datasets are available
mrnet_root = discover_mrnet_root()
try:
    rsna_root = discover_rsna_root()
    has_rsna = True
except RSNADiscoveryError:
    rsna_root = None
    has_rsna = False

# Build data source options
data_sources = ["Upload File"]
if has_rsna:
    data_sources.append("Explore RSNA Studies")
if mrnet_root:
    data_sources.append("Explore MRNet Dataset")

data_source = st.sidebar.radio("Data Source", data_sources, horizontal=False)

volume = None
status_msg = ""
decision_report = None
study_metadata = {}

if data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image / MRI Scan",
        type=["png", "jpg", "jpeg", "dcm", "nii", "gz", "npy"],
        help="Supported formats: PNG, JPG, JPEG, DICOM (.dcm), NIfTI (.nii, .nii.gz), NumPy (.npy)"
    )
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            volume = load_mri(file_bytes, filename=uploaded_file.name)
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
    else:
        status_msg = "Upload an image or MRI file (PNG, JPG, DICOM, NIfTI, NPY) and click 'Process Image' to begin."

elif data_source == "Explore RSNA Studies":
    try:
        metadata = load_rsna_metadata_safe(rsna_root)
        available_studies = discover_available_studies(rsna_root, metadata)

        if not available_studies:
            st.sidebar.warning("No downloaded RSNA studies found. Download in progress?")
        else:
            st.sidebar.caption(f"Found {len(available_studies)} downloaded studies")
            selected_study = st.sidebar.selectbox("Select Study", available_studies)
            available_planes = get_available_planes(rsna_root, metadata, selected_study)

            if available_planes:
                selected_plane = st.sidebar.selectbox("Select Plane", available_planes)

                try:
                    volume, meta = load_rsna_study_series(
                        rsna_root, metadata, selected_study, selected_plane
                    )
                    study_metadata = meta
                    st.sidebar.success(f"Loaded {selected_plane} series")
                except RSNAStudyNotAvailable as e:
                    st.sidebar.warning(str(e))
                except Exception as e:
                    st.sidebar.error(f"Error loading series: {e}")
            else:
                st.sidebar.info(f"No planes available yet for study {selected_study}. Download in progress?")
    except RSNADiscoveryError as e:
        st.sidebar.error(f"RSNA discovery failed: {e}")
    except Exception as e:
        st.sidebar.error(f"Error loading RSNA metadata: {e}")

elif data_source == "Explore MRNet Dataset":
    st.sidebar.caption(f"Discovered MRNet at: `{mrnet_root.name}`")
    plane = st.sidebar.selectbox("Select Imaging Plane", ["axial", "coronal", "sagittal"])
    exams = list_mrnet_exams(plane=plane)
    if exams:
        exam_options = [e.name for e in exams[:200]]
        selected_exam = st.sidebar.selectbox("Select Examination", exam_options, index=0)
        try:
            volume = load_mrnet_exam(selected_exam, plane=plane)
        except Exception as e:
            st.error(f"Error loading exam '{selected_exam}': {e}")
    else:
        st.sidebar.warning(f"No exam files found for plane '{plane}'.")

seg_method = st.sidebar.selectbox("Select Segmentation Method", ["Otsu", "Adaptive"])

if volume is not None:
    try:
        # Display volume details
        st.sidebar.markdown(f"**Format:** `{volume.format_type}`")
        if volume.num_slices > 1:
            st.sidebar.info(f"Multi-slice volume: **{volume.num_slices}** slices")
            slice_idx = st.sidebar.slider(
                "Select Slice",
                min_value=0,
                max_value=volume.num_slices - 1,
                value=volume.default_slice_index,
                help=f"Default middle slice: {volume.default_slice_index}"
            )
            st.sidebar.caption(f"Viewing Slice {slice_idx + 1} of {volume.num_slices} (Index: {slice_idx})")
        else:
            slice_idx = 0
            
        # Non-identifying metadata expander
        if volume.metadata:
            with st.sidebar.expander("Technical Imaging Metadata", expanded=False):
                meta_df = pd.DataFrame(
                    [{"Property": k, "Value": str(v)} for k, v in volume.metadata.items()]
                )
                st.dataframe(meta_df, use_container_width=True, hide_index=True)
                
        # Retrieve display slice for selected index
        display_slice = volume.get_display_slice(slice_idx)
        
        # Action button
        process_btn = st.sidebar.button("Process Image")
        
        if process_btn:
            is_mri = (volume.format_type in ("DICOM", "NIFTI", "NUMPY"))
            results = process_mri_image(
                display_slice,
                segmentation_method=seg_method,
                is_mri=is_mri,
                is_inverted=volume.is_inverted
            )

            col1, col2 = st.columns(2)
            with col1:
                orig = results["original"]
                st.subheader("Original")
                st.image(orig, channels="BGR" if len(orig.shape) == 3 else "GRAY", use_container_width=True)
            with col2:
                st.subheader("Preprocessed")
                st.image(results["preprocessed"], clamp=True, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Segmentation Mask")
                st.image(results["mask"], clamp=True, use_container_width=True)
            with col4:
                overlay_rgb = results["overlay"][..., ::-1] if len(results["overlay"].shape) == 3 else results["overlay"]
                st.subheader("Overlay")
                st.image(overlay_rgb, use_container_width=True)

            st.divider()
            st.subheader("Extracted Image Features")

            features = results["features"]
            formatted_features = {k: round(v, 2) if isinstance(v, float) else v for k, v in features.items()}
            df = pd.DataFrame(list(formatted_features.items()), columns=["Feature", "Value"])
            st.dataframe(df, use_container_width=True)

            # Generate decision report if RSNA study
            if data_source == "Explore RSNA Studies" and study_metadata:
                st.divider()
                st.subheader("Quality Assessment Report")

                model_predictions = None
                if RSNA_CHECKPOINT_PATH.is_file():
                    try:
                        model_predictions = run_inference(
                            metadata=metadata,
                            checkpoint_path=RSNA_CHECKPOINT_PATH,
                            study_uid=study_metadata["study_uid"],
                        )
                    except RSNAInferenceError as e:
                        st.warning(f"Model inference unavailable for this study: {e}")

                try:
                    decision_report = generate_decision_report(
                        study_uid=study_metadata["study_uid"],
                        series_uid=study_metadata["series_uid"],
                        plane=study_metadata["plane"],
                        volume_data=volume.raw_data,
                        preprocessed_data=results["preprocessed"],
                        mask=results["mask"],
                        features=features,
                        num_slices=study_metadata["num_slices"],
                        model_predictions=model_predictions,
                    )

                    # Display report summary
                    col_left, col_right = st.columns([1, 2])
                    with col_left:
                        status_color = {
                            "OK": "🟢",
                            "REVIEW": "🟡",
                            "INVALID": "🔴"
                        }
                        status_text = decision_report.quality_status.value
                        st.metric("Quality Status", f"{status_color[status_text]} {status_text}")

                    with col_right:
                        st.write("**Study Information**")
                        st.write(f"Study UID: `{decision_report.study_uid}`")
                        st.write(f"Plane: {decision_report.plane}")
                        st.write(f"Volume: {decision_report.num_slices} slices, {decision_report.image_height}×{decision_report.image_width}px")

                    # Quality metrics
                    st.write("**Image Quality Metrics**")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Mean Intensity", f"{decision_report.quality_metrics.mean_intensity:.1f}")
                        st.metric("Std Deviation", f"{decision_report.quality_metrics.std_intensity:.1f}")
                    with metrics_col2:
                        st.metric("Intensity Range", f"{decision_report.quality_metrics.intensity_range:.1f}")
                        st.metric("Foreground %", f"{decision_report.quality_metrics.foreground_fraction*100:.1f}%")
                    with metrics_col3:
                        st.metric("Foreground Pixels", f"{decision_report.segmentation_quality.foreground_pixels:,}")
                        st.metric("Contours Found", "Yes" if decision_report.segmentation_quality.has_contours else "No")

                    # Quality flags
                    if decision_report.quality_flags:
                        st.warning(f"**Quality Flags:** {'; '.join(decision_report.quality_flags)}")

                    # Model status
                    st.info(f"**Model Status:** {decision_report.model_status}")

                    if decision_report.model_predictions is not None:
                        st.caption("Model outputs are experimental probabilities and are not a clinical diagnosis.")
                        predictions_df = pd.DataFrame(
                            list(decision_report.model_predictions.items()),
                            columns=["Target", "Probability"],
                        )
                        st.dataframe(predictions_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Error generating decision report: {e}")
        else:
            # Preview before processing
            st.info("File loaded successfully. Click 'Process Image' in the sidebar to run analysis.")
            st.subheader(f"Selected Slice Preview (Slice {slice_idx + 1}/{volume.num_slices})")
            st.image(
                display_slice,
                channels="BGR" if (len(display_slice.shape) == 3 and display_slice.shape[2] == 3) else "GRAY",
                width=450
            )

    except Exception as e:
        st.error(f"Error processing slice: {e}")
else:
    if status_msg:
        st.info(status_msg)
