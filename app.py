import streamlit as st
import numpy as np
import pandas as pd
import cv2

from mri_core.loader import load_mri
from mri_core.pipeline import process_mri_image

st.set_page_config(page_title="MRI Vision Core", layout="wide")

st.title("MRI Vision Core")
st.subheader("Medical MRI & OpenCV Image Processing & Feature Exploration")

st.warning("Research and educational prototype. Not for medical diagnosis or clinical decision-making.")

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload Image / MRI Scan",
    type=["png", "jpg", "jpeg", "dcm", "nii", "gz"],
    help="Supported formats: PNG, JPG, JPEG, DICOM (.dcm), NIfTI (.nii, .nii.gz)"
)

seg_method = st.sidebar.selectbox("Select Segmentation Method", ["Otsu", "Adaptive"])

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        volume = load_mri(file_bytes, filename=uploaded_file.name)
        
        # Display volume details
        st.sidebar.markdown(f"**Format:** `{volume.format_type}`")
        if volume.num_slices > 1:
            st.sidebar.info(f"Multi-slice volume: **{volume.num_slices}** slices")
            slice_idx = st.sidebar.slider(
                "Select Slice",
                min_value=0,
                max_value=volume.num_slices - 1,
                value=volume.default_slice_index,
                help=f"Middle slice: {volume.default_slice_index}"
            )
            st.sidebar.caption(f"Viewing Slice {slice_idx + 1} of {volume.num_slices}")
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
        
        # If user clicked process or has a processed state
        if process_btn:
            is_mri = (volume.format_type in ("DICOM", "NIFTI"))
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
        st.error(f"Error loading or processing file: {e}")
else:
    st.info("Upload an image or MRI file (PNG, JPG, DICOM, NIfTI) and click 'Process Image' to begin.")
