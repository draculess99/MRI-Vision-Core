import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io

from mri_core.loader import load_image
from mri_core.pipeline import process_mri_image

st.set_page_config(page_title="MRI Vision Core", layout="wide")

st.title("MRI Vision Core")
st.subheader("OpenCV MRI Image Processing & Feature Exploration")

st.warning("Research and educational prototype. Not for medical diagnosis or clinical decision-making.")

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload Image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
seg_method = st.sidebar.selectbox("Select Segmentation Method", ["Otsu", "Adaptive"])

if st.sidebar.button("Process Image"):
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            image_array = load_image(file_bytes)
            
            results = process_mri_image(image_array, segmentation_method=seg_method)
            
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
                overlay_rgb = results["overlay"][..., ::-1] # BGR to RGB for st.image
                st.subheader("Overlay")
                st.image(overlay_rgb, use_container_width=True)
                
            st.divider()
            st.subheader("Extracted Image Features")
            
            features = results["features"]
            # Formatting to float nicely
            formatted_features = {k: round(v, 2) if isinstance(v, float) else v for k, v in features.items()}
            
            df = pd.DataFrame(list(formatted_features.items()), columns=["Feature", "Value"])
            st.dataframe(df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.info("Please upload an image first.")
else:
    if uploaded_file is None:
        st.info("Upload an image and click 'Process Image' to begin.")
