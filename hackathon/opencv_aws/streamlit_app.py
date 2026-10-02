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

# Sidebar for input method selection
st.sidebar.header("Input")
input_method = st.sidebar.radio(
    "Select input method:",
    ["Use bundled demo image", "Upload an image"],
)

# Load image based on input method
image_array = None
image_source = ""

if input_method == "Use bundled demo image":
    demo_image_path = Path(__file__).resolve().parents[2] / "docs" / "images" / "opencv_aws_demo" / "real_knee_mri_original.png"

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

            # Convert to grayscale if needed
            if pil_image.mode != "L":
                pil_image = pil_image.convert("L")

            image_array = np.array(pil_image, dtype=np.float32)
            image_source = f"Uploaded: {uploaded_file.name}"
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error loading uploaded image: {e}")

# Main content
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

# Process and display results
if image_array is not None:
    if st.button("Run OpenCV Processing", key="process_btn"):
        with st.spinner("Processing..."):
            try:
                result = process_mri_slice(image_array)

                # Display in 2x2 grid
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original")
                    st.image(
                        result.original,
                        use_container_width=True,
                        caption="Normalized grayscale",
                    )

                with col2:
                    st.subheader("Enhanced")
                    st.image(
                        result.enhanced,
                        use_container_width=True,
                        caption="CLAHE contrast enhancement",
                    )

                col3, col4 = st.columns(2)

                with col3:
                    st.subheader("Detected Edges")
                    st.image(
                        result.edges,
                        use_container_width=True,
                        caption="Canny edge detection",
                    )

                with col4:
                    st.subheader("Overlay")
                    # Convert BGR to RGB for proper Streamlit display
                    overlay_rgb = cv2.cvtColor(result.overlay, cv2.COLOR_BGR2RGB)
                    st.image(
                        overlay_rgb,
                        use_container_width=True,
                        caption="Edge overlay (green)",
                    )

                st.success("✓ Processing complete")

                # Show image info
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

# Footer
st.markdown("---")
st.markdown(
    """
    **About**: This demo uses deterministic OpenCV functions for structural analysis.
    No machine learning or clinical interpretation is performed.
    """
)
