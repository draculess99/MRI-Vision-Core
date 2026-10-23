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

from hackathon.opencv_aws.change_guard import ChangeGuard
from hackathon.opencv_aws.opencv_adapter import OpenCVProcessor
from mri_core.config import load_execution_config
from mri_core.execution import ExecutionRouter

# Page configuration
st.set_page_config(
    page_title="MRI Vision Core — OpenCV Demo",
    layout="wide",
)

# Initialize session state for baseline image
if "baseline_enhanced" not in st.session_state:
    st.session_state.baseline_enhanced = None

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
    ["Upload an image"],
)

# Load image based on input method
image_array = None
image_source = ""

if input_method == "Upload an image":
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

# Initialize execution router
try:
    config = load_execution_config()
    router = ExecutionRouter(config=config, local_processor=OpenCVProcessor.process)
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()

# Process and display results
if image_array is not None:
    if st.button("Run OpenCV Processing", key="process_btn"):
        with st.spinner("Processing..."):
            try:
                payload = {"source_filename": image_source}

                # Encode image as base64 for router
                success, png_bytes = cv2.imencode(".png", image_array.astype(np.uint8))
                if success:
                    import base64
                    payload["image_data"] = base64.b64encode(png_bytes.tobytes()).decode('utf-8')

                # Execute through router
                exec_result = router.process(payload)

                if not exec_result.success:
                    st.error(f"Processing failed: {exec_result.error}")
                    if exec_result.fallback_used:
                        st.info("Note: Remote execution failed, fell back to local processing.")
                else:
                    artifacts = exec_result.data.get("artifacts", {})
                    manifest = exec_result.data.get("manifest", {})

                    # Show execution mode
                    mode_text = exec_result.mode
                    if exec_result.fallback_used:
                        mode_text += " (fallback)"
                    st.info(f"Execution mode: {mode_text}")

                    # Display in 2x2 grid
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Original")
                        st.image(
                            artifacts.get("original"),
                            width=300,
                            caption="Normalized grayscale",
                        )

                    with col2:
                        st.subheader("Enhanced")
                        st.image(
                            artifacts.get("enhanced"),
                            width=300,
                            caption="CLAHE contrast enhancement",
                        )

                    col3, col4 = st.columns(2)

                    with col3:
                        st.subheader("Detected Edges")
                        st.image(
                            artifacts.get("edges"),
                            width=300,
                            caption="Canny edge detection",
                        )

                    with col4:
                        st.subheader("Overlay")
                        overlay = artifacts.get("overlay")
                        if overlay is not None:
                            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                            st.image(
                                overlay_rgb,
                                width=300,
                                caption="Edge overlay (green)",
                            )

                    st.success("✓ Processing complete")

                    # ChangeGuard: Deterministic baseline comparison
                    st.markdown("---")
                    st.subheader("ChangeGuard Deterministic Comparison")

                    enhanced_img = artifacts.get("enhanced")
                    if enhanced_img is not None:
                        # Compare against baseline
                        comparison = ChangeGuard.compare(
                            st.session_state.baseline_enhanced,
                            enhanced_img
                        )

                        # Display status with color
                        if comparison.status == "STABLE":
                            st.success(f"**{comparison.status}** — Baseline unchanged")
                        elif comparison.status == "REVIEW":
                            st.warning(f"**{comparison.status}** — Baseline changed significantly")
                        else:
                            st.info(f"**{comparison.status}** — {comparison.reason}")

                        # Display deltas if valid comparison was made
                        if comparison.foreground_delta is not None:
                            col_cg1, col_cg2 = st.columns(2)
                            with col_cg1:
                                st.metric(
                                    "Foreground Delta",
                                    f"{comparison.foreground_delta:.1f}%",
                                    delta=f"Threshold: {ChangeGuard.FOREGROUND_DELTA_THRESHOLD}%"
                                )
                            with col_cg2:
                                st.metric(
                                    "Mean Intensity Delta",
                                    f"{comparison.mean_intensity_delta:.0f}",
                                    delta=f"Threshold: {ChangeGuard.MEAN_INTENSITY_DELTA_THRESHOLD}"
                                )

                        # Store as baseline for next run
                        st.session_state.baseline_enhanced = enhanced_img

                        # Safety notice
                        st.caption(
                            "**Deterministic research signal only.** Not for clinical interpretation. "
                            "Compares baseline vs current enhanced image using fixed thresholds. "
                            "Foreground delta: pixels with ≥10 intensity change. "
                            "Review triggered if foreground ≥15% OR mean intensity ≥20 counts."
                        )

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

                    # Show manifest hash if available
                    if manifest:
                        st.markdown("---")
                        st.subheader("Evidence Manifest")
                        hash_val = manifest.get("source_hash_sha256", "N/A")[:16] + "..."
                        st.code(f"SHA-256: {hash_val}", language="text")

            except Exception as e:
                st.error(f"Processing error: {e}")
else:
    st.info("Upload an image to begin processing.")

st.markdown("---")
st.markdown(
    """
    **About**: This demo uses deterministic OpenCV functions for structural analysis.
    No machine learning or clinical interpretation is performed.
    """
)
