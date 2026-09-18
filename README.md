# MRI Vision Core (Version 0.2)

A clean Python/OpenCV MRI image-processing and volumetric analysis application with a Streamlit UI.
**Disclaimer:** Research and educational prototype. Not for medical diagnosis or clinical decision-making.

## Purpose
Provides a shared computer-vision core for medical image loading (DICOM, NIfTI, PNG, JPG), OpenCV-based preprocessing, volumetric slice navigation, basic segmentation, and feature extraction.

## Architecture
- **mri_core/**: Core computer vision and medical imaging algorithms.
  - `mri_volume.py`: `MRIVolume` abstraction managing volumetric and 2D medical data, slice indexing, and normalized display slices.
  - `loader.py`: Auto-detects format; handles DICOM (`.dcm`), NIfTI (`.nii`, `.nii.gz`), and generic images (PNG/JPG/JPEG). Strictly excludes patient PII.
  - `preprocessing.py`: MRI-aware pathway (finite validation, robust percentile normalization, clipping, CLAHE, conservative denoising) plus generic V0.1 preprocessing.
  - `segmentation.py`: Otsu and Adaptive thresholding with morphological cleanup.
  - `features.py`: Computes basic CV metrics and region measurements.
  - `visualization.py`: Creates mask overlay views.
  - `pipeline.py`: High-level orchestration for image and MRI slice processing.
- **app.py**: Streamlit application with multi-slice slider, metadata expander, and visual comparison columns.
- **tests/**: Pytest suite (18 automated tests) verifying loaders, privacy, slice selection, NaN/Inf handling, and pipelines using synthetic data.

## Installation & Windows Setup
1. Create a virtual environment:
   ```cmd
   python -m venv .venv
   ```
2. Activate it:
   ```cmd
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

## How to Test
Run the test suite:
```cmd
pytest -q
```

## How to Run
Start the Streamlit application:
```cmd
streamlit run app.py
```

## Capabilities

### Version 0.2 (Current)
- **Medical Imaging Formats**: Full support for DICOM (`.dcm`) via `pydicom`, NIfTI (`.nii`, `.nii.gz`) via `nibabel`, and NumPy volumetric arrays (`.npy`, including Stanford MRNet knee examinations).
- **MRNet Dataset Discovery**: Dynamic path resolution for MRNet (`axial`, `coronal`, `sagittal`) without relying on hardcoded directory names, spaces, or apostrophes.
- **DICOM Handling**: Rescale slope/intercept adjustment, photometric interpretation (`MONOCHROME1` inverted contrast vs `MONOCHROME2`), and technical non-identifying metadata extraction.
- **Volumetric Multi-Slice Navigation**: `MRIVolume` abstraction with slice slider defaulting to the middle slice without loading unnecessary volume slices into the CV pipeline. Handles both (Slices, H, W) and (H, W, Slices) orientations.
- **MRI Preprocessing Pathway**: Finite-value validation (`NaN`/`Inf` sanitization), robust percentile intensity normalization, percentile clipping, uint8 conversion, CLAHE contrast enhancement, and conservative denoising.
- **Privacy Enforcement**: Zero patient-identifying fields logged or displayed; `.gitignore` exclusions for `*.dcm`, `*.nii`, `*.nii.gz`, `*.npy`, and dataset directories.
- **21 Automated Tests**: Comprehensive suite covering synthetic DICOM/NIfTI/NumPy loading, real MRNet discovery verification, slice indexing, PII protection, and numerical edge cases.


### Version 0.1 (Validated Core)
- Generic image loading (PNG, JPG, JPEG)
- OpenCV preprocessing, CLAHE, Gaussian denoising
- Otsu/adaptive segmentation & overlay visualization
- Basic image feature extraction & Streamlit UI

Research and educational prototype. Not for medical diagnosis or clinical decision-making.

## Limitations
- Basic thresholding segmentation is NOT medically robust for clinical diagnostics.
- Radiometry is relative: MRI signal intensities are not calibrated physical units (unlike CT Hounsfield Units).
- CPU-based OpenCV processing.

## Future Roadmap
- Region of interest (ROI) interactive detection
- Deep learning segmentation (e.g. UNet / SAM)
- 3D volumetric radiomics features
- Model comparison & REST API