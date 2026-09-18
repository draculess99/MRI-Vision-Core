# MRI Vision Core (Version 0.1)

A clean Python/OpenCV MRI image-processing application with a Streamlit UI.
**Disclaimer:** Research and educational prototype. Not for medical diagnosis or clinical decision-making.

## Purpose
Provides a shared computer-vision core for image loading, OpenCV-based preprocessing, basic segmentation, and feature extraction of MRI images.

## Architecture
- **mri_core/**: Core OpenCV computer vision algorithms.
  - `loader.py`: Handles loading PNG/JPG/JPEG into OpenCV formats.
  - `preprocessing.py`: Grayscaling, aspect-preserving resize, intensity normalization, CLAHE, and Gaussian denoising.
  - `segmentation.py`: Otsu and Adaptive thresholding with morphological cleanup.
  - `features.py`: Computes basic CV metrics.
  - `visualization.py`: Creates overlay views.
  - `pipeline.py`: High-level orchestration for the core components.
- **app.py**: Streamlit application UI.
- **tests/**: Pytest suite ensuring components function properly.

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
Run the tests with:
```cmd
pytest -q
```

## How to Run
Start the application:
```cmd
streamlit run app.py
```

## Current Capabilities
- Loading standard image formats (PNG, JPG).
- Image preprocessing (CLAHE, Denoising).
- Threshold-based segmentation.
- Extracted features display.
- Visual overlays.

## Limitations
- Version 0.1 only supports simple image formats (PNG/JPG).
- Basic segmentation is NOT medically robust.
- Requires CPU inference, no GPU optimizations included yet.

## Future Roadmap
Future versions may add:
- DICOM/NIfTI loadings
- ROI detection
- advanced segmentation
- radiomics/features
- ML/DL inference
- model comparison
- REST API
- cloud deployment
