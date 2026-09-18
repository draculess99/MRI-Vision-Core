from typing import Dict, Any
import numpy as np

from .preprocessing import preprocess_image, resize_preserve_aspect
from .segmentation import segment_image
from .features import extract_features
from .visualization import create_overlay

def process_mri_image(image: np.ndarray, segmentation_method: str = "otsu") -> Dict[str, Any]:
    """
    High level API to process MRI image.
    Returns dictionary with: original, preprocessed, mask, overlay, features.
    """
    preprocessed = preprocess_image(image)
    mask = segment_image(preprocessed, method=segmentation_method)
    features = extract_features(preprocessed, mask)
    
    h, w = preprocessed.shape[:2]
    # Keep aspect ratio same for original visualization to match
    vis_original = resize_preserve_aspect(image, target_width=w)
    overlay = create_overlay(vis_original, mask)
    
    return {
        "original": vis_original,
        "preprocessed": preprocessed,
        "mask": mask,
        "overlay": overlay,
        "features": features
    }
