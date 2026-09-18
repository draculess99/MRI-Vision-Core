import cv2
import numpy as np

def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()

def resize_preserve_aspect(image: np.ndarray, target_width: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    ratio = target_width / w
    target_height = int(h * ratio)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def normalize_intensity(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 0)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Combined preprocessing pipeline.
    Does not modify the original image array.
    """
    gray = to_grayscale(image)
    resized = resize_preserve_aspect(gray)
    normalized = normalize_intensity(resized)
    enhanced = apply_clahe(normalized)
    denoised = denoise(enhanced)
    return denoised
