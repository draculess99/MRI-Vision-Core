import cv2
import numpy as np
from PIL import Image
import io

def load_image(file_bytes: bytes) -> np.ndarray:
    """
    Loads an image from bytes (PNG, JPG, JPEG) into an OpenCV format.
    Provides useful errors for corrupt/unsupported images.
    """
    try:
        # Use PIL to read image from bytes
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Handle different color modes
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            # RGBA to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            # RGB to BGR for OpenCV
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to load image. Ensure it is a valid PNG/JPG/JPEG. Error: {e}")
