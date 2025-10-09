"""
Image processing utilities service.
"""

import cv2
import numpy as np
import base64
import tempfile
from typing import Optional


class ImageService:
    """Service for image processing operations."""
    
    @staticmethod
    def base64_to_array(base64_string: str) -> Optional[np.ndarray]:
        """
        Convert base64 string to OpenCV image array.
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            NumPy array or None on failure
        """
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"❌ Base64 to image conversion failed: {e}")
            return None
    
    @staticmethod
    def array_to_base64(image_array: np.ndarray) -> Optional[str]:
        """
        Convert OpenCV image array to base64 string.
        
        Args:
            image_array: NumPy image array
            
        Returns:
            Base64 encoded string or None on failure
        """
        try:
            _, buffer = cv2.imencode('.jpg', image_array)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
        except Exception as e:
            print(f"❌ Image to base64 conversion failed: {e}")
            return None
    
    @staticmethod
    def save_temp_image(image: np.ndarray) -> Optional[str]:
        """
        Save image to temporary file.
        
        Args:
            image: NumPy image array
            
        Returns:
            Temporary file path or None on failure
        """
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_file.name, image)
            return temp_file.name
        except Exception as e:
            print(f"❌ Failed to save temp image: {e}")
            return None
    
    @staticmethod
    def validate_image_data(image_data: bytes, max_size_mb: int = 10) -> bool:
        """
        Validate image data.
        
        Args:
            image_data: Raw image bytes
            max_size_mb: Maximum allowed size in MB
            
        Returns:
            True if valid, False otherwise
        """
        # Check size
        size_mb = len(image_data) / (1024 * 1024)
        if size_mb > max_size_mb:
            print(f"❌ Image too large: {size_mb:.2f}MB > {max_size_mb}MB")
            return False
        
        # Try to decode
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            print("❌ Invalid image data")
            return False
        
        return True


# Create global image service instance
image_service = ImageService()
