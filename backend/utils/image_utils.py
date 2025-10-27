"""
Image processing utilities for resizing and optimization.
"""

from PIL import Image
import logging
import gc

logger = logging.getLogger(__name__)


def resize_image_for_display(image: Image.Image, max_width: int) -> Image.Image:
    """
    Resize image to fit within max_width while:
    1. Never upscaling (only downscale if needed)
    2. Always maintaining aspect ratio
    3. Preserving image quality
    4. Minimizing memory usage
    
    NOTE: This function is kept for compatibility but for large images,
    prefer resizing with OpenCV before converting to PIL.
    
    Args:
        image: PIL Image object (typically with annotations)
        max_width: Maximum width requested by frontend (device-specific)
    
    Returns:
        Resized PIL Image object (or original if already smaller)
    """
    original_width, original_height = image.size
    
    # Never upscale - only downscale if image is larger than max_width
    if original_width <= max_width:
        logger.info(
            f"Image width {original_width}px is already ≤ max_width {max_width}px. "
            f"No resizing needed."
        )
        return image
    
    # Calculate new dimensions maintaining aspect ratio
    scale_factor = max_width / original_width
    new_width = max_width
    new_height = int(original_height * scale_factor)
    
    logger.info(
        f"Resizing image from {original_width}×{original_height} to "
        f"{new_width}×{new_height} (scale factor: {scale_factor:.3f})"
    )
    
    # Use high-quality downsampling filter
    try:
        # Try new PIL API first (Pillow >= 10.0.0)
        try:
            resampling_filter = Image.Resampling.LANCZOS
        except AttributeError:
            # Fall back to old API for older Pillow versions
            resampling_filter = Image.LANCZOS
        
        resized_image = image.resize(
            (new_width, new_height),
            resampling_filter  # High-quality downsampling filter
        )
        logger.info("Image resized successfully using LANCZOS filter")
        
        # Clear original image from memory if it's large
        if original_width > 2000 or original_height > 2000:
            gc.collect()
        
        return resized_image
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        # Return original image if resize fails
        return image


def calculate_optimal_jpeg_quality(image_width: int) -> int:
    """
    Calculate optimal JPEG quality based on image width.
    Lower quality for larger images to reduce memory and bandwidth usage.
    
    Args:
        image_width: Width of the image in pixels
    
    Returns:
        JPEG quality value (1-100)
    """
    if image_width <= 800:
        return 88  # Good quality for small images
    elif image_width <= 1920:
        return 85  # Balanced quality for medium images
    else:
        return 82  # Reasonable quality for large images
