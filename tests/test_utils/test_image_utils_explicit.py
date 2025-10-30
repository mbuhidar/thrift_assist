"""Explicit tests for image utility functions with various edge cases."""

import pytest
from PIL import Image
import numpy as np


@pytest.mark.unit
class TestImageUtilsExplicit:
    """Explicit test cases for image utilities."""
    
    def test_calculate_optimal_jpeg_quality_small_image(self):
        """Test JPEG quality for small images (≤800px)."""
        from backend.utils.image_utils import calculate_optimal_jpeg_quality
        
        assert calculate_optimal_jpeg_quality(400) == 88
        assert calculate_optimal_jpeg_quality(800) == 88
        assert calculate_optimal_jpeg_quality(1) == 88
    
    def test_calculate_optimal_jpeg_quality_medium_image(self):
        """Test JPEG quality for medium images (801-1920px)."""
        from backend.utils.image_utils import calculate_optimal_jpeg_quality
        
        assert calculate_optimal_jpeg_quality(801) == 85
        assert calculate_optimal_jpeg_quality(1920) == 85
        assert calculate_optimal_jpeg_quality(1000) == 85
    
    def test_calculate_optimal_jpeg_quality_large_image(self):
        """Test JPEG quality for large images (>1920px)."""
        from backend.utils.image_utils import calculate_optimal_jpeg_quality
        
        assert calculate_optimal_jpeg_quality(1921) == 82
        assert calculate_optimal_jpeg_quality(4000) == 82
        assert calculate_optimal_jpeg_quality(10000) == 82
    
    def test_calculate_optimal_jpeg_quality_boundary_values(self):
        """Test JPEG quality at exact boundary values."""
        from backend.utils.image_utils import calculate_optimal_jpeg_quality
        
        # Boundaries
        assert calculate_optimal_jpeg_quality(800) == 88
        assert calculate_optimal_jpeg_quality(801) == 85
        assert calculate_optimal_jpeg_quality(1920) == 85
        assert calculate_optimal_jpeg_quality(1921) == 82
    
    def test_calculate_optimal_jpeg_quality_zero_and_negative(self):
        """Test JPEG quality with zero and negative values."""
        from backend.utils.image_utils import calculate_optimal_jpeg_quality
        
        # Edge cases - should handle gracefully
        assert calculate_optimal_jpeg_quality(0) == 88
        assert calculate_optimal_jpeg_quality(-1) == 88  # Treat as small
    
    def test_resize_image_no_resize_needed_exact_size(self):
        """Test resize when image is exactly max_width."""
        from backend.utils.image_utils import resize_image_for_display
        
        # Create image exactly at max_width
        img = Image.new('RGB', (1920, 1080))
        result = resize_image_for_display(img, 1920)
        
        # Should return same image, no resize
        assert result.size == (1920, 1080)
    
    def test_resize_image_no_resize_needed_smaller(self):
        """Test resize when image is smaller than max_width."""
        from backend.utils.image_utils import resize_image_for_display
        
        # Create small image
        img = Image.new('RGB', (800, 600))
        result = resize_image_for_display(img, 1920)
        
        # Should NOT upscale
        assert result.size == (800, 600)
    
    def test_resize_image_resize_needed(self):
        """Test resize when image exceeds max_width."""
        from backend.utils.image_utils import resize_image_for_display
        
        # Create large image
        img = Image.new('RGB', (3840, 2160))
        result = resize_image_for_display(img, 1920)
        
        # Should downscale to max_width
        assert result.size[0] == 1920
        # Check aspect ratio preserved
        expected_height = int(2160 * (1920 / 3840))
        assert result.size[1] == expected_height
    
    def test_resize_image_aspect_ratio_preserved_wide(self):
        """Test that aspect ratio is preserved for wide images."""
        from backend.utils.image_utils import resize_image_for_display
        
        # Wide image (16:9)
        img = Image.new('RGB', (2560, 1440))
        result = resize_image_for_display(img, 1920)
        
        # Calculate expected dimensions
        scale = 1920 / 2560
        expected_height = int(1440 * scale)
        
        assert result.size == (1920, expected_height)
    
    def test_resize_image_aspect_ratio_preserved_tall(self):
        """Test that aspect ratio is preserved for tall images."""
        from backend.utils.image_utils import resize_image_for_display
        
        # Tall image (9:16)
        img = Image.new('RGB', (1080, 1920))
        result = resize_image_for_display(img, 800)
        
        # Calculate expected dimensions
        scale = 800 / 1080
        expected_height = int(1920 * scale)
        
        assert result.size == (800, expected_height)
    
    def test_resize_image_aspect_ratio_preserved_square(self):
        """Test that aspect ratio is preserved for square images."""
        from backend.utils.image_utils import resize_image_for_display
        
        # Square image
        img = Image.new('RGB', (2000, 2000))
        result = resize_image_for_display(img, 1000)
        
        # Should be square after resize
        assert result.size == (1000, 1000)
    
    def test_resize_image_very_small_dimensions(self):
        """Test resize with very small images."""
        from backend.utils.image_utils import resize_image_for_display
        
        # Tiny image
        img = Image.new('RGB', (10, 10))
        result = resize_image_for_display(img, 1920)
        
        # Should not upscale
        assert result.size == (10, 10)
    
    def test_resize_image_very_large_dimensions(self):
        """Test resize with very large images."""
        from backend.utils.image_utils import resize_image_for_display
        
        # Very large image
        img = Image.new('RGB', (10000, 8000))
        result = resize_image_for_display(img, 1920)
        
        # Should downscale significantly
        assert result.size[0] == 1920
        expected_height = int(8000 * (1920 / 10000))
        assert result.size[1] == expected_height
    
    def test_resize_image_odd_dimensions(self):
        """Test resize with odd-numbered dimensions."""
        from backend.utils.image_utils import resize_image_for_display
        
        # Odd dimensions
        img = Image.new('RGB', (2501, 1403))
        result = resize_image_for_display(img, 1920)
        
        # Should handle odd numbers correctly
        assert result.size[0] == 1920
        assert isinstance(result.size[1], int)
    
    def test_resize_image_different_color_modes(self):
        """Test resize with different PIL image modes."""
        from backend.utils.image_utils import resize_image_for_display
        
        # RGB
        img_rgb = Image.new('RGB', (2000, 1000))
        result_rgb = resize_image_for_display(img_rgb, 1000)
        assert result_rgb.size[0] == 1000
        
        # RGBA (with alpha channel)
        img_rgba = Image.new('RGBA', (2000, 1000))
        result_rgba = resize_image_for_display(img_rgba, 1000)
        assert result_rgba.size[0] == 1000
        
        # Grayscale
        img_l = Image.new('L', (2000, 1000))
        result_l = resize_image_for_display(img_l, 1000)
        assert result_l.size[0] == 1000


@pytest.mark.unit
@pytest.mark.parametrize("width,expected_quality", [
    (100, 88),
    (500, 88),
    (800, 88),
    (801, 85),
    (1000, 85),
    (1920, 85),
    (1921, 82),
    (2500, 82),
    (4000, 82),
    (8000, 82),
])
class TestJPEGQualityParameterized:
    """Parametrized tests for JPEG quality calculation."""
    
    def test_quality_for_width(self, width, expected_quality):
        """Test JPEG quality calculation for various widths."""
        from backend.utils.image_utils import calculate_optimal_jpeg_quality
        
        actual_quality = calculate_optimal_jpeg_quality(width)
        assert actual_quality == expected_quality, \
            f"Width {width} should produce quality {expected_quality}, got {actual_quality}"


@pytest.mark.unit
@pytest.mark.parametrize("original_width,max_width,should_resize", [
    (1920, 1920, False),  # Exact match
    (1000, 1920, False),  # Smaller, no upscale
    (500, 1920, False),   # Much smaller
    (2000, 1920, True),   # Slightly larger
    (3840, 1920, True),   # Much larger
    (10000, 1920, True),  # Very large
])
class TestResizeDecisionParameterized:
    """Parametrized tests for resize decision logic."""
    
    def test_resize_decision(self, original_width, max_width, should_resize):
        """Test whether resize is needed for various dimensions."""
        from backend.utils.image_utils import resize_image_for_display
        
        # Create test image
        img = Image.new('RGB', (original_width, 1000))
        result = resize_image_for_display(img, max_width)
        
        if should_resize:
            assert result.size[0] == max_width, \
                f"Image {original_width}px should be resized to {max_width}px"
        else:
            assert result.size[0] == original_width, \
                f"Image {original_width}px should NOT be resized (no upscaling)"
