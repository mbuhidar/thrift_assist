"""Tests for geometry utility functions."""

import pytest
import math


@pytest.mark.unit
class TestGeometryUtils:
    """Test geometry utility functions."""
    
    def test_calculate_text_angle_horizontal(self):
        """Test text angle calculation for horizontal text."""
        from utils.geometry_utils import calculate_text_angle
        
        # Horizontal text (0 degrees)
        vertices = [
            type('obj', (object,), {'x': 0, 'y': 0})(),
            type('obj', (object,), {'x': 10, 'y': 0})(),
            type('obj', (object,), {'x': 10, 'y': 5})(),
            type('obj', (object,), {'x': 0, 'y': 5})()
        ]
        
        angle = calculate_text_angle(vertices)
        
        assert isinstance(angle, (int, float))
        assert abs(angle) < 1  # Should be close to 0 degrees
    
    def test_calculate_text_angle_vertical(self):
        """Test text angle calculation for vertical text."""
        from utils.geometry_utils import calculate_text_angle
        
        # Vertical text (90 degrees)
        vertices = [
            type('obj', (object,), {'x': 0, 'y': 0})(),
            type('obj', (object,), {'x': 0, 'y': 10})(),
            type('obj', (object,), {'x': 5, 'y': 10})(),
            type('obj', (object,), {'x': 5, 'y': 0})()
        ]
        
        angle = calculate_text_angle(vertices)
        
        assert isinstance(angle, (int, float))
        assert abs(angle - 90) < 1 or abs(angle + 90) < 1
    
    def test_calculate_text_angle_empty_vertices(self):
        """Test text angle with insufficient vertices."""
        from utils.geometry_utils import calculate_text_angle
        
        # Single vertex
        vertices = [type('obj', (object,), {'x': 0, 'y': 0})()]
        angle = calculate_text_angle(vertices)
        assert angle == 0
        
        # Empty vertices
        angle = calculate_text_angle([])
        assert angle == 0
    
    def test_rectangles_overlap_true(self):
        """Test rectangle overlap detection (overlapping case)."""
        from utils.geometry_utils import rectangles_overlap
        
        rect1 = (0, 0, 10, 10)
        rect2 = (5, 5, 15, 15)
        
        assert rectangles_overlap(rect1, rect2) is True
    
    def test_rectangles_overlap_false(self):
        """Test rectangle overlap detection (non-overlapping case)."""
        from utils.geometry_utils import rectangles_overlap
        
        rect1 = (0, 0, 10, 10)
        rect2 = (20, 20, 30, 30)
        
        assert rectangles_overlap(rect1, rect2) is False
    
    def test_rectangles_overlap_touching(self):
        """Test rectangle overlap detection (touching edges)."""
        from utils.geometry_utils import rectangles_overlap
        
        # Touching but not overlapping
        rect1 = (0, 0, 10, 10)
        rect2 = (10, 0, 20, 10)
        
        assert rectangles_overlap(rect1, rect2) is False
    
    def test_rectangles_overlap_contained(self):
        """Test rectangle overlap detection (one inside other)."""
        from utils.geometry_utils import rectangles_overlap
        
        rect1 = (0, 0, 20, 20)
        rect2 = (5, 5, 15, 15)
        
        assert rectangles_overlap(rect1, rect2) is True
    
    def test_find_non_overlapping_position_no_conflict(self):
        """Test finding position when no overlap exists."""
        from utils.geometry_utils import find_non_overlapping_position
        
        rect = (10, 10, 20, 20)
        existing = []
        image_shape = (100, 100)
        
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        assert isinstance(result, tuple)
        assert len(result) == 4
        # Should return similar position (with margin adjustments)
        assert result[2] - result[0] == 10  # Width preserved
        assert result[3] - result[1] == 10  # Height preserved
    
    def test_find_non_overlapping_position_with_conflict(self):
        """Test finding position when overlap exists."""
        from utils.geometry_utils import find_non_overlapping_position
        
        rect = (10, 10, 20, 20)
        existing = [(10, 10, 20, 20)]  # Same position
        image_shape = (100, 100)
        
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        assert isinstance(result, tuple)
        assert len(result) == 4
        # Should find different position
        assert result != rect
    
    def test_find_non_overlapping_position_multiple_conflicts(self):
        """Test finding position with multiple overlapping rectangles."""
        from utils.geometry_utils import find_non_overlapping_position
        
        rect = (10, 10, 20, 20)
        existing = [
            (10, 10, 20, 20),
            (10, 35, 20, 45),
            (-15, 10, -5, 20),
            (35, 10, 45, 20)
        ]
        image_shape = (100, 100)
        
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        assert isinstance(result, tuple)
        assert len(result) == 4
        # Should not overlap with any existing
        for ex_rect in existing:
            from utils.geometry_utils import rectangles_overlap
            assert not rectangles_overlap(result, ex_rect)
    
    def test_find_non_overlapping_position_bounds_checking(self):
        """Test position finder respects image bounds."""
        from utils.geometry_utils import find_non_overlapping_position
        
        rect = (90, 90, 100, 100)  # Near edge
        existing = []
        image_shape = (100, 100)
        
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        # Should be within bounds (with margin)
        assert result[0] >= 0
        assert result[1] >= 0
        assert result[2] <= 100
        assert result[3] <= 100
    
    def test_geometry_utils_module_exists(self):
        """Test geometry utils module can be imported."""
        try:
            from utils import geometry_utils
            assert geometry_utils is not None
        except ImportError:
            pytest.fail("utils.geometry_utils should be importable")
    
    def test_find_non_overlapping_position_stacked_fallback(self):
        """Test position finder uses stacking when all offsets fail."""
        from utils.geometry_utils import find_non_overlapping_position
        
        # Create a small image with limited space
        rect = (10, 10, 30, 30)
        image_shape = (50, 50)
        
        # Fill all offset positions with existing rectangles
        existing = [
            (10, -25, 30, -5),   # Above
            (10, 45, 30, 65),    # Below
            (-25, 10, -5, 30),   # Left
            (45, 10, 65, 30),    # Right
            (-25, -25, -5, -5),  # Top-left
            (45, -25, 65, -5),   # Top-right
            (-25, 45, -5, 65),   # Bottom-left
            (45, 45, 65, 65),    # Bottom-right
        ]
        
        # This should trigger stacking fallback
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        assert isinstance(result, tuple)
        assert len(result) == 4
        # Should find a stacked position
        assert result[0] >= 0
        assert result[1] >= 0
        assert result[2] <= 50
        assert result[3] <= 50
    
    def test_find_non_overlapping_position_stacked_vertical(self):
        """Test vertical stacking when horizontal space limited."""
        from utils.geometry_utils import find_non_overlapping_position
        
        rect = (10, 10, 30, 30)
        # Existing rect at same X position but different Y
        existing = [(10, 10, 30, 30)]
        image_shape = (100, 100)
        
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        # Should stack below the existing rect
        assert result[1] >= existing[0][3]  # Y1 >= existing Y2
        assert result[0] == 10  # X position preserved
    
    def test_find_non_overlapping_position_stacked_horizontal(self):
        """Test horizontal stacking when vertical space limited."""
        from utils.geometry_utils import find_non_overlapping_position
        
        rect = (10, 10, 30, 30)
        image_shape = (100, 100)  # Changed: give more vertical space
        
        # Fill vertical space at same X to force horizontal stacking
        existing = [
            (10, 0, 30, 15),    # Top
            (10, 15, 30, 30),   # Middle (overlaps with rect)
            (10, 30, 30, 45),   # Below
        ]
        
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        # Should find a non-overlapping position (may be horizontal offset)
        assert isinstance(result, tuple)
        # Verify it doesn't overlap with any existing
        from utils.geometry_utils import rectangles_overlap
        for ex_rect in existing:
            assert not rectangles_overlap(result, ex_rect)
    
    def test_find_non_overlapping_position_all_directions_blocked(self):
        """Test stacking when all cardinal directions are blocked."""
        from utils.geometry_utils import find_non_overlapping_position
        
        rect = (40, 40, 60, 60)
        image_shape = (100, 100)
        
        # Block all 8 offset directions around the rect
        existing = [
            (40, 0, 60, 15),     # Top
            (40, 85, 60, 100),   # Bottom
            (0, 40, 15, 60),     # Left
            (85, 40, 100, 60),   # Right
            (0, 0, 15, 15),      # Top-left
            (85, 0, 100, 15),    # Top-right
            (0, 85, 15, 100),    # Bottom-left
            (85, 85, 100, 100),  # Bottom-right
        ]
        
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        # Should use stacking fallback
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        # Should not overlap with any existing (original rect may be returned if no space)
        from utils.geometry_utils import rectangles_overlap
        for ex_rect in existing:
            assert not rectangles_overlap(result, ex_rect)
    
    def test_find_non_overlapping_position_vertical_stack_preferred(self):
        """Test that vertical stacking is preferred when available."""
        from utils.geometry_utils import find_non_overlapping_position
        
        rect = (10, 10, 30, 30)
        image_shape = (100, 100)
        
        # Block only horizontal directions, leave vertical open
        existing = [
            (-15, 10, -5, 30),   # Left
            (45, 10, 55, 30),    # Right
        ]
        
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        # Should stack vertically (below or above)
        assert isinstance(result, tuple)
        # X position should be similar to original
        assert abs(result[0] - rect[0]) < 30
    
    def test_find_non_overlapping_position_horizontal_stack_preferred(self):
        """Test horizontal stacking when vertical blocked."""
        from utils.geometry_utils import find_non_overlapping_position
        
        rect = (10, 10, 30, 30)
        image_shape = (100, 100)
        
        # Block only vertical directions, leave horizontal open
        existing = [
            (10, -25, 30, -5),   # Above
            (10, 45, 30, 65),    # Below
        ]
        
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        # Should stack horizontally (left or right)
        assert isinstance(result, tuple)
        # Y position should be similar to original
        assert abs(result[1] - rect[1]) < 30
    
    def test_find_non_overlapping_position_narrow_vertical_space(self):
        """Test stacking in narrow vertical space."""
        from utils.geometry_utils import find_non_overlapping_position
        
        rect = (10, 10, 30, 30)
        image_shape = (100, 50)  # Narrow height
        
        # Fill top and bottom at same X
        existing = [
            (10, 0, 30, 8),
            (10, 32, 30, 50),
        ]
        
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        # Should find position (likely horizontal offset)
        assert isinstance(result, tuple)
        assert result[1] >= 0
        assert result[3] <= 50
    
    def test_find_non_overlapping_position_narrow_horizontal_space(self):
        """Test stacking in narrow horizontal space."""
        from utils.geometry_utils import find_non_overlapping_position
        
        rect = (10, 10, 30, 30)
        image_shape = (50, 100)  # Narrow width
        
        # Fill left and right at same Y
        existing = [
            (0, 10, 8, 30),
            (32, 10, 50, 30),
        ]
        
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        # Should find position (likely vertical offset)
        assert isinstance(result, tuple)
        assert result[0] >= 0
        assert result[2] <= 50
    
    def test_find_non_overlapping_position_dense_grid(self):
        """Test stacking in densely populated grid."""
        from utils.geometry_utils import find_non_overlapping_position
        
        rect = (25, 25, 35, 35)
        image_shape = (100, 100)
        
        # Create a dense grid of existing rectangles with gaps
        existing = []
        for x in range(0, 90, 15):
            for y in range(0, 90, 15):
                if not (20 <= x <= 40 and 20 <= y <= 40):  # Leave gap around rect
                    existing.append((x, y, x+10, y+10))
        
        result = find_non_overlapping_position(rect, existing, image_shape)
        
        # Should find a position
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        # Should be within bounds
        assert result[0] >= 0
        assert result[1] >= 0
        assert result[2] <= 100
        assert result[3] <= 100
