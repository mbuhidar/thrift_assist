"""Tests for geometry utility functions."""

import pytest
import math


@pytest.mark.unit
class TestGeometryUtils:
    """Test geometry utility functions."""
    
    def test_calculate_text_angle(self):
        """Test text angle calculation."""
        from utils.geometry_utils import calculate_text_angle
        
        # Create mock vertices for horizontal text
        vertices = [
            type('obj', (object,), {'x': 0, 'y': 0})(),
            type('obj', (object,), {'x': 10, 'y': 0})(),
            type('obj', (object,), {'x': 10, 'y': 5})(),
            type('obj', (object,), {'x': 0, 'y': 5})()
        ]
        
        angle = calculate_text_angle(vertices)
        
        # Should return an angle (may be 0 for horizontal)
        assert isinstance(angle, (int, float))
        assert -180 <= angle <= 180
    
    def test_geometry_utils_module_exists(self):
        """Test geometry utils module can be imported."""
        try:
            from utils import geometry_utils
            assert geometry_utils is not None
        except ImportError:
            pytest.fail("utils.geometry_utils should be importable")
    
    def test_calculate_y_position(self):
        """Test Y position calculation."""
        from utils.geometry_utils import calculate_y_position
        
        # Simple square bbox
        vertices = [
            type('obj', (object,), {'x': 0, 'y': 0})(),
            type('obj', (object,), {'x': 10, 'y': 0})(),
            type('obj', (object,), {'x': 10, 'y': 10})(),
            type('obj', (object,), {'x': 0, 'y': 10})()
        ]
        
        y_pos = calculate_y_position(vertices)
        
        # Should return average Y position
        assert isinstance(y_pos, (int, float))
        assert y_pos >= 0
    
    def test_points_to_line_segment(self):
        """Test converting points to line segment."""
        from utils.geometry_utils import points_to_line_segment
        
        # Two points
        points = [
            type('obj', (object,), {'x': 0, 'y': 0})(),
            type('obj', (object,), {'x': 10, 'y': 5})()
        ]
        
        segment = points_to_line_segment(points)
        
        # Should return a valid line segment
        assert segment is not None
        assert len(segment) >= 2
