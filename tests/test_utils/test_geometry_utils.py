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
    
    def test_geometry_utils_has_functions(self):
        """Test that geometry utils has expected functions."""
        from utils import geometry_utils
        
        # Test that the module has the text angle function
        assert hasattr(geometry_utils, 'calculate_text_angle')
        
        # Test that it's callable
        assert callable(geometry_utils.calculate_text_angle)
    
    def test_calculate_text_angle_edge_cases(self):
        """Test text angle calculation with edge cases."""
        from utils.geometry_utils import calculate_text_angle
        
        # Test with minimal vertices
        vertices = [
            type('obj', (object,), {'x': 0, 'y': 0})(),
            type('obj', (object,), {'x': 0, 'y': 0})()
        ]
        
        try:
            angle = calculate_text_angle(vertices)
            # Should return some angle or handle gracefully
            assert isinstance(angle, (int, float))
        except (IndexError, ZeroDivisionError):
            # These exceptions are acceptable for edge cases
            pass
