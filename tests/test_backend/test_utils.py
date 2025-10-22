"""Tests for backend utilities."""

import pytest


@pytest.mark.unit
class TestBackendUtils:
    """Test backend utility functions."""
    
    def test_image_utils_module_exists(self):
        """Test backend.utils.image_utils module can be imported."""
        try:
            from backend.utils import image_utils
            assert image_utils is not None
        except ImportError:
            pytest.fail("backend.utils.image_utils should be importable")
