"""Tests for main.py entry point."""

import pytest
from unittest.mock import patch


@pytest.mark.unit
class TestMainEntry:
    """Test main.py module."""
    
    def test_app_exists(self):
        """Test that app is defined in main module."""
        import main
        
        assert hasattr(main, 'app')
        assert main.app is not None
    
    def test_app_is_fastapi(self):
        """Test that app is a FastAPI instance."""
        import main
        from fastapi import FastAPI
        
        assert isinstance(main.app, FastAPI)
    
    def test_main_execution(self):
        """Test main module loads without errors."""
        # Just verify the module loads - don't mock uvicorn
        import main
        assert main is not None
        assert hasattr(main, 'app')
