"""
Tests for main application entry point.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestMainApp:
    """Test main application entry point."""
    
    def test_main_module_imports(self):
        """Test that main module can be imported."""
        try:
            import main
            assert hasattr(main, 'app')
        except ImportError as e:
            pytest.fail(f"Failed to import main: {e}")
    
    def test_main_app_type(self):
        """Test that app is a FastAPI application."""
        import main
        from fastapi import FastAPI
        
        assert isinstance(main.app, FastAPI)
    
    def test_main_app_has_routes(self):
        """Test that app has expected routes."""
        import main
        
        # Get all routes
        routes = [route.path for route in main.app.routes]
        
        # Should have at least the root route
        assert any('/' in route for route in routes)
    
    def test_main_app_configuration(self):
        """Test that app has basic configuration."""
        import main
        
        # App should have a title
        assert hasattr(main.app, 'title')
        assert main.app.title is not None
        
        # App should have routes defined
        assert len(main.app.routes) > 0


@pytest.mark.integration
@pytest.mark.smoke
class TestMainIntegration:
    """Integration tests for main application."""
    
    def test_main_module_structure(self):
        """Test that main module has expected structure."""
        import main
        
        # Should have app instance
        assert hasattr(main, 'app')
        
        # Should be importable without executing
        assert main.app is not None
        
        # App should be a FastAPI instance
        from fastapi import FastAPI
        assert isinstance(main.app, FastAPI)
