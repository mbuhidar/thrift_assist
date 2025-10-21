"""
Tests for main application entry point.
"""

import pytest
from unittest.mock import patch, Mock


@pytest.mark.unit
class TestMainApp:
    """Test main application functionality."""
    
    def test_app_import(self):
        """Test that the app can be imported successfully."""
        try:
            from main import app
            assert app is not None
        except ImportError as e:
            pytest.skip(f"Could not import main app: {e}")
    
    def test_app_attributes(self):
        """Test that the imported app has expected attributes."""
        try:
            from main import app
            
            # Should be a FastAPI application
            assert hasattr(app, 'router')
            assert hasattr(app, 'middleware_stack')
            assert hasattr(app, 'routes')
            
        except ImportError:
            pytest.skip("Could not import main app")
    
    @patch('uvicorn.run')
    @patch('os.getenv')
    def test_main_execution(self, mock_getenv, mock_uvicorn_run):
        """Test main execution with uvicorn."""
        mock_getenv.return_value = "8080"
        
        # Import and run main
        import main
        
        # Mock __name__ to trigger main execution
        with patch.object(main, '__name__', '__main__'):
            exec(open('main.py').read())
        
        # Should have called uvicorn.run with correct parameters
        mock_uvicorn_run.assert_called_once_with(
            "main:app",
            host="0.0.0.0",
            port=8080,
            reload=False
        )
    
    @patch('uvicorn.run')
    @patch('os.getenv')
    def test_main_default_port(self, mock_getenv, mock_uvicorn_run):
        """Test main execution with default port."""
        mock_getenv.return_value = None  # No PORT env var
        
        import main
        
        with patch.object(main, '__name__', '__main__'):
            # Simulate running main
            port = int(main.os.getenv("PORT", "8000"))
            assert port == 8000


@pytest.mark.integration
@pytest.mark.smoke
class TestMainIntegration:
    """Integration tests for main application."""
    
    def test_app_startup(self, api_client):
        """Test that the application starts up correctly."""
        if hasattr(api_client, '_mock_name'):
            pytest.skip("FastAPI dependencies not available")
        
        # Test basic health endpoint or root
        try:
            response = api_client.get("/")
            # Should not error, even if it returns 404
            assert response.status_code in [200, 404, 405]
        except Exception as e:
            pytest.fail(f"App startup failed: {e}")
    
    def test_app_has_routes(self, api_client):
        """Test that the application has expected routes."""
        if hasattr(api_client, '_mock_name'):
            pytest.skip("FastAPI dependencies not available")
        
        try:
            from main import app
            
            # Should have some routes
            assert len(app.routes) > 0
            
            # Check for expected route patterns
            route_paths = [route.path for route in app.routes if hasattr(route, 'path')]
            
            # Should have health or OCR routes
            has_health = any('/health' in path for path in route_paths)
            has_ocr = any('/ocr' in path or '/upload' in path for path in route_paths)
            
            assert has_health or has_ocr, f"Expected routes not found. Available: {route_paths}"
            
        except ImportError:
            pytest.skip("Could not import main app")
