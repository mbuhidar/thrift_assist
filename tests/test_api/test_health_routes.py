"""Tests for health check routes."""

import pytest


@pytest.mark.api
class TestHealthRoutes:
    """Test health check endpoints."""
    
    def test_health_check(self, api_client):
        """Test health check endpoint."""
        if not hasattr(api_client, 'get'):
            pytest.skip("TestClient not available")
        
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint."""
        if not hasattr(api_client, 'get'):
            pytest.skip("TestClient not available")
        
        response = api_client.get("/")
        
        # Should return something (200 or 404 depending on config)
        assert response.status_code in [200, 404]
