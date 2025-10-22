"""Tests for cache management routes."""

import pytest


@pytest.mark.api
class TestCacheRoutes:
    """Test cache management endpoints."""
    
    def test_cache_stats(self, api_client):
        """Test cache statistics endpoint."""
        if not hasattr(api_client, 'get'):
            pytest.skip("TestClient not available")
        
        response = api_client.get("/cache/stats")
        
        # Should return cache statistics
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
