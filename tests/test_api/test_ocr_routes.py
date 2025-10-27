"""
Tests for OCR API routes.
"""

import pytest
import json
from unittest.mock import patch, Mock
import io


@pytest.mark.api
class TestOCRRoutes:
    """Test OCR API endpoints."""
    
    def test_upload_and_detect_success(self, api_client, sample_image, mock_ocr_results):
        """Test successful image upload and phrase detection."""
        # Skip if API client is mocked (dependencies not available)
        if hasattr(api_client, '_mock_name'):
            pytest.skip("FastAPI dependencies not available")

        # Create proper mock OCR results with annotated_image
        import numpy as np
        import tempfile
        import cv2
        import os
        
        mock_results_with_image = mock_ocr_results.copy()
        mock_results_with_image['annotated_image'] = sample_image
        
        # Create a real temporary file that cv2.imread can read
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        try:
            # Write the sample image to the temp file
            cv2.imwrite(temp_path, sample_image)
            os.close(temp_fd)  # Close the file descriptor

            with patch('backend.api.routes.ocr.ocr_service') as mock_ocr, \
                 patch('backend.api.routes.ocr.cache_service') as mock_cache, \
                 patch('backend.api.routes.ocr.image_service') as mock_image:

                # Setup mocks with proper return values
                mock_cache.get_image_hash.return_value = "test_hash"
                mock_cache.get_cached_result.return_value = None
                mock_cache.cache_result.return_value = None
                
                mock_image.validate_image_data.return_value = True
                mock_image.save_temp_image.return_value = temp_path  # Return real temp file path
                mock_image.base64_to_array.return_value = sample_image
                
                # Mock OCR service with results that include annotated_image
                mock_ocr.detect_phrases.return_value = mock_results_with_image
                mock_ocr.format_matches_for_api.return_value = mock_results_with_image['matches']

                # Create test file
                test_file = pytest.create_test_upload_file(sample_image)

                # Make request
                response = api_client.post(
                    "/ocr/upload",
                    files={"file": ("test.jpg", test_file.file, "image/jpeg")},
                    data={
                        "search_phrases": json.dumps(["test phrase"]),
                        "threshold": 80,
                        "text_scale": 100,
                        "max_image_width": 1920
                    }
                )

                # Assertions
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "matches" in data
                assert "test phrase" in data["matches"]
                
                # Check for image data in response
                assert "image" in data or "annotated_image_base64" in data
                
                # Verify mocks were called
                mock_ocr.detect_phrases.assert_called_once()
                mock_cache.get_image_hash.assert_called()
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_upload_invalid_file_type(self, api_client):
        """Test upload with invalid file type."""
        if hasattr(api_client, '_mock_name'):
            pytest.skip("FastAPI dependencies not available")
        
        # Create text file instead of image
        text_content = b"This is not an image"
        text_file = io.BytesIO(text_content)
        
        response = api_client.post(
            "/ocr/upload",
            files={"file": ("test.txt", text_file, "text/plain")},
            data={
                "search_phrases": json.dumps(["test"]),
                "threshold": 80
            }
        )
        
        assert response.status_code == 400
        error_detail = response.json().get("detail", "")
        assert "File must be an image" in error_detail
    
    def test_upload_invalid_search_phrases(self, api_client, sample_image):
        """Test upload with invalid search phrases format."""
        if hasattr(api_client, '_mock_name'):
            pytest.skip("FastAPI dependencies not available")
        
        test_file = pytest.create_test_upload_file(sample_image)
        
        response = api_client.post(
            "/ocr/upload",
            files={"file": ("test.jpg", test_file.file, "image/jpeg")},
            data={
                "search_phrases": "invalid json",
                "threshold": 80
            }
        )
        
        assert response.status_code == 400
        error_detail = response.json().get("detail", "")
        assert "Invalid search_phrases format" in error_detail


@pytest.mark.unit
class TestOCRRoutesUnit:
    """Unit tests that don't require full API setup."""
    
    def test_search_phrases_parsing(self):
        """Test search phrases JSON parsing logic."""
        import json
        
        # Valid JSON
        valid_json = '["test phrase", "another phrase"]'
        phrases = json.loads(valid_json)
        assert isinstance(phrases, list)
        assert len(phrases) == 2
        
        # Invalid JSON should raise exception
        with pytest.raises(json.JSONDecodeError):
            json.loads("invalid json")
    
    def test_image_dimensions_logic(self):
        """Test image dimension calculations."""
        original_width = 2000
        max_width = 1200
        
        if original_width > max_width:
            scale_factor = max_width / original_width
            assert scale_factor == 0.6
            
            new_width = max_width
            assert new_width == 1200
            
            original_height = 1500
            new_height = int(original_height * scale_factor)
            assert new_height == 900
