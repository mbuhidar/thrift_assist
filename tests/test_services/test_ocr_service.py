"""
Tests for OCR service functionality.
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
import numpy as np
from pathlib import Path


@pytest.mark.ocr
@pytest.mark.unit
class TestOCRService:
    """Test OCR service methods."""
    
    def test_format_matches_for_api(self, mock_ocr_results):
        """Test formatting matches for API response."""
        from backend.services.ocr_service import OCRService
        
        service = OCRService()
        formatted = service.format_matches_for_api(mock_ocr_results)
        
        assert isinstance(formatted, dict)
        assert 'test phrase' in formatted
        assert len(formatted['test phrase']) == 1
        assert formatted['test phrase'][0]['score'] == 95.5
    
    def test_detect_phrases_success(self, sample_image_file, mock_search_phrases):
        """Test phrase detection with mocked detector."""
        from backend.services.ocr_service import OCRService
        
        # Mock the VisionPhraseDetector class before service initialization
        with patch('backend.services.ocr_service.VisionPhraseDetector') as MockDetector:
            mock_detector_instance = Mock()
            mock_detector_instance.detect.return_value = {
                'matches': {
                    'test phrase': [
                        (
                            {
                                'text': 'test phrase found',
                                'angle': 0,
                                'annotations': []
                            },
                            95.5,
                            'complete_phrase'
                        )
                    ]
                },
                'annotated_image': np.zeros((300, 400, 3), dtype=np.uint8),
                'all_detected_text': 'test phrase found'
            }
            MockDetector.return_value = mock_detector_instance
            
            service = OCRService()
            
            results = service.detect_phrases(
                image_path=str(sample_image_file),
                search_phrases=mock_search_phrases,
                threshold=80
            )
            
            assert results is not None
            assert 'matches' in results
            assert 'test phrase' in results['matches']
    
    def test_detect_phrases_no_text_found(self, sample_image_file, mock_search_phrases):
        """Test phrase detection when no text is found."""
        from backend.services.ocr_service import OCRService
        
        # Mock the VisionPhraseDetector to return None
        with patch('backend.services.ocr_service.VisionPhraseDetector') as MockDetector:
            mock_detector_instance = Mock()
            mock_detector_instance.detect.return_value = None
            MockDetector.return_value = mock_detector_instance
            
            service = OCRService()
            
            results = service.detect_phrases(
                image_path=str(sample_image_file),
                search_phrases=mock_search_phrases,
                threshold=80
            )
            
            # Service returns a result dict with empty matches, not None
            assert results is not None
            assert 'matches' in results
            assert len(results['matches']) == 0  # No matches found

    def test_detect_phrases_invalid_image_path(self, mock_search_phrases):
        """Test phrase detection with invalid image path."""
        from backend.services.ocr_service import OCRService
        
        service = OCRService()
        results = service.detect_phrases(
            image_path="/nonexistent/path.jpg",
            search_phrases=mock_search_phrases,
            threshold=80
        )
        
        assert results is None


@pytest.mark.ocr
@pytest.mark.integration
@pytest.mark.slow
class TestOCRServiceIntegration:
    """Integration tests for OCR service (slower, more comprehensive)."""
    
    def test_end_to_end_detection(self, sample_image_file):
        """Test complete OCR detection pipeline."""
        # This would test with actual OCR models (slower)
        # Marked as integration and slow
        pytest.skip("Integration test - requires OCR models")
