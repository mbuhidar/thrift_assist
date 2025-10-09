"""
OCR service for text detection and phrase matching.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core.config import settings

# Import OCR functionality
try:
    from thriftassist_googlevision import (
        detect_and_annotate_phrases,
        FUZZY_AVAILABLE,
        COMMON_WORDS
    )
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ OCR module not available: {e}")
    OCR_AVAILABLE = False
    
    # Stub function for when OCR is unavailable
    def detect_and_annotate_phrases(*args, **kwargs):
        import numpy as np
        return {
            'total_matches': 0,
            'matches': {},
            'annotated_image': np.zeros((100, 100, 3), dtype=np.uint8),
            'all_text': 'OCR module not available'
        }
    FUZZY_AVAILABLE = False
    COMMON_WORDS = set()


class OCRService:
    """Service for OCR text detection and phrase matching."""
    
    def __init__(self):
        self.ocr_available = OCR_AVAILABLE
        self.fuzzy_available = FUZZY_AVAILABLE
    
    def is_available(self) -> bool:
        """Check if OCR service is available."""
        return self.ocr_available
    
    def detect_phrases(
        self,
        image_path: str,
        search_phrases: List[str],
        threshold: int = None,
        text_scale: int = None,
        show_plot: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Detect phrases in an image using Google Cloud Vision API.
        
        Args:
            image_path: Path to the image file
            search_phrases: List of phrases to search for
            threshold: Similarity threshold (50-100)
            text_scale: Text size scale percentage
            show_plot: Whether to show matplotlib plot
            
        Returns:
            Dictionary with detection results or None on failure
        """
        if not self.ocr_available:
            print("❌ OCR service not available")
            return None
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        # Use default values from settings if not provided
        threshold = threshold if threshold is not None else settings.DEFAULT_THRESHOLD
        text_scale = text_scale if text_scale is not None else settings.DEFAULT_TEXT_SCALE
        
        try:
            print(f"🔍 Running OCR with threshold={threshold}%, text_scale={text_scale}%")
            
            results = detect_and_annotate_phrases(
                image_path=image_path,
                search_phrases=search_phrases,
                threshold=threshold,
                text_scale=text_scale,
                show_plot=show_plot
            )
            
            if results:
                print(f"✅ OCR completed: {results.get('total_matches', 0)} matches found")
            else:
                print("⚠️ OCR returned no results")
            
            return results
            
        except Exception as e:
            print(f"❌ OCR error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def format_matches_for_api(self, results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert OCR results to API-friendly format.
        
        Args:
            results: Raw OCR results dictionary
            
        Returns:
            Serializable matches dictionary
        """
        serializable_matches = {}
        
        for phrase, matches in results.get('matches', {}).items():
            serializable_matches[phrase] = []
            for match_data, score, match_type in matches:
                serializable_matches[phrase].append({
                    'text': match_data.get('text', ''),
                    'score': float(score),
                    'match_type': match_type,
                    'angle': match_data.get('angle', 0),
                    'is_spanning': 'span_info' in match_data
                })
        
        return serializable_matches


# Create global OCR service instance
ocr_service = OCRService()
