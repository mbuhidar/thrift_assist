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

# Import OCR functionality - hybrid approach to ensure full compatibility
try:
    from vision.detector import VisionPhraseDetector
    from config.vision_config import VisionConfig
    # Also import key functions from legacy module for missing functionality
    from thriftassist_googlevision import (
        normalize_text_for_search,
        is_meaningful_phrase,
        explain_match_score,
        try_reverse_text_matching
    )
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ OCR module not available: {e}")
    OCR_AVAILABLE = False
    
    # Stub implementations
    class VisionPhraseDetector:
        def detect(self, *args, **kwargs):
            import numpy as np
            return {
                'total_matches': 0,
                'matches': {},
                'annotated_image': np.zeros((100, 100, 3), dtype=np.uint8),
                'all_text': 'OCR module not available'
            }
    
    def normalize_text_for_search(text):
        return text.lower()
    
    def is_meaningful_phrase(phrase):
        return True
    
    def explain_match_score(*args, **kwargs):
        return {'explanation': 'OCR not available'}
    
    def try_reverse_text_matching(*args, **kwargs):
        return 0


class OCRService:
    """Service for OCR text detection and phrase matching."""
    
    def __init__(self):
        self.ocr_available = OCR_AVAILABLE
        if self.ocr_available:
            # Initialize detector with config
            config = VisionConfig()
            config.fuzz_threshold = settings.DEFAULT_THRESHOLD
            config.default_text_scale = settings.DEFAULT_TEXT_SCALE
            self.detector = VisionPhraseDetector(config)
        else:
            self.detector = VisionPhraseDetector()  # Stub version
    
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
        Enhanced with additional functionality from legacy implementation.
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
            
            # Pre-filter meaningful phrases using legacy functionality
            filtered_phrases = [
                phrase for phrase in search_phrases 
                if is_meaningful_phrase(phrase)
            ]
            
            if len(filtered_phrases) < len(search_phrases):
                skipped = set(search_phrases) - set(filtered_phrases)
                print(f"⏭️  Skipped {len(skipped)} common word phrases: {', '.join(skipped)}")
            
            results = self.detector.detect(
                image_path=image_path,
                search_phrases=filtered_phrases,
                threshold=threshold,
                text_scale=text_scale,
                show_plot=show_plot
            )
            
            if results:
                print(f"✅ OCR completed: {results.get('total_matches', 0)} matches found")
                
                # Enhance results with explainability data
                results = self._enhance_with_explanations(results, threshold)
            else:
                print("⚠️ OCR returned no results")
            
            return results
            
        except Exception as e:
            print(f"❌ OCR error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _enhance_with_explanations(self, results: Dict[str, Any], threshold: int) -> Dict[str, Any]:
        """
        Add explanation data to matches using legacy explainability functions.
        """
        if not results or 'matches' not in results:
            return results
        
        enhanced_matches = {}
        
        for phrase, matches in results['matches'].items():
            enhanced_matches[phrase] = []
            
            for match_data, score, match_type in matches:
                # Add explanation using legacy function
                match_text = match_data.get('text', '')
                explanation = explain_match_score(phrase, match_text, score, match_type)
                
                # Add explanation to match data
                if isinstance(match_data, dict):
                    match_data['explanation'] = explanation
                
                enhanced_matches[phrase].append((match_data, score, match_type))
        
        results['matches'] = enhanced_matches
        return results
    
    def normalize_search_text(self, text: str) -> str:
        """
        Normalize text for searching using legacy normalization logic.
        """
        if not self.ocr_available:
            return text.lower()
        
        return normalize_text_for_search(text)
    
    def calculate_match_similarity(self, phrase: str, text: str) -> float:
        """
        Calculate similarity between phrase and text using legacy matching logic.
        """
        if not self.ocr_available:
            return 0.0
        
        return try_reverse_text_matching(phrase, text)
    
    def validate_search_phrase(self, phrase: str) -> bool:
        """
        Check if a phrase is meaningful for searching.
        """
        if not self.ocr_available:
            return True
        
        return is_meaningful_phrase(phrase)
    
    def format_matches_for_api(self, results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert OCR results to API-friendly format with explainability data.
        
        Args:
            results: Raw OCR results dictionary
            
        Returns:
            Serializable matches dictionary
        """
        serializable_matches = {}
        
        for phrase, matches in results.get('matches', {}).items():
            serializable_matches[phrase] = []
            for match_data, score, match_type in matches:
                match_dict = {
                    'text': match_data.get('text', ''),
                    'score': float(score),
                    'match_type': match_type,
                    'angle': match_data.get('angle', 0),
                    'is_spanning': 'span_info' in match_data
                }
                
                # Add explanation if available
                if 'explanation' in match_data:
                    match_dict['explanation'] = match_data['explanation']
                
                serializable_matches[phrase].append(match_dict)
        
        return serializable_matches


# Create global OCR service instance
ocr_service = OCRService()
