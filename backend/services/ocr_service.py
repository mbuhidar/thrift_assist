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

# Import OCR functionality - use VisionPhraseDetector directly
try:
    from vision.detector import VisionPhraseDetector
    from config.vision_config import VisionConfig
    # Import utility functions from utils package instead
    from utils.text_utils import normalize_text_for_search, is_meaningful_phrase
    try:
        from rapidfuzz import fuzz
        FUZZY_AVAILABLE = True
    except ImportError:
        FUZZY_AVAILABLE = False
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ OCR module not available: {e}")
    OCR_AVAILABLE = False
    FUZZY_AVAILABLE = False
    
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


def explain_match_score(phrase, matched_text, score, match_type):
    """
    Generate human-readable explanation for why a match occurred.
    
    Returns:
        dict: Explanation with reasoning, contributing factors, and confidence breakdown
    """
    explanation = {
        'phrase_searched': phrase,
        'text_found': matched_text,
        'overall_score': score,
        'match_type': match_type,
        'reasoning': [],
        'confidence_factors': {},
        'warnings': []
    }
    
    try:
        phrase_normalized = normalize_text_for_search(phrase)
        text_normalized = normalize_text_for_search(matched_text)
    except:
        phrase_normalized = phrase.lower()
        text_normalized = matched_text.lower()
    
    # Exact match reasoning
    if phrase_normalized in text_normalized:
        explanation['reasoning'].append("Exact substring match found (case-insensitive)")
        explanation['confidence_factors']['exact_match'] = 100
    
    # Word-level analysis
    phrase_words = set(phrase_normalized.split())
    text_words = set(text_normalized.split())
    common_words = phrase_words & text_words
    
    if common_words:
        word_match_pct = (len(common_words) / len(phrase_words)) * 100
        explanation['confidence_factors']['word_overlap'] = round(word_match_pct, 1)
        explanation['reasoning'].append(
            f"{len(common_words)}/{len(phrase_words)} words matched: {', '.join(sorted(common_words))}"
        )
    
    # Character similarity
    if FUZZY_AVAILABLE:
        try:
            char_similarity = fuzz.ratio(phrase_normalized, text_normalized)
            explanation['confidence_factors']['character_similarity'] = char_similarity
            
            if char_similarity < 70:
                explanation['warnings'].append("Low character-level similarity - may be a false positive")
        except:
            pass
    
    # Orientation analysis
    if match_type in ['upside_down', 'reversed']:
        explanation['reasoning'].append(f"Text detected at unusual orientation ({match_type})")
        explanation['warnings'].append("Verify this match - text may be rotated or mirrored")
    
    # Spanning match explanation
    if 'span' in match_type:
        explanation['reasoning'].append("Match spans multiple lines of text")
        explanation['confidence_factors']['spanning'] = True
    
    # Final confidence assessment
    if score == 100:
        explanation['confidence_level'] = 'Very High'
        explanation['recommendation'] = 'This is an exact match'
    elif score >= 85:
        explanation['confidence_level'] = 'High'
        explanation['recommendation'] = 'Strong match - likely correct'
    elif score >= 70:
        explanation['confidence_level'] = 'Medium'
        explanation['recommendation'] = 'Probable match - verify if critical'
    else:
        explanation['confidence_level'] = 'Low'
        explanation['recommendation'] = 'Weak match - manual verification recommended'
        explanation['warnings'].append(f"Score below 70% - may be incorrect")
    
    return explanation


def try_reverse_text_matching(phrase, text):
    """
    Try matching text that might be upside down or reversed.
    Returns similarity score if a good match is found.
    """
    if not FUZZY_AVAILABLE:
        return 0
    
    try:
        # Normalize both phrase and text to lowercase for case-insensitive matching
        phrase_normalized = normalize_text_for_search(phrase)
        text_normalized = normalize_text_for_search(text)
    except:
        phrase_normalized = phrase.lower()
        text_normalized = text.lower()
    
    try:
        # Try normal matching first
        normal_score = fuzz.token_set_ratio(phrase_normalized, text_normalized)
        
        # Try word-reversed matching (for upside-down text)
        text_words_reversed = ' '.join(text_normalized.split()[::-1])
        reverse_score = fuzz.token_set_ratio(phrase_normalized, text_words_reversed)
        
        # Try character-reversed matching (for completely flipped text)
        text_char_reversed = text_normalized[::-1]
        char_reverse_score = fuzz.token_set_ratio(phrase_normalized, text_char_reversed)
        
        return max(normal_score, reverse_score, char_reverse_score)
    except:
        return 0


class OCRService:
    """Service for OCR text detection and phrase matching."""
    
    def __init__(self):
        self.ocr_available = OCR_AVAILABLE
        if self.ocr_available:
            # Initialize detector with config
            try:
                config = VisionConfig()
                config.fuzz_threshold = settings.DEFAULT_THRESHOLD
                config.default_text_scale = settings.DEFAULT_TEXT_SCALE
                self.detector = VisionPhraseDetector(config)
                print("✅ OCR service initialized with VisionPhraseDetector")
            except Exception as e:
                print(f"⚠️ Failed to initialize VisionPhraseDetector: {e}")
                self.ocr_available = False
                self.detector = VisionPhraseDetector()  # Stub version
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
        """
        if not self.ocr_available:
            print("❌ OCR service not available")
            return {
                'image': None,
                'annotated_image': None,
                'matches': {},
                'total_matches': 0,
                'all_text': 'OCR service not available'
            }
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        # Use default values from settings if not provided
        threshold = threshold if threshold is not None else settings.DEFAULT_THRESHOLD
        text_scale = text_scale if text_scale is not None else settings.DEFAULT_TEXT_SCALE
        
        try:
            print(f"🔍 Running OCR with threshold={threshold}%, text_scale={text_scale}%")
            
            # Pre-filter meaningful phrases if utility is available
            filtered_phrases = []
            for phrase in search_phrases:
                try:
                    if is_meaningful_phrase(phrase):
                        filtered_phrases.append(phrase)
                    else:
                        print(f"⏭️  Skipping common word phrase: '{phrase}'")
                except:
                    # If is_meaningful_phrase fails, include the phrase anyway
                    filtered_phrases.append(phrase)
            
            if not filtered_phrases:
                filtered_phrases = search_phrases  # Fallback to all phrases
            
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
                return {
                    'image': None,
                    'annotated_image': None,
                    'matches': {},
                    'total_matches': 0,
                    'all_text': 'No text detected'
                }
            
            return results
            
        except Exception as e:
            print(f"❌ OCR error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'image': None,
                'annotated_image': None,
                'matches': {},
                'total_matches': 0,
                'all_text': f'OCR error: {str(e)}'
            }
    
    def _enhance_with_explanations(self, results: Dict[str, Any], threshold: int) -> Dict[str, Any]:
        """
        Add explanation data to matches using explainability functions.
        """
        if not results or 'matches' not in results:
            return results
        
        enhanced_matches = {}
        
        for phrase, matches in results['matches'].items():
            enhanced_matches[phrase] = []
            
            for match_data, score, match_type in matches:
                # Add explanation using explainability function
                match_text = match_data.get('text', '') if isinstance(match_data, dict) else str(match_data)
                explanation = explain_match_score(phrase, match_text, score, match_type)
                
                # Add explanation to match data
                if isinstance(match_data, dict):
                    match_data['explanation'] = explanation
                else:
                    # Convert to dict if not already
                    match_data = {
                        'text': str(match_data),
                        'explanation': explanation
                    }
                
                enhanced_matches[phrase].append((match_data, score, match_type))
        
        results['matches'] = enhanced_matches
        return results
    
    def calculate_match_similarity(self, phrase: str, text: str) -> float:
        """
        Calculate similarity between phrase and text using fuzzy matching logic.
        """
        if not self.ocr_available or not FUZZY_AVAILABLE:
            return 0.0
        
        return try_reverse_text_matching(phrase, text)
    
    def normalize_search_text(self, text: str) -> str:
        """
        Normalize text for searching.
        """
        if not self.ocr_available:
            return text.lower()
        
        try:
            return normalize_text_for_search(text)
        except:
            return text.lower()
    
    def validate_search_phrase(self, phrase: str) -> bool:
        """
        Check if a phrase is meaningful for searching.
        """
        if not self.ocr_available:
            return True
        
        try:
            return is_meaningful_phrase(phrase)
        except:
            return True
    
    def format_matches_for_api(self, results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert OCR results to API-friendly format with explainability data.
        """
        if not results:
            return {}
        
        serializable_matches = {}
        
        for phrase, matches in results.get('matches', {}).items():
            serializable_matches[phrase] = []
            for match_data, score, match_type in matches:
                match_dict = {
                    'text': match_data.get('text', '') if isinstance(match_data, dict) else str(match_data),
                    'score': float(score) if score is not None else 0.0,
                    'match_type': match_type or 'unknown',
                    'angle': match_data.get('angle', 0) if isinstance(match_data, dict) else 0,
                    'is_spanning': 'span_info' in match_data if isinstance(match_data, dict) else False
                }
                
                # Add explanation if available
                if isinstance(match_data, dict) and 'explanation' in match_data:
                    match_dict['explanation'] = match_data['explanation']
                
                serializable_matches[phrase].append(match_dict)
        
        return serializable_matches


# Create global OCR service instance
ocr_service = OCRService()
