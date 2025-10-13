"""
OCR service for text detection and phrase matching.
"""

import os
import sys
import re
from typing import List, Dict, Any, Optional, Tuple, Set

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


def preprocess_text(text: str) -> str:
    """
    Advanced text preprocessing for better matching.
    Handles special characters, whitespace, and common OCR artifacts.
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Handle common OCR artifacts
    text = text.replace('|', 'I').replace('0', 'O')  # Common substitutions
    
    # Normalize quotes and apostrophes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Single quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Double quotes
    
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s\-.,!?\'"]', ' ', text)
    
    return text.strip()


def extract_phrases_from_text(text: str, min_words: int = 2, max_words: int = 5) -> Set[str]:
    """
    Extract meaningful phrases from text for matching.
    Returns phrases of different lengths for better coverage.
    """
    if not text:
        return set()
    
    words = preprocess_text(text).split()
    phrases = set()
    
    # Extract n-grams of different sizes
    for n in range(min_words, min(max_words + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            if is_meaningful_phrase(phrase):
                phrases.add(phrase.lower())
    
    return phrases


def calculate_multi_algorithm_score(phrase: str, text: str) -> Tuple[float, str]:
    """
    Calculate similarity using multiple fuzzy matching algorithms.
    Returns the best score and the algorithm that produced it.
    """
    if not FUZZY_AVAILABLE:
        # Simple substring matching fallback
        phrase_lower = phrase.lower()
        text_lower = text.lower()
        if phrase_lower in text_lower:
            return 100.0, 'exact'
        elif any(word in text_lower for word in phrase_lower.split()):
            return 50.0, 'partial'
        return 0.0, 'none'
    
    try:
        phrase_normalized = normalize_text_for_search(phrase)
        text_normalized = normalize_text_for_search(text)
    except:
        phrase_normalized = phrase.lower()
        text_normalized = text.lower()
    
    scores = {}
    
    try:
        # Exact substring match (highest priority)
        if phrase_normalized in text_normalized:
            return 100.0, 'exact'
        
        # Token set ratio (good for word order variations)
        scores['token_set'] = fuzz.token_set_ratio(phrase_normalized, text_normalized)
        
        # Token sort ratio (good for reordered words)
        scores['token_sort'] = fuzz.token_sort_ratio(phrase_normalized, text_normalized)
        
        # Partial ratio (good for substring matches)
        scores['partial'] = fuzz.partial_ratio(phrase_normalized, text_normalized)
        
        # Standard ratio (character-level similarity)
        scores['ratio'] = fuzz.ratio(phrase_normalized, text_normalized)
        
        # Word-level matching
        phrase_words = set(phrase_normalized.split())
        text_words = set(text_normalized.split())
        word_overlap = len(phrase_words & text_words) / len(phrase_words) * 100 if phrase_words else 0
        scores['word_overlap'] = word_overlap
        
    except Exception as e:
        print(f"⚠️ Error in fuzzy matching: {e}")
        return 0.0, 'error'
    
    # Return best score and algorithm
    if scores:
        best_algo = max(scores.items(), key=lambda x: x[1])
        return float(best_algo[1]), best_algo[0]
    
    return 0.0, 'none'


def find_spanning_matches(phrase: str, text_blocks: List[Dict[str, Any]], threshold: int = 70) -> List[Tuple[str, float, str]]:
    """
    Find matches that span across multiple lines/blocks of text.
    This helps catch phrases that are split across lines.
    """
    if not text_blocks or len(text_blocks) < 2:
        return []
    
    matches = []
    phrase_normalized = normalize_text_for_search(phrase) if OCR_AVAILABLE else phrase.lower()
    
    # Try combinations of adjacent blocks
    for i in range(len(text_blocks) - 1):
        # Try 2-3 adjacent blocks
        for span_size in [2, 3]:
            if i + span_size > len(text_blocks):
                break
            
            # Combine text from adjacent blocks
            combined_text = ' '.join(
                block.get('description', '') for block in text_blocks[i:i+span_size]
            )
            
            if not combined_text.strip():
                continue
            
            # Calculate similarity
            score, algo = calculate_multi_algorithm_score(phrase, combined_text)
            
            if score >= threshold:
                matches.append((
                    combined_text,
                    score,
                    f'spanning_{span_size}_blocks_{algo}'
                ))
    
    return matches


def enhance_match_with_context(
    phrase: str,
    matched_text: str,
    score: float,
    match_type: str,
    surrounding_text: List[str] = None
) -> Dict[str, Any]:
    """
    Enhance match data with contextual information.
    """
    enhancement = {
        'matched_text': matched_text,
        'score': score,
        'match_type': match_type,
        'context': {},
        'quality_indicators': {}
    }
    
    # Add surrounding context if available
    if surrounding_text:
        enhancement['context']['surrounding'] = surrounding_text
        enhancement['context']['appears_in_sentence'] = any(
            matched_text.lower() in context.lower() for context in surrounding_text
        )
    
    # Quality indicators
    phrase_words = set(phrase.lower().split())
    matched_words = set(matched_text.lower().split())
    
    enhancement['quality_indicators']['word_coverage'] = (
        len(phrase_words & matched_words) / len(phrase_words) * 100 if phrase_words else 0
    )
    enhancement['quality_indicators']['exact_word_match'] = phrase_words == matched_words
    enhancement['quality_indicators']['length_ratio'] = (
        len(matched_text) / len(phrase) if phrase else 0
    )
    
    return enhancement


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
        Enhanced with multi-pass matching and spanning text detection.
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
            
            # Pre-filter and expand phrases
            filtered_phrases = []
            expanded_phrases = set()
            
            for phrase in search_phrases:
                try:
                    if is_meaningful_phrase(phrase):
                        filtered_phrases.append(phrase)
                        # Also extract sub-phrases for better matching
                        expanded_phrases.update(extract_phrases_from_text(phrase, min_words=1, max_words=4))
                    else:
                        print(f"⏭️  Skipping common word phrase: '{phrase}'")
                except:
                    filtered_phrases.append(phrase)
            
            # Combine original and expanded phrases
            all_search_phrases = list(set(filtered_phrases) | expanded_phrases)
            
            if not all_search_phrases:
                all_search_phrases = search_phrases
            
            print(f"📝 Searching for {len(filtered_phrases)} phrases (+ {len(expanded_phrases)} expanded)")
            
            results = self.detector.detect(
                image_path=image_path,
                search_phrases=all_search_phrases,
                threshold=threshold,
                text_scale=text_scale,
                show_plot=show_plot
            )
            
            if results:
                # Post-process with advanced matching
                results = self._apply_advanced_matching(results, search_phrases, threshold)
                
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
    
    def _apply_advanced_matching(
        self,
        results: Dict[str, Any],
        original_phrases: List[str],
        threshold: int
    ) -> Dict[str, Any]:
        """
        Apply advanced multi-pass matching to improve results.
        """
        if not results or 'all_text' not in results:
            return results
        
        all_text = results.get('all_text', '')
        existing_matches = results.get('matches', {})
        
        # Get text blocks if available for spanning detection
        text_blocks = results.get('text_blocks', [])
        
        enhanced_matches = dict(existing_matches)
        
        # Multi-pass matching for each original phrase
        for phrase in original_phrases:
            if phrase in enhanced_matches and enhanced_matches[phrase]:
                continue  # Already has matches
            
            # Try different matching strategies
            new_matches = []
            
            # Strategy 1: Multi-algorithm scoring on full text
            score, algo = calculate_multi_algorithm_score(phrase, all_text)
            if score >= threshold:
                new_matches.append((
                    {'text': all_text, 'source': 'full_text'},
                    score,
                    f'multi_algo_{algo}'
                ))
            
            # Strategy 2: Spanning text detection
            if text_blocks:
                spanning_matches = find_spanning_matches(phrase, text_blocks, threshold)
                for text, score, match_type in spanning_matches:
                    new_matches.append((
                        {'text': text, 'source': 'spanning'},
                        score,
                        match_type
                    ))
            
            # Strategy 3: Preprocessed text matching
            preprocessed = preprocess_text(all_text)
            if preprocessed and preprocessed != all_text:
                score, algo = calculate_multi_algorithm_score(phrase, preprocessed)
                if score >= threshold:
                    new_matches.append((
                        {'text': preprocessed, 'source': 'preprocessed'},
                        score,
                        f'preprocessed_{algo}'
                    ))
            
            if new_matches:
                # Deduplicate and sort by score
                seen_texts = set()
                unique_matches = []
                for match_data, score, match_type in sorted(new_matches, key=lambda x: x[1], reverse=True):
                    text = match_data.get('text', '')
                    if text not in seen_texts:
                        unique_matches.append((match_data, score, match_type))
                        seen_texts.add(text)
                
                if phrase in enhanced_matches:
                    enhanced_matches[phrase].extend(unique_matches)
                else:
                    enhanced_matches[phrase] = unique_matches
        
        results['matches'] = enhanced_matches
        results['total_matches'] = sum(len(matches) for matches in enhanced_matches.values())
        
        return results
    
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
        Calculate similarity between phrase and text using advanced fuzzy matching.
        """
        if not self.ocr_available or not FUZZY_AVAILABLE:
            return 0.0
        
        score, _ = calculate_multi_algorithm_score(phrase, text)
        return score
    
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
