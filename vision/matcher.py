"""Phrase matching utilities with fuzzy matching support."""

from typing import List, Dict, Tuple, Optional

# Changed to absolute imports
from config.vision_config import VisionConfig
from utils.text_utils import normalize_text_for_search, is_meaningful_phrase

try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


class PhraseMatcher:
    """Matches phrases in text using fuzzy matching."""
    
    def __init__(self, config: VisionConfig):
        self.config = config
    
    def find_matches(self, phrase: str, text_lines: List[Dict], 
                    full_text: str, threshold: int = 85) -> List[Tuple[Dict, float, str]]:
        """Find complete phrase matches in text."""
        if not is_meaningful_phrase(phrase, self.config.common_words):
            return []
        
        phrase_normalized = normalize_text_for_search(phrase)
        matches = []
        
        # Search in lines
        matches.extend(self._search_in_lines(phrase, phrase_normalized, text_lines, threshold))
        
        # Search spanning lines
        if len(phrase_normalized.split()) > 1:
            matches.extend(self._search_spanning(phrase, phrase_normalized, text_lines, threshold))
        
        return self._deduplicate_matches(matches)
    
    def _search_in_lines(self, phrase: str, phrase_normalized: str,
                        text_lines: List[Dict], threshold: int) -> List:
        """Search for phrase in individual lines."""
        matches = []
        phrase_words = phrase_normalized.split()
        
        for line in text_lines:
            line_normalized = normalize_text_for_search(line['text'])
            
            # Exact match
            if phrase_normalized in line_normalized:
                enhanced = self._enhance_match_boundaries(line, phrase, phrase_normalized, line_normalized)
                matches.append((enhanced, 100, "complete_phrase"))
                continue
            
            # Fuzzy match
            if FUZZY_AVAILABLE and len(line_normalized) >= len(phrase_normalized):
                is_common = all(w in self.config.common_words for w in phrase_words)
                
                if not is_common:
                    similarity = self._calculate_similarity(phrase_normalized, line_normalized)
                    if similarity >= threshold:
                        match_type = "upside_down" if similarity > 95 else "fuzzy_phrase"
                        matches.append((line.copy(), similarity, match_type))
        
        return matches
    
    def _calculate_similarity(self, phrase: str, text: str) -> float:
        """Calculate text similarity score."""
        if not FUZZY_AVAILABLE:
            return 0
        
        normal_score = fuzz.token_set_ratio(phrase, text)
        reverse_score = fuzz.token_set_ratio(phrase, ' '.join(text.split()[::-1]))
        char_reverse_score = fuzz.token_set_ratio(phrase, text[::-1])
        
        return max(normal_score, reverse_score, char_reverse_score)
    
    def _search_spanning(self, phrase: str, phrase_normalized: str,
                        text_lines: List[Dict], threshold: int) -> List:
        """Search for phrases spanning multiple lines."""
        matches = []
        phrase_words = phrase_normalized.split()
        
        for i, current_line in enumerate(text_lines):
            current_normalized = normalize_text_for_search(current_line['text'])
            current_words = current_normalized.split()
            
            phrase_matches = self._find_word_matches(phrase_words, current_words)
            if not phrase_matches:
                continue
            
            # Check next lines
            for offset in range(1, min(4, len(text_lines) - i)):
                next_line = text_lines[i + offset]
                
                if not self._lines_compatible(current_line, next_line):
                    continue
                
                spanning = self._try_spanning_match(
                    phrase_words, phrase_matches, next_line, 
                    current_line, i, offset, threshold
                )
                
                if spanning:
                    matches.append(spanning)
                    break
            
            # Check previous lines
            for offset in range(1, min(3, i + 1)):
                prev_line = text_lines[i - offset]
                
                if not self._lines_compatible(prev_line, current_line):
                    continue
                
                spanning = self._try_spanning_match_reverse(
                    phrase_words, phrase_matches, prev_line,
                    current_line, i, offset, threshold
                )
                
                if spanning:
                    matches.append(spanning)
                    break
        
        return matches
    
    def _find_word_matches(self, phrase_words: List[str], text_words: List[str]) -> List:
        """Find matching words between phrase and text."""
        matches = []
        for pw in phrase_words:
            for j, tw in enumerate(text_words):
                if pw == tw or (FUZZY_AVAILABLE and fuzz.ratio(pw, tw) > 85):
                    matches.append((pw, j, tw))
        return matches
    
    def _lines_compatible(self, line1: Dict, line2: Dict) -> bool:
        """Check if two lines are compatible for spanning match."""
        y_distance = abs(line2['y_position'] - line1['y_position'])
        angle_diff = abs(line2.get('angle', 0) - line1.get('angle', 0))
        return y_distance <= 100 and angle_diff <= 30
    
    def _try_spanning_match(self, phrase_words: List[str], current_matches: List,
                           next_line: Dict, current_line: Dict, 
                           line_idx: int, offset: int, threshold: int) -> Optional[Tuple]:
        """Try to create a spanning match with next line."""
        next_normalized = normalize_text_for_search(next_line['text'])
        remaining = [pw for pw in phrase_words if not any(pw == m[0] for m in current_matches)]
        
        next_matches = self._find_word_matches(remaining, next_normalized.split())
        if not next_matches:
            return None
        
        total_matched = len(current_matches) + len(next_matches)
        match_pct = min((total_matched / len(phrase_words)) * 100, 100)
        
        if match_pct >= 70:
            combined = {
                'text': f"{current_line['text']} {next_line['text']}",
                'annotations': current_line.get('annotations', []) + next_line.get('annotations', []),
                'y_position': current_line['y_position'],
                'angle': current_line.get('angle', 0),
                'span_info': {
                    'line_indices': [line_idx, line_idx + offset],
                    'matched_words': current_matches + next_matches,
                    'total_lines': offset + 1
                }
            }
            
            # Filter annotations for exact matches
            if match_pct == 100:
                combined['annotations'] = self._filter_matched_annotations(
                    combined['annotations'], current_matches + next_matches
                )
            
            match_type = "exact_spanning" if match_pct == 100 else "fuzzy_spanning"
            return (combined, match_pct, match_type)
        
        return None
    
    def _try_spanning_match_reverse(self, phrase_words: List[str], current_matches: List,
                                   prev_line: Dict, current_line: Dict,
                                   line_idx: int, offset: int, threshold: int) -> Optional[Tuple]:
        """Try to create a spanning match with previous line."""
        prev_normalized = normalize_text_for_search(prev_line['text'])
        remaining = [pw for pw in phrase_words if not any(pw == m[0] for m in current_matches)]
        
        prev_matches = self._find_word_matches(remaining, prev_normalized.split())
        if not prev_matches:
            return None
        
        total_matched = len(current_matches) + len(prev_matches)
        match_pct = min((total_matched / len(phrase_words)) * 100, 100)
        
        if match_pct >= 70:
            combined = {
                'text': f"{prev_line['text']} {current_line['text']}",
                'annotations': prev_line.get('annotations', []) + current_line.get('annotations', []),
                'y_position': prev_line['y_position'],
                'angle': prev_line.get('angle', 0),
                'span_info': {
                    'line_indices': [line_idx - offset, line_idx],
                    'matched_words': prev_matches + current_matches,
                    'total_lines': offset + 1
                }
            }
            
            if match_pct == 100:
                combined['annotations'] = self._filter_matched_annotations(
                    combined['annotations'], prev_matches + current_matches
                )
            
            match_type = "exact_spanning" if match_pct == 100 else "fuzzy_spanning"
            return (combined, match_pct, match_type)
        
        return None
    
    def _filter_matched_annotations(self, annotations, matched_words):
        """Filter annotations to only those matching the phrase words."""
        matched_texts = [m[2] for m in matched_words]
        filtered = [a for a in annotations if a.description.lower() in matched_texts]
        return filtered if filtered else annotations
    
    def _enhance_match_boundaries(self, line, phrase, phrase_normalized, line_normalized):
        """Enhance matched line with precise word boundary information."""
        enhanced = line.copy()
        
        if 'annotations' not in line or not line['annotations']:
            return enhanced
        
        phrase_words = phrase_normalized.split()
        line_words = line_normalized.split()
        
        # Find phrase start position
        phrase_start_idx = None
        for i in range(len(line_words) - len(phrase_words) + 1):
            if line_words[i:i+len(phrase_words)] == phrase_words:
                phrase_start_idx = i
                break
        
        if phrase_start_idx is not None:
            phrase_annotations = []
            word_count = 0
            
            for annotation in line['annotations']:
                if phrase_start_idx <= word_count < phrase_start_idx + len(phrase_words):
                    phrase_annotations.append(annotation)
                word_count += len(annotation.description.split())
                if word_count > phrase_start_idx + len(phrase_words):
                    break
            
            if phrase_annotations:
                enhanced['annotations'] = phrase_annotations
        
        return enhanced
    
    def _deduplicate_matches(self, matches: List) -> List:
        """Remove duplicate matches."""
        seen = set()
        unique = []
        
        for match in sorted(matches, key=lambda x: x[1], reverse=True):
            match_text = match[0].get('text', str(match[0]))
            
            if 'span_info' in match[0]:
                key = f"{normalize_text_for_search(match_text)}_span_{match[0]['span_info']['line_indices']}"
            else:
                key = normalize_text_for_search(match_text)
            
            if key not in seen:
                seen.add(key)
                unique.append(match)
        
        return unique
