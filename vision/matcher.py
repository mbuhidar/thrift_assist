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
        
        # Search in lines first
        line_matches = self._search_in_lines(phrase, phrase_normalized, text_lines, threshold)
        matches.extend(line_matches)
        
        # Only search spanning if no complete single-line match found
        if len(phrase_normalized.split()) > 1:
            has_complete = any(score == 100 and 'span_info' not in match_data 
                             for match_data, score, _ in line_matches)
            
            if not has_complete:
                matches.extend(self._search_spanning(phrase, phrase_normalized, text_lines, threshold))
        
        return self._deduplicate_matches(matches)
    
    def _search_in_lines(self, phrase: str, phrase_normalized: str,
                        text_lines: List[Dict], threshold: int) -> List:
        """Search for phrase in individual lines."""
        matches = []
        phrase_words = phrase_normalized.split()
        
        for line in text_lines:
            line_normalized = normalize_text_for_search(line['text'])
            
            # Check for phrase match with possessives handled
            if self._phrase_exists_in_line(phrase_normalized, line_normalized):
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
                        enhanced = self._enhance_match_boundaries(line, phrase, phrase_normalized, line_normalized)
                        matches.append((enhanced, similarity, match_type))
        
        return matches
    
    def _phrase_exists_in_line(self, phrase_normalized: str, line_normalized: str) -> bool:
        """Check if phrase exists in line, handling possessives."""
        # Direct match
        if phrase_normalized in line_normalized:
            return True
        
        # Check with possessives
        phrase_words = phrase_normalized.split()
        line_words = line_normalized.split()
        
        for i in range(len(line_words) - len(phrase_words) + 1):
            if self._words_sequence_match(phrase_words, line_words[i:i+len(phrase_words)]):
                return True
        
        return False
    
    def _words_sequence_match(self, phrase_words: List[str], line_words: List[str]) -> bool:
        """Check if phrase words match line words sequence with possessives."""
        if len(phrase_words) != len(line_words):
            return False
        
        for pw, lw in zip(phrase_words, line_words):
            if not self._word_matches(pw, lw):
                return False
        return True
    
    def _word_matches(self, word1: str, word2: str) -> bool:
        """Check if two words match, handling possessives."""
        if word1 == word2:
            return True
        # Strip possessives
        w1 = word1.rstrip("'s").rstrip("'")
        w2 = word2.rstrip("'s").rstrip("'")
        if w1 == w2:
            return True
        # Prefix matching (short differences)
        if (w2.startswith(w1) and len(w2) - len(w1) <= 2) or \
           (w1.startswith(w2) and len(w1) - len(w2) <= 2):
            return True
        return False
    
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
            
            # Skip if this line already has all the phrase words
            if len(phrase_matches) >= len(phrase_words):
                continue
            
            # Check next lines (increased back to 3)
            for offset in range(1, min(3, len(text_lines) - i)):
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
            for offset in range(1, min(2, i + 1)):
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
        matched_indices = set()  # Track which text word indices we've already matched
        
        for pw in phrase_words:
            for j, tw in enumerate(text_words):
                # Skip if this text word already matched
                if j in matched_indices:
                    continue
                    
                if self._word_matches(pw, tw):
                    matches.append((pw, j, tw))
                    matched_indices.add(j)
                    break  # Move to next phrase word
                elif FUZZY_AVAILABLE and fuzz.ratio(pw, tw) > 85:
                    matches.append((pw, j, tw))
                    matched_indices.add(j)
                    break  # Move to next phrase word
        
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
        
        # Lowered threshold back to 70%
        if match_pct >= 70:
            # Store the original phrase words for display
            phrase_text = ' '.join([m[0] for m in current_matches + next_matches])
            
            combined = {
                'text': phrase_text,  # Use matched phrase words, not full line text
                'original_text': f"{current_line['text']} {next_line['text']}",  # Keep original for reference
                'annotations': current_line.get('annotations', []) + next_line.get('annotations', []),
                'y_position': current_line['y_position'],
                'angle': current_line.get('angle', 0),
                'span_info': {
                    'line_indices': [line_idx, line_idx + offset],
                    'matched_words': current_matches + next_matches,
                    'total_lines': offset + 1
                }
            }
            
            # Always filter annotations to get tighter bounds
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
        
        # Lowered threshold back to 70%
        if match_pct >= 70:
            # Store the original phrase words for display (in correct order)
            phrase_text = ' '.join([m[0] for m in prev_matches + current_matches])
            
            combined = {
                'text': phrase_text,  # Use matched phrase words, not full line text
                'original_text': f"{prev_line['text']} {current_line['text']}",  # Keep original for reference
                'annotations': prev_line.get('annotations', []) + current_line.get('annotations', []),
                'y_position': prev_line['y_position'],
                'angle': prev_line.get('angle', 0),
                'span_info': {
                    'line_indices': [line_idx - offset, line_idx],
                    'matched_words': prev_matches + current_matches,
                    'total_lines': offset + 1
                }
            }
            
            # Always filter annotations to get tighter bounds
            combined['annotations'] = self._filter_matched_annotations(
                combined['annotations'], prev_matches + current_matches
            )
            
            match_type = "exact_spanning" if match_pct == 100 else "fuzzy_spanning"
            return (combined, match_pct, match_type)
        
        return None
    
    def _filter_matched_annotations(self, annotations, matched_words):
        """Filter annotations to only those matching the phrase words."""
        if not annotations or not matched_words:
            return annotations
        
        matched_texts = set(m[2] for m in matched_words)  # Use set for faster lookup
        filtered = []
        
        # First pass: exact matches only
        for a in annotations:
            try:
                ann_desc = normalize_text_for_search(a.description)
                # Must exactly match one of the matched word texts
                if ann_desc in matched_texts:
                    filtered.append(a)
                    continue
                
                # Or match with possessive handling
                for mt in matched_texts:
                    if self._word_matches(ann_desc, mt):
                        filtered.append(a)
                        break
            except (AttributeError, TypeError):
                continue
        
        # If we got good matches, return them
        if len(filtered) >= len(matched_words) * 0.7:  # Have at least 70% of expected words
            return filtered
        
        # Second pass: slightly more lenient but still strict
        if not filtered or len(filtered) < len(matched_words) * 0.5:
            for a in annotations:
                try:
                    ann_desc = normalize_text_for_search(a.description)
                    # Check if annotation is a close substring match
                    for mt in matched_texts:
                        if len(ann_desc) >= 3 and len(mt) >= 3:  # Minimum length check
                            if (ann_desc in mt and len(ann_desc) / len(mt) > 0.7) or \
                               (mt in ann_desc and len(mt) / len(ann_desc) > 0.7):
                                filtered.append(a)
                                break
                except (AttributeError, TypeError):
                    continue
        
        # Return filtered or all annotations as last resort
        return filtered if filtered else annotations
    
    def _enhance_match_boundaries(self, line, phrase, phrase_normalized, line_normalized):
        """Enhance matched line with precise word boundary information."""
        enhanced = line.copy()
        
        if 'annotations' not in line or not line['annotations']:
            return enhanced
        
        phrase_words = phrase_normalized.split()
        line_words = line_normalized.split()
        
        # Find phrase start position with possessive handling
        phrase_start_idx = None
        for i in range(len(line_words) - len(phrase_words) + 1):
            if self._words_sequence_match(phrase_words, line_words[i:i+len(phrase_words)]):
                phrase_start_idx = i
                break
        
        if phrase_start_idx is not None:
            phrase_annotations = []
            word_count = 0
            
            for annotation in line['annotations']:
                try:
                    # Check if annotation is within phrase range
                    if phrase_start_idx <= word_count < phrase_start_idx + len(phrase_words):
                        phrase_annotations.append(annotation)
                    
                    word_count += len(annotation.description.split())
                    
                    if word_count > phrase_start_idx + len(phrase_words):
                        break
                except (AttributeError, TypeError):
                    continue
            
            if phrase_annotations:
                enhanced['annotations'] = phrase_annotations
                
                # Recalculate angle from the filtered phrase annotations
                # The line angle might not match the actual phrase angle
                from utils.geometry_utils import calculate_text_angle
                if phrase_annotations and hasattr(phrase_annotations[0], 'bounding_poly'):
                    # If we have multiple annotations, calculate from first one's vertices
                    # (they should all have similar angles since they're on same line)
                    try:
                        phrase_angle = calculate_text_angle(
                            phrase_annotations[0].bounding_poly.vertices
                        )
                        enhanced['angle'] = phrase_angle
                    except (AttributeError, TypeError):
                        # Keep original line angle if calculation fails
                        pass
        
        return enhanced
    
    def _deduplicate_matches(self, matches: List) -> List:
        """Remove duplicate matches, preferring single-line and higher-scoring matches."""
        seen = set()
        unique = []
        
        # Sort by: score (desc), then line count (asc), then is_spanning (False first)
        def sort_key(match):
            match_data, score, match_type = match
            line_count = match_data.get('span_info', {}).get('total_lines', 1)
            is_spanning = 'span_info' in match_data
            # Prefer: higher score, fewer lines, non-spanning
            return (-score, line_count, is_spanning)
        
        sorted_matches = sorted(matches, key=sort_key)
        
        for match in sorted_matches:
            match_data, score, match_type = match
            match_text_norm = normalize_text_for_search(match_data.get('text', ''))
            
            # Create key for this match
            if 'span_info' in match_data:
                key = f"{match_text_norm}_span_{match_data['span_info']['line_indices']}"
            else:
                key = match_text_norm
            
            # Check if we already have a better match for this text
            already_matched = False
            for existing_key in seen:
                existing_text = existing_key.split('_span_')[0]
                # If same text already matched with better score/fewer lines, skip
                if match_text_norm == existing_text:
                    already_matched = True
                    break
            
            if not already_matched and key not in seen:
                seen.add(key)
                unique.append(match)
        
        return unique
