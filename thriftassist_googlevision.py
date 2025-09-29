"""
OCR Phrase Detection and Annotation
Combines phrase detection with visual annotation like the other OCR scripts.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from contextlib import redirect_stderr
from io import StringIO
import re
from collections import Counter

# Suppress warnings
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GRPC_TRACE'] = ''
os.environ['GOOGLE_CLOUD_DISABLE_GRPC_FOR_GAE'] = 'true'
os.environ['GLOG_minloglevel'] = '3'
warnings.filterwarnings("ignore")

# Set credentials path directly in the script (optional)
credentials_path = "credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json"
if os.path.exists(credentials_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

'''
# Configuration
IMAGE_PATH = "image/iCloud_Photos/IMG_4918.JPEG"

SEARCH_TERMS = [
    'Homecoming',
    'Circle of Three',
    'Lee Child',
    'Homegoing',
    'Beginnings',
    'Tom Clancy',
    ]
'''

# Configuration
IMAGE_PATH = "image/iCloud_Photos/IMG_4970.JPEG"

SEARCH_TERMS = [
    'Billy Joel',
    'Jewel',
    'U2',
    'Keagey',
    ]

FUZZ_THRESHOLD = 75  # Similarity threshold for phrase matching

# Common words to de-emphasize when searching (only filter if appear alone)
COMMON_WORDS = {
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'among', 'a', 'an', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
    'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
    'its', 'our', 'their'
}
"""
Google Cloud Vision API with fuzzy matching to detect and annotate
phrases in images with visual annotation like the other OCR scripts.
"""


def suppress_stderr_warnings():
    """Context manager to suppress stderr warnings during client creation."""
    class FilteredStringIO(StringIO):
        def write(self, s):
            alts_warning = 'ALTS creds ignored' not in s
            log_warning = 'absl::InitializeLog' not in s
            if alts_warning and log_warning:
                return super().write(s)
            return len(s)
    
    return redirect_stderr(FilteredStringIO())


# Import with warnings suppressed
with suppress_stderr_warnings():
    from google.cloud import vision


# Configuration
FUZZ_THRESHOLD = 50 # Similarity threshold for phrase matching


def calculate_text_angle(vertices):
    """
    Calculate the angle of text based on bounding box vertices.
    
    Args:
        vertices: List of bounding box vertices
        
    Returns:
        Angle in degrees (0 = horizontal, positive = clockwise)
    """

    if len(vertices) < 2:
        return 0
    
    # Calculate angle from first two vertices
    p1, p2 = vertices[0], vertices[1]
    import math
    angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
    return math.degrees(angle)


def group_text_into_lines(text_annotations):
    """
    Group individual text annotations into text lines based on proximity and angle.
    Handles text at various orientations including upside down.
    
    Args:
        text_annotations: Google Vision API text annotations
        
    Returns:
        List of text lines with combined text and bounding boxes
    """

    if len(text_annotations) <= 1:
        return []
    
    # Skip the first annotation (full text) and process word-level annotations
    word_annotations = text_annotations[1:]
    
    # Group words by their angle and position
    angle_groups = {}
    angle_tolerance = 15  # degrees - allow some variation in angle
    
    for annotation in word_annotations:
        if not annotation.bounding_poly.vertices:
            continue
            
        # Calculate text angle
        angle = calculate_text_angle(annotation.bounding_poly.vertices)
        
        # Normalize angle to handle rotations
        normalized_angle = angle % 360
        if normalized_angle > 180:
            normalized_angle -= 360
            
        # Find existing angle group within tolerance
        angle_key = None
        for existing_angle in angle_groups.keys():
            if abs(normalized_angle - existing_angle) <= angle_tolerance:
                angle_key = existing_angle
                break
        
        if angle_key is None:
            angle_key = normalized_angle
            angle_groups[angle_key] = []
        
        angle_groups[angle_key].append(annotation)
    
    # Process each angle group separately
    all_text_lines = []
    
    for angle, annotations_in_angle in angle_groups.items():
        # Group words by their position (considering rotation)
        lines = {}
        tolerance = 20  # pixels - increased for rotated text
        
        for annotation in annotations_in_angle:
            # For rotated text, use different positioning logic
            if abs(angle) < 45:  # Nearly horizontal text
                y_pos = annotation.bounding_poly.vertices[0].y
                x_pos = annotation.bounding_poly.vertices[0].x
                sort_key = x_pos
            elif abs(angle - 90) < 45 or abs(angle + 90) < 45:  # Nearly vertical
                x_pos = annotation.bounding_poly.vertices[0].x
                y_pos = annotation.bounding_poly.vertices[0].y
                sort_key = -y_pos  # Reverse for top-to-bottom reading
            else:  # Diagonal text
                # Use center point for diagonal text
                center_x = sum(v.x for v in annotation.bounding_poly.vertices) / 4
                center_y = sum(v.y for v in annotation.bounding_poly.vertices) / 4
                y_pos = center_y
                sort_key = center_x
            
            # Find existing line within tolerance
            line_key = None
            for existing_pos in lines.keys():
                if abs(y_pos - existing_pos) <= tolerance:
                    line_key = existing_pos
                    break
            
            if line_key is None:
                line_key = y_pos
                lines[line_key] = []
            
            lines[line_key].append((sort_key, annotation.description, annotation))
        
        # Convert angle group to sorted lines
        for line_pos in sorted(lines.keys()):
            # Sort words in each line by position
            line_words = sorted(lines[line_pos], key=lambda x: x[0])
            
            # Combine words into line text
            line_text = ' '.join([word[1] for word in line_words]).strip()
            line_annotations = [word[2] for word in line_words]
            
            if line_text:  # Only add non-empty lines
                all_text_lines.append({
                    'text': line_text,
                    'annotations': line_annotations,
                    'y_position': line_pos,
                    'angle': angle
                })
    
    return all_text_lines


def is_meaningful_phrase(phrase):
    """
    Check if a phrase is meaningful (not just common words).
    Returns True if the phrase should be searched for.
    """
    words = phrase.lower().strip().split()
    
    # Always allow single words that aren't common
    if len(words) == 1:
        return words[0] not in COMMON_WORDS
    
    # For multi-word phrases, require at least one non-common word
    # OR allow phrases where common words appear multiple times (like "The The")
    non_common_words = [w for w in words if w not in COMMON_WORDS]
    
    # Allow if there are non-common words
    if non_common_words:
        return True
    
    # Allow if common words are repeated (like "The The", "All All")
    if len(set(words)) < len(words):
        return True
    
    # Filter out phrases that are only common words
    return False


def normalize_text_for_search(text):
    """
    Normalize text for better matching across different orientations.
    Handles OCR artifacts from rotated text.
    """
    # Remove extra spaces and normalize whitespace
    normalized = re.sub(r'\s+', ' ', text.strip())
    # Handle common OCR errors in rotated and upside-down text
    char_map = {
        '|': 'I', '¡': 'i', '1': 'l', '0': 'O', '5': 'S',
        'б': '6', 'g': '9', 'q': 'p', 'd': 'b', 'u': 'n',
        '∩': 'n', 'ᴍ': 'w', 'ɹ': 'r', 'ɐ': 'a', 'ǝ': 'e'
    }
    for old_char, new_char in char_map.items():
        normalized = normalized.replace(old_char, new_char)
    return normalized.lower()


def try_reverse_text_matching(phrase, text):
    """
    Try matching text that might be upside down or reversed.
    Returns similarity score if a good match is found.
    Enhanced to be more precise for exact phrase matching.
    """

    if not FUZZY_AVAILABLE:
        return 0
    
    phrase_normalized = normalize_text_for_search(phrase)
    text_normalized = normalize_text_for_search(text)
    
    # For common word phrases, use substring matching first
    phrase_words = phrase_normalized.split()
    is_common_word_phrase = all(word in COMMON_WORDS for word in phrase_words)
    
    if is_common_word_phrase:
        # For phrases like "The The", check if the exact phrase appears
        if phrase_normalized in text_normalized:
            return 100
        # Check reversed word order for upside-down text
        text_words_reversed = ' '.join(text_normalized.split()[::-1])
        if phrase_normalized in text_words_reversed:
            return 100
        # Use very strict token_set_ratio only if substring match fails
        token_score = fuzz.token_set_ratio(phrase_normalized, text_normalized)
        return token_score if token_score >= 98 else 0
    
    # For non-common word phrases, use original fuzzy matching
    # Try normal matching first
    normal_score = fuzz.token_set_ratio(phrase_normalized, text_normalized)
    
    # Try word-reversed matching (for upside-down text)
    text_words_reversed = ' '.join(text_normalized.split()[::-1])
    reverse_score = fuzz.token_set_ratio(phrase_normalized,
                                         text_words_reversed)
    
    # Try character-reversed matching (for completely flipped text)
    text_char_reversed = text_normalized[::-1]
    char_reverse_score = fuzz.token_set_ratio(phrase_normalized,
                                              text_char_reversed)
    
    return max(normal_score, reverse_score, char_reverse_score)


def find_complete_phrases(phrase, text_lines, full_text, threshold=85):
    """
    Find complete phrase matches in text lines and full text.
    Only returns matches for complete phrases, not individual words.
    Enhanced to handle text at various angles including upside down and phrases spanning multiple lines.
    
    Args:
        phrase: Target phrase to find
        text_lines: List of text lines from group_text_into_lines()
        full_text: Complete text from first annotation
        threshold: Minimum similarity for fuzzy matching
    
    Returns:
        List of matching text segments with scores
    """
    matches = []
    phrase_normalized = normalize_text_for_search(phrase)
    phrase_words = phrase_normalized.split()
    
    # Allow single words if they are meaningful (not common words)
    if len(phrase_words) == 1:
        if not is_meaningful_phrase(phrase):
            return matches
    
    # Search in individual text lines for complete phrases
    for line in text_lines:
        line_text_normalized = normalize_text_for_search(line['text'])
        
        # Exact substring match (highest priority)
        if phrase_normalized in line_text_normalized:
            matches.append((line, 100, "complete_phrase"))
            continue
        
        # High-threshold fuzzy matching for complete phrases only
        if FUZZY_AVAILABLE and len(line_text_normalized) >= len(phrase_normalized):
            # Check if this is a phrase made entirely of common words
            is_common_word_phrase = all(word in COMMON_WORDS
                                        for word in phrase_words)
            
            if is_common_word_phrase:
                # For common word phrases like "The The", ONLY use exact matching
                # No fuzzy matching to avoid false positives
                continue
            else:
                # For phrases with non-common words, use fuzzy matching
                similarity = try_reverse_text_matching(phrase, line['text'])
                if similarity >= threshold:
                    match_type = "upside_down" if similarity > 95 else "fuzzy_phrase"
                    matches.append((line, similarity, match_type))
                    continue
                
                # Also try partial ratio for phrases that span multiple words
                partial_sim = fuzz.partial_ratio(phrase_normalized,
                                                 line_text_normalized)
                if partial_sim >= 90:  # High threshold for partial matches
                    matches.append((line, partial_sim, "partial_phrase"))
    
    # NEW: Search for phrases that span multiple lines
    if len(phrase_words) > 1 and len(text_lines) > 1:
        for i in range(len(text_lines)):
            current_line = text_lines[i]
            current_text_normalized = normalize_text_for_search(current_line['text'])
            
            # Check if current line contains any words from our phrase
            current_words = current_text_normalized.split()
            phrase_word_matches = []
            
            for pw in phrase_words:
                for j, cw in enumerate(current_words):
                    if pw == cw or (FUZZY_AVAILABLE and fuzz.ratio(pw, cw) > 85):
                        phrase_word_matches.append((pw, j, cw))
            
            # If current line has some phrase words, look in nearby lines for the rest
            if phrase_word_matches:
                # Check next few lines (look ahead up to 3 lines)
                for next_offset in range(1, min(4, len(text_lines) - i)):
                    next_line = text_lines[i + next_offset]
                    
                    # Skip if lines are too far apart vertically (different sections)
                    y_distance = abs(next_line['y_position'] - current_line['y_position'])
                    if y_distance > 100:  # Adjust this threshold as needed
                        continue
                    
                    # Skip if angles are very different (different orientations)
                    angle_diff = abs(next_line.get('angle', 0) - current_line.get('angle', 0))
                    if angle_diff > 30:  # Allow some angle variation
                        continue
                    
                    next_text_normalized = normalize_text_for_search(next_line['text'])
                    next_words = next_text_normalized.split()
                    
                    # Check if next line contains remaining phrase words
                    remaining_phrase_words = [pw for pw in phrase_words 
                                            if not any(pw == match[0] for match in phrase_word_matches)]
                    
                    next_line_matches = []
                    for pw in remaining_phrase_words:
                        for j, nw in enumerate(next_words):
                            if pw == nw or (FUZZY_AVAILABLE and fuzz.ratio(pw, nw) > 85):
                                next_line_matches.append((pw, j, nw))
                    
                    # If we found matches for remaining words, create a spanning match
                    if next_line_matches:
                        total_matched_words = len(phrase_word_matches) + len(next_line_matches)
                        match_percentage = (total_matched_words / len(phrase_words)) * 100
                        
                        if match_percentage >= 70:  # At least 70% of phrase words found
                            # Create combined line data
                            combined_text = current_line['text'] + ' ' + next_line['text']
                            combined_annotations = current_line.get('annotations', []) + next_line.get('annotations', [])
                            
                            spanning_match = {
                                'text': combined_text,
                                'annotations': combined_annotations,
                                'y_position': current_line['y_position'],
                                'angle': current_line.get('angle', 0),
                                'span_info': {
                                    'line_indices': [i, i + next_offset],
                                    'matched_words': phrase_word_matches + next_line_matches,
                                    'total_lines': next_offset + 1
                                }
                            }
                            
                            match_type = "exact_spanning" if match_percentage >= 95 else "fuzzy_spanning"
                            matches.append((spanning_match, match_percentage, match_type))
                            
                            # Break after finding first spanning match for this line
                            break
                
                # Also check previous lines (look back up to 2 lines)
                for prev_offset in range(1, min(3, i + 1)):
                    prev_line = text_lines[i - prev_offset]
                    
                    # Skip if lines are too far apart vertically
                    y_distance = abs(prev_line['y_position'] - current_line['y_position'])
                    if y_distance > 100:
                        continue
                    
                    # Skip if angles are very different
                    angle_diff = abs(prev_line.get('angle', 0) - current_line.get('angle', 0))
                    if angle_diff > 30:
                        continue
                    
                    prev_text_normalized = normalize_text_for_search(prev_line['text'])
                    prev_words = prev_text_normalized.split()
                    
                    # Check if previous line contains remaining phrase words
                    remaining_phrase_words = [pw for pw in phrase_words 
                                            if not any(pw == match[0] for match in phrase_word_matches)]
                    
                    prev_line_matches = []
                    for pw in remaining_phrase_words:
                        for j, pw_word in enumerate(prev_words):
                            if pw == pw_word or (FUZZY_AVAILABLE and fuzz.ratio(pw, pw_word) > 85):
                                prev_line_matches.append((pw, j, pw_word))
                    
                    if prev_line_matches:
                        total_matched_words = len(phrase_word_matches) + len(prev_line_matches)
                        match_percentage = (total_matched_words / len(phrase_words)) * 100
                        
                        if match_percentage >= 70:
                            # Create combined line data (previous line first)
                            combined_text = prev_line['text'] + ' ' + current_line['text']
                            combined_annotations = prev_line.get('annotations', []) + current_line.get('annotations', [])
                            
                            spanning_match = {
                                'text': combined_text,
                                'annotations': combined_annotations,
                                'y_position': prev_line['y_position'],
                                'angle': prev_line.get('angle', 0),
                                'span_info': {
                                    'line_indices': [i - prev_offset, i],
                                    'matched_words': prev_line_matches + phrase_word_matches,
                                    'total_lines': prev_offset + 1
                                }
                            }
                            
                            match_type = "exact_spanning" if match_percentage >= 95 else "fuzzy_spanning"
                            matches.append((spanning_match, match_percentage, match_type))
                            break
    
    # Remove duplicates and sort by score
    seen_texts = set()
    unique_matches = []
    for match in sorted(matches, key=lambda x: x[1], reverse=True):
        match_text = match[0]['text'] if 'text' in match[0] else str(match[0])
        # Create a unique key that includes span info if available
        if 'span_info' in match[0]:
            match_key = f"{normalize_text_for_search(match_text)}_span_{match[0]['span_info']['line_indices']}"
        else:
            match_key = normalize_text_for_search(match_text)
        
        if match_key not in seen_texts:
            seen_texts.add(match_key)
            unique_matches.append(match)
    
    return unique_matches


def draw_phrase_annotations(image, phrase_matches, phrase_colors=None):
    """
    Draw bounding boxes around detected complete phrases.
    Enhanced to align bounding boxes with text direction and orientation.
    
    Args:
        image: Input image (BGR format)
        phrase_matches: Dictionary of phrase -> list of matches
        phrase_colors: Dictionary of phrase -> color (BGR)
    
    Returns:
        Annotated image
    """
    if phrase_colors is None:
        # All phrases use green color
        green_color = (0, 255, 0)  # Green in BGR format
        phrase_colors = {}
        for phrase in phrase_matches.keys():
            phrase_colors[phrase] = green_color
    
    annotated = image.copy()
    
    for phrase, matches in phrase_matches.items():
        color = phrase_colors.get(phrase, (0, 255, 0))
        
        for match_data, score, match_type in matches:
            # Get annotations for this line/phrase
            annotations = match_data.get('annotations', [])
            if not annotations:
                continue
            
            # Get the text angle from the match data
            text_angle = match_data.get('angle', 0)
            
            # Find the specific words in line that match the phrase
            line_text_normalized = normalize_text_for_search(match_data.get('text', ''))
            phrase_normalized = normalize_text_for_search(phrase)
            phrase_words = phrase_normalized.split()
            
            # Calculate precise bounding box for ONLY the phrase words
            phrase_annotations = []
            
            if phrase_normalized in line_text_normalized:
                # Find exact phrase match - get only the words that are part of the phrase
                phrase_start_idx = line_text_normalized.find(phrase_normalized)
                
                # Split the line into words and find which words correspond to the phrase
                line_words = line_text_normalized.split()
                phrase_word_list = phrase_normalized.split()
                
                # Find the starting word index
                for start_idx in range(len(line_words) - len(phrase_word_list) + 1):
                    window = ' '.join(line_words[start_idx:start_idx + len(phrase_word_list)])
                    if window == phrase_normalized:
                        # Found the exact phrase - get corresponding annotations
                        phrase_annotations = annotations[start_idx:start_idx + len(phrase_word_list)]
                        break
                
                # Fallback: if we couldn't match exactly, try fuzzy word matching
                if not phrase_annotations:
                    for phrase_word in phrase_word_list:
                        for i, annotation in enumerate(annotations):
                            word_normalized = normalize_text_for_search(annotation.description)
                            if (phrase_word == word_normalized or 
                                (FUZZY_AVAILABLE and fuzz.ratio(phrase_word, word_normalized) > 85)):
                                if annotation not in phrase_annotations:
                                    phrase_annotations.append(annotation)
            else:
                # For fuzzy matches, try to match individual words more precisely
                line_words = [normalize_text_for_search(ann.description) for ann in annotations]
                
                # Try to find consecutive words that match the phrase
                for start_idx in range(len(line_words) - len(phrase_words) + 1):
                    match_count = 0
                    temp_annotations = []
                    
                    for i, phrase_word in enumerate(phrase_words):
                        if start_idx + i < len(line_words):
                            line_word = line_words[start_idx + i]
                            if (phrase_word in line_word or line_word in phrase_word or
                                (FUZZY_AVAILABLE and fuzz.ratio(phrase_word, line_word) > 75)):
                                match_count += 1
                                temp_annotations.append(annotations[start_idx + i])
                    
                    # If we matched most of the phrase words consecutively
                    if match_count >= len(phrase_words) * 0.7:  # 70% match threshold
                        phrase_annotations = temp_annotations
                        break
                
                # Fallback: individual word matching if consecutive matching failed
                if not phrase_annotations:
                    for phrase_word in phrase_words:
                        for i, (line_word, annotation) in enumerate(zip(line_words, annotations)):
                            if (phrase_word in line_word or line_word in phrase_word or
                                (FUZZY_AVAILABLE and fuzz.ratio(phrase_word, line_word) > 80)):
                                if annotation not in phrase_annotations:
                                    phrase_annotations.append(annotation)
            
            # If we still couldn't find specific words, limit to reasonable subset
            if not phrase_annotations and len(annotations) > 3:
                # Take only the first few words to avoid huge bounding boxes
                max_words = min(len(phrase_words) + 1, len(annotations), 3)
                phrase_annotations = annotations[:max_words]
            elif not phrase_annotations:
                phrase_annotations = annotations
            
            # Create aligned bounding box based on text direction
            if len(phrase_annotations) == 1:
                # Single word - use original oriented bounding box vertices
                annotation = phrase_annotations[0]
                if annotation.bounding_poly.vertices:
                    vertices = annotation.bounding_poly.vertices
                    pts = np.array([(v.x, v.y) for v in vertices], dtype=np.int32)
                    
                    # Draw the oriented bounding box
                    cv2.polylines(annotated, [pts], True, color, 3)
                    
                    # Calculate label position based on text orientation
                    if abs(text_angle) < 15:  # Horizontal text
                        label_x = min(v.x for v in vertices)
                        label_y = min(v.y for v in vertices)
                    elif abs(text_angle - 90) < 15:  # Vertical text (90°)
                        label_x = max(v.x for v in vertices)
                        label_y = min(v.y for v in vertices)
                    elif abs(text_angle + 90) < 15:  # Vertical text (-90°)
                        label_x = min(v.x for v in vertices)
                        label_y = max(v.y for v in vertices)
                    else:  # Diagonal text
                        # Use center point for diagonal orientations
                        label_x = sum(v.x for v in vertices) // 4
                        label_y = min(v.y for v in vertices)
            else:
                # Multiple words - create oriented bounding rectangle aligned with text
                all_points = []
                for annotation in phrase_annotations:
                    if annotation.bounding_poly.vertices:
                        for vertex in annotation.bounding_poly.vertices:
                            all_points.append([vertex.x, vertex.y])
                
                if len(all_points) >= 4:
                    points = np.array(all_points, dtype=np.float32)
                    
                    # Always use minimum area rectangle to align with text direction
                    rect = cv2.minAreaRect(points)
                    box_points = cv2.boxPoints(rect)
                    box_points = np.int32(box_points)
                    
                    # Draw the oriented bounding box
                    cv2.polylines(annotated, [box_points], True, color, 3)
                    
                    # Calculate label position based on text orientation
                    center_x, center_y = rect[0]
                    angle_rad = np.radians(rect[2])
                    
                    if abs(text_angle) < 15:  # Horizontal text
                        label_x = int(min(box_points[:, 0]))
                        label_y = int(min(box_points[:, 1]))
                    elif abs(text_angle - 90) < 15:  # Vertical text (90°)
                        label_x = int(max(box_points[:, 0]))
                        label_y = int(min(box_points[:, 1]))
                    elif abs(text_angle + 90) < 15:  # Vertical text (-90°)
                        label_x = int(min(box_points[:, 0]))
                        label_y = int(max(box_points[:, 1]))
                    elif abs(abs(text_angle) - 180) < 15:  # Upside-down text
                        label_x = int(max(box_points[:, 0]))
                        label_y = int(max(box_points[:, 1]))
                    else:  # Diagonal text
                        # For diagonal text, place label at the "top" relative to reading direction
                        label_x = int(min(box_points[:, 0]))
                        label_y = int(min(box_points[:, 1]))
                else:
                    continue
            
            # Add text label with background, positioned based on text orientation
            label = f"{phrase} ({score:.0f}%)"
            
            # Calculate text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Position label based on text orientation
            if abs(text_angle) < 15:  # Horizontal text
                # Place label above the text
                bg_x1, bg_y1 = label_x, label_y - text_h - 10
                bg_x2, bg_y2 = label_x + text_w + 4, label_y - 4
                text_x, text_y = label_x + 2, label_y - 7
            elif abs(text_angle - 90) < 15:  # Vertical text (90°)
                # Place label to the right of vertical text
                bg_x1, bg_y1 = label_x + 5, label_y
                bg_x2, bg_y2 = label_x + text_w + 9, label_y + text_h + 4
                text_x, text_y = label_x + 7, label_y + text_h
            elif abs(text_angle + 90) < 15:  # Vertical text (-90°)
                # Place label to the left of vertical text
                bg_x1, bg_y1 = label_x - text_w - 9, label_y - text_h - 4
                bg_x2, bg_y2 = label_x - 5, label_y
                text_x, text_y = label_x - text_w - 7, label_y - 4
            elif abs(abs(text_angle) - 180) < 15:  # Upside-down text
                # Place label below upside-down text
                bg_x1, bg_y1 = label_x - text_w - 4, label_y + 4
                bg_x2, bg_y2 = label_x, label_y + text_h + 10
                text_x, text_y = label_x - text_w - 2, label_y + text_h + 7
            else:  # Diagonal text
                # For diagonal text, use offset positioning
                offset_x = int(20 * np.cos(np.radians(text_angle + 90)))
                offset_y = int(20 * np.sin(np.radians(text_angle + 90)))
                
                bg_x1 = label_x + offset_x
                bg_y1 = label_y + offset_y
                bg_x2 = bg_x1 + text_w + 4
                bg_y2 = bg_y1 + text_h + 4
                text_x = bg_x1 + 2
                text_y = bg_y1 + text_h
            
            # Ensure label stays within image bounds
            img_h, img_w = annotated.shape[:2]
            bg_x1 = max(0, min(bg_x1, img_w - text_w - 4))
            bg_y1 = max(0, min(bg_y1, img_h - text_h - 4))
            bg_x2 = max(text_w + 4, min(bg_x2, img_w))
            bg_y2 = max(text_h + 4, min(bg_y2, img_h))
            text_x = bg_x1 + 2
            text_y = bg_y1 + text_h
            
            # Draw background rectangle for text
            cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            
            # Draw the label text
            cv2.putText(annotated, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated


def detect_and_annotate_phrases(image_path, search_phrases=None, 
                               threshold=75, show_plot=True):
    """
    Detect and visually annotate phrases in an image.
    
    Args:
        image_path: Path to image file
        search_phrases: List of phrases to search for (required)
        threshold: Fuzzy matching threshold
        show_plot: Whether to display the result
    
    Returns:
        Dictionary with results and annotated image, or None if no search phrases
    """

    try:
        # Load image
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Instantiate Vision API client
        with suppress_stderr_warnings():
            client = vision.ImageAnnotatorClient()
        
        # Perform advanced text detection with multi-angle support
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        
        # Use document_text_detection for better angle handling
        response = client.document_text_detection(image=image)
        
        # Extract both text annotations and document structure
        text_annotations = response.text_annotations
        
        # Also try regular text_detection as fallback
        if not text_annotations:
            print("📐 No text found with document detection, trying basic detection...")
            response_basic = client.text_detection(image=image)
            text_annotations = response_basic.text_annotations
        
        if not text_annotations:
            print("No text detected in image.")
            return None
        
        print(f"📸 Detected {len(text_annotations)-1} text elements")
        
        # Group text into lines for better phrase detection
        text_lines = group_text_into_lines(text_annotations)
        full_text = text_annotations[0].description
        print(f"📋 Grouped into {len(text_lines)} text lines")
        
        # Show info about detected text angles
        angles = set(line.get('angle', 0) for line in text_lines)
        if angles:
            angle_info = []
            for angle in sorted(angles):
                if abs(angle) < 5:
                    angle_info.append("horizontal")
                elif abs(angle - 90) < 15:
                    angle_info.append("vertical↑")
                elif abs(angle + 90) < 15:
                    angle_info.append("vertical↓")
                elif abs(abs(angle) - 180) < 15:
                    angle_info.append("upside-down")
                else:
                    angle_info.append(f"{angle:.0f}°")
            if len(angle_info) > 1:
                print(f"📐 Text orientations detected: {', '.join(angle_info)}")
        
        # Require search phrases to be provided
        if search_phrases is None:
            print("❌ No search phrases provided. Please specify phrases to search for.")
            return None
        
        # Find complete phrase matches only
        phrase_matches = {}
        total_matches = 0
        
        for phrase in search_phrases:
            # Check if this phrase is meaningful (not just common words)
            if not is_meaningful_phrase(phrase):
                print(f"⏭️  Skipping common word phrase: '{phrase}'")
                continue
                
            print(f"🔍 Searching for: '{phrase}'")
            
            matches = find_complete_phrases(phrase, text_lines, full_text, 
                                           threshold)
            if matches:
                phrase_matches[phrase] = matches
                total_matches += len(matches)
                print(f"🎯 Found {len(matches)} complete matches for '{phrase}'")
                
                # Show what was actually matched
                for match_data, score, match_type in matches:
                    match_text = match_data.get('text', 'Unknown')
                    if 'span_info' in match_data:
                        print(f"   📍 Spanning match: '{match_text}' ({score:.1f}% {match_type})")
                    else:
                        print(f"   📍 Match: '{match_text}' ({score:.1f}% {match_type})")
            else:
                print(f"❌ No matches found for '{phrase}'")
        
        if not phrase_matches:
            print("❌ No phrase matches found.")
            return {
                'image': orig_img,
                'annotated_image': orig_img.copy(),
                'matches': {},
                'total_matches': 0
            }
        
        # Create annotated image
        annotated = draw_phrase_annotations(orig_img, phrase_matches)
        
        # Display results
        if show_plot:
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            plt.title(f"Phrase Detection Results - {total_matches} matches found")
            plt.axis('off')
            
            # Add legend
            legend_text = []
            for phrase, matches in phrase_matches.items():
                legend_text.append(f"{phrase}: {len(matches)} matches")
            
            if legend_text:
                plt.figtext(0.02, 0.02, '\n'.join(legend_text), 
                           fontsize=10, verticalalignment='bottom')
            
            plt.tight_layout()
            plt.show()
        
        # Print detailed results
        print("\n📋 DETAILED MATCHES:")
        for phrase, matches in phrase_matches.items():
            print(f"\n🔍 '{phrase}' ({len(matches)} matches):")
            for match_data, score, match_type in matches:
                if match_data.get('source') == 'full_text':
                    text = f"Full text contains: {phrase}"
                else:
                    text = match_data.get('text', 'Unknown text')
                text = text.replace('\n', ' ')
                print(f"   • {text} ({score:.1f}% {match_type})")
        
        return {
            'image': orig_img,
            'annotated_image': annotated,
            'matches': phrase_matches,
            'total_matches': total_matches,
            'all_text': (text_annotations[0].description
                         if text_annotations else "")
        }
        
    except Exception as e:
        error_msg = str(e)
        if ("DefaultCredentialsError" in error_msg or
                "credentials" in error_msg.lower()):
            print(error_msg)
            print("❌ Google Cloud Authentication Error!")
            print("Please set up authentication or use local OCR scripts.")
        else:
            print(f"❌ Error: {e}")
        return None


def validate_common_word_filtering():
    """Validate that common word filtering works correctly."""
    print("\n🧪 TESTING COMMON WORD FILTERING")
    print("=" * 40)
    
    test_cases = [
        'the',              # Single common word - should be skipped
        'and',              # Single common word - should be skipped  
        'the and',          # Only common words - should be skipped
        'was were',         # Only common words - should be skipped
    ]
    
    image_path = "image/wonder_books_cds1.jpg"
    
    for test_phrase in test_cases:
        results = detect_and_annotate_phrases(
            image_path,
            search_phrases=[test_phrase],
            threshold=FUZZ_THRESHOLD,
            show_plot=False
        )
        
        if results and results['total_matches'] > 0:
            print(f"❌ FAILED: '{test_phrase}' should have been filtered but found {results['total_matches']} matches")
        else:
            print(f"✅ PASSED: '{test_phrase}' correctly filtered out")


def main():
    """Main function to demonstrate phrase detection and annotation."""

    image_path = IMAGE_PATH
    
    # Search terms for music CDs (meaningful phrases only)
    search_terms = SEARCH_TERMS
    
    print("🎵 MEDIA DETECTION AND ANNOTATION")
    print("=" * 50)
    
    results = detect_and_annotate_phrases(
        image_path,
        search_phrases=search_terms,
        threshold=FUZZ_THRESHOLD,
        show_plot=True
    )
    
    if results:
        print(f"\n✅ Successfully processed image with "
              f"{results['total_matches']} phrase matches!")
    else:
        print("❌ Failed to process image.")
    
    # Optionally run validation tests
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        validate_common_word_filtering()


if __name__ == "__main__":
    main()