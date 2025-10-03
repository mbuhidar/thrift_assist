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
    
    THRESHOLD EXPLANATION:
    - Score ranges from 0-100 (percentage match)
    - Higher threshold = more strict matching
    - Lower threshold = more lenient matching
    
    Examples:
    - threshold=90: "Lee Child" matches "Lee Child" (100%) but not "L. Child" (85%)
    - threshold=75: "Lee Child" matches both "Lee Child" (100%) and "L. Child" (85%)
    - threshold=50: Very lenient, matches partial/fuzzy text
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
        threshold: Minimum similarity for fuzzy matching (50-100)
                  - 50-60: Very lenient (catches many false positives)
                  - 70-80: Balanced (good for most use cases)
                  - 85-95: Strict (fewer false positives, may miss variations)
                  - 95-100: Very strict (only near-exact matches)
    
    Returns:
        List of matching text segments with scores
        
    THRESHOLD BEHAVIOR:
    - Exact substring matches always return 100% regardless of threshold
    - Fuzzy matching uses threshold to filter results
    - Common word phrases (like "the", "and") use stricter matching
    - Non-common phrases use the specified threshold
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
            # Find the exact word boundaries for tighter bounding box
            enhanced_line = enhance_match_with_word_boundaries(line, phrase, phrase_normalized, line_text_normalized)
            matches.append((enhanced_line, 100, "complete_phrase"))
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


def draw_phrase_annotations(image, phrase_matches, phrase_colors=None, text_scale=100):
    """
    Draw bounding boxes around detected complete phrases.
    Enhanced to fit text phrases more closely using exact word boundaries.
    """
    annotated = image.copy()
    
    if not phrase_matches:
        return annotated
    
    # Set up default colors for phrases if not provided
    if phrase_colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
        phrase_colors = {phrase: colors[i % len(colors)] 
                        for i, phrase in enumerate(phrase_matches.keys())}
    
    # Collect all text label positions to detect overlaps
    label_positions = []
    
    for phrase, matches in phrase_matches.items():
        color = phrase_colors.get(phrase, (255, 0, 0))
        
        for match_data, score, match_type in matches:
            # Extract bounding box coordinates more precisely
            x1, y1, x2, y2 = None, None, None, None
            
            if 'annotations' in match_data and match_data['annotations']:
                # Calculate tight bounding box from actual matched words
                phrase_words = phrase.lower().split()
                matched_annotations = []
                
                # Find annotations that correspond to our phrase words
                for annotation in match_data['annotations']:
                    annotation_text = annotation.description.lower()
                    # Check if this annotation contains any of our phrase words
                    if any(word in annotation_text or annotation_text in word for word in phrase_words):
                        matched_annotations.append(annotation)
                
                # If we found specific word matches, use those; otherwise use all annotations
                target_annotations = matched_annotations if matched_annotations else match_data['annotations']
                
                # Calculate tight bounding box from target annotations
                all_x_coords = []
                all_y_coords = []
                
                for annotation in target_annotations:
                    if hasattr(annotation, 'bounding_poly') and annotation.bounding_poly.vertices:
                        for vertex in annotation.bounding_poly.vertices:
                            all_x_coords.append(vertex.x)
                            all_y_coords.append(vertex.y)
                
                if all_x_coords and all_y_coords:
                    # Add small padding for better visibility
                    padding = 3
                    x1 = min(all_x_coords) - padding
                    y1 = min(all_y_coords) - padding
                    x2 = max(all_x_coords) + padding
                    y2 = max(all_y_coords) + padding
            
            # If still no coordinates found, skip this match
            if any(coord is None for coord in [x1, y1, x2, y2]):
                print(f"WARNING: No bounding box coordinates found for '{phrase}', skipping annotation")
                continue
            
            # Ensure coordinates are valid and within image bounds
            height, width = annotated.shape[:2]
            x1 = max(0, min(int(x1), width - 1))
            y1 = max(0, min(int(y1), height - 1))
            x2 = max(x1 + 1, min(int(x2), width))
            y2 = max(y1 + 1, min(int(y2), height))
            
            # Draw tighter bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Add text label with background
            label = f"{phrase} ({score:.0f}%)"
            
            # Calculate text size for background - SCALABLE FONT SIZE
            base_font_scale = 0.8
            font_scale = base_font_scale * (text_scale / 100.0)
            base_thickness = 2
            thickness = max(1, int(base_thickness * (text_scale / 100.0)))
            
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Calculate initial label position
            label_x = x1
            label_y = max(text_h + 5, y1 - 10)
            
            # Check for overlaps and adjust position with leader lines
            label_rect = (label_x, label_y - text_h - 5, label_x + text_w + 10, label_y + 5)
            adjusted_position = find_non_overlapping_position(label_rect, label_positions, annotated.shape)
            
            if adjusted_position != label_rect:
                # Draw leader line from bounding box to adjusted label position
                box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                label_center = (adjusted_position[0] + text_w // 2, adjusted_position[1] + text_h // 2)
                
                # Draw thin leader line
                cv2.line(annotated, box_center, label_center, color, 2)
                
                # Update label position
                label_x = adjusted_position[0]
                label_y = adjusted_position[1] + text_h
            
            # Store the final position
            label_positions.append((label_x, label_y - text_h - 5, label_x + text_w + 10, label_y + 5))
            
            # Draw background rectangle for text
            cv2.rectangle(annotated, (label_x, label_y - text_h - 5), 
                         (label_x + text_w + 10, label_y + 5), color, -1)
            
            # Draw the label text
            cv2.putText(annotated, label, (label_x + 5, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return annotated

def find_non_overlapping_position(rect, existing_positions, image_shape):
    """
    Find a position for the label that doesn't overlap with existing labels.
    
    Args:
        rect: (x1, y1, x2, y2) - desired rectangle position
        existing_positions: List of existing label rectangles
        image_shape: (height, width, channels) of the image
    
    Returns:
        Adjusted rectangle position (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1
    
    # Check if current position overlaps with any existing label
    for existing_rect in existing_positions:
        if rectangles_overlap(rect, existing_rect):
            # Try positions around the original location
            offsets = [
                (0, -height - 10),    # Above
                (0, height + 10),     # Below
                (-width - 10, 0),     # Left
                (width + 10, 0),      # Right
                (-width - 10, -height - 10),  # Top-left
                (width + 10, -height - 10),   # Top-right
                (-width - 10, height + 10),   # Bottom-left
                (width + 10, height + 10),    # Bottom-right
            ]
            
            for dx, dy in offsets:
                new_x1 = max(0, min(x1 + dx, image_shape[1] - width))
                new_y1 = max(0, min(y1 + dy, image_shape[0] - height))
                new_rect = (new_x1, new_y1, new_x1 + width, new_y1 + height)
                
                # Check if this position is clear
                if not any(rectangles_overlap(new_rect, existing) for existing in existing_positions):
                    return new_rect
            
            # If no clear position found, use a stacked position
            return find_stacked_position(rect, existing_positions, image_shape)
    
    return rect

def rectangles_overlap(rect1, rect2):
    """Check if two rectangles overlap."""
    x1a, y1a, x2a, y2a = rect1
    x1b, y1b, x2b, y2b = rect2
    
    return not (x2a <= x1b or x2b <= x1a or y2a <= y1b or y2b <= y1a)

def find_stacked_position(rect, existing_positions, image_shape):
    """Find a position by stacking labels vertically."""
    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1
    
    # Find the lowest existing label in the area
    max_y = 0
    for existing_rect in existing_positions:
        ex1, ey1, ex2, ey2 = existing_rect
        # Check if labels are in similar horizontal area
        if not (x2 <= ex1 or ex2 <= x1):
            max_y = max(max_y, ey2)
    
    # Place new label below the lowest one
    new_y1 = max_y + 5
    new_y2 = new_y1 + height
    
    # Ensure it fits in the image
    if new_y2 > image_shape[0]:
        new_y1 = max(0, image_shape[0] - height)
        new_y2 = new_y1 + height
    
    return (x1, new_y1, x2, new_y2)

def detect_and_annotate_phrases(image_path, search_phrases=None, 
                               threshold=75, show_plot=True, text_scale=100):
    """
    Detect and visually annotate phrases in an image.
    
    Args:
        image_path: Path to image file
        search_phrases: List of phrases to search for (required)
        threshold: Fuzzy matching threshold (50-100, default=75)
        show_plot: Whether to display the result
        text_scale: Text annotation size percentage (50-200, default=100)
    
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
        annotated = draw_phrase_annotations(orig_img, phrase_matches, text_scale=text_scale)
        
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


def enhance_match_with_word_boundaries(line, original_phrase, phrase_normalized, line_text_normalized):
    """
    Enhance a matched line with precise word boundary information for tighter bounding boxes.
    """
    enhanced_line = line.copy()
    
    if 'annotations' not in line or not line['annotations']:
        return enhanced_line
    
    # Find which words in the line correspond to our phrase
    phrase_words = phrase_normalized.split()
    line_words = line_text_normalized.split()
    
    # Find the starting position of our phrase in the line
    phrase_start_idx = None
    for i in range(len(line_words) - len(phrase_words) + 1):
        if line_words[i:i+len(phrase_words)] == phrase_words:
            phrase_start_idx = i
            break
    
    if phrase_start_idx is not None:
        # Select only the annotations that correspond to our phrase words
        phrase_annotations = []
        word_count = 0
        
        for annotation in line['annotations']:
            if phrase_start_idx <= word_count < phrase_start_idx + len(phrase_words):
                phrase_annotations.append(annotation)
            word_count += len(annotation.description.split())
            if word_count > phrase_start_idx + len(phrase_words):
                break
        
        if phrase_annotations:
            enhanced_line['annotations'] = phrase_annotations
    
    return enhanced_line