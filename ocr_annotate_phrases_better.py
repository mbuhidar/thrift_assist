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

# Configuration
IMAGE_PATH = "image/iCloud_Photos/IMG_4918.JPEG"

SEARCH_TERMS = [
    'Homecoming',
    'Circle of Three',
    ]

FUZZ_THRESHOLD = 80  # Similarity threshold for phrase matching

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
    Enhanced to handle text at various angles including upside down.
    
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
    
    # Search in text lines for complete phrases
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
    
    # Remove duplicates and sort by score
    seen_texts = set()
    unique_matches = []
    for match in sorted(matches, key=lambda x: x[1], reverse=True):
        match_text = match[0]['text'] if 'text' in match[0] else str(match[0])
        if match_text not in seen_texts:
            seen_texts.add(match_text)
            unique_matches.append(match)
    
    return unique_matches


def draw_phrase_annotations(image, phrase_matches, phrase_colors=None):
    """
    Draw bounding boxes around detected complete phrases.
    
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
            
            # Find the specific words in line that match the phrase
            line_text_normalized = normalize_text_for_search(match_data.get('text', ''))
            phrase_normalized = normalize_text_for_search(phrase)
            phrase_words = phrase_normalized.split()
            
            # Calculate precise bounding box for the phrase
            phrase_annotations = []
            
            if phrase_normalized in line_text_normalized:
                # Find the starting position of the phrase in the line
                phrase_start = line_text_normalized.find(phrase_normalized)
                phrase_end = phrase_start + len(phrase_normalized)
                
                # Map character positions to word annotations
                char_pos = 0
                for annotation in annotations:
                    word = annotation.description.lower()
                    word_start = char_pos
                    word_end = char_pos + len(word)
                    
                    # Check if this word overlaps with our phrase
                    if (word_start < phrase_end and word_end > phrase_start):
                        phrase_annotations.append(annotation)
                    
                    char_pos = word_end + 1  # +1 for space
            else:
                # For fuzzy matches, try to match individual words
                line_words = [normalize_text_for_search(ann.description) for ann in annotations]
                for phrase_word in phrase_words:
                    for i, (line_word, annotation) in enumerate(zip(line_words, annotations)):
                        if (phrase_word in line_word or line_word in phrase_word or
                            (FUZZY_AVAILABLE and fuzz.ratio(phrase_word, line_word) > 80)):
                            if annotation not in phrase_annotations:
                                phrase_annotations.append(annotation)
            
            # If we couldn't find specific words, use all annotations as fallback
            if not phrase_annotations:
                phrase_annotations = annotations
            
            # Calculate tighter bounding box
            all_points = []
            for annotation in phrase_annotations:
                if annotation.bounding_poly.vertices:
                    for vertex in annotation.bounding_poly.vertices:
                        all_points.append((vertex.x, vertex.y))
            
            if len(all_points) >= 4:
                # Create combined bounding box
                min_x = min(p[0] for p in all_points)
                max_x = max(p[0] for p in all_points)
                min_y = min(p[1] for p in all_points)
                max_y = max(p[1] for p in all_points)
                
                # Draw bounding box
                pts = np.array([(min_x, min_y), (max_x, min_y),
                               (max_x, max_y), (min_x, max_y)], dtype=np.int32)
                cv2.polylines(annotated, [pts], True, color, 3)
                
                # Add text label
                label = f"{phrase} ({score:.0f}%)"
                
                # Add background for text
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (min_x, min_y-text_h-8),
                             (min_x+text_w+4, min_y-2), color, -1)
                
                # Add text
                cv2.putText(annotated, label, (min_x+2, min_y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
        document = response.full_text_annotation
        
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
        
        # Debug: Show more detected text lines to find missing artists
        print(f"\nFound {len(text_lines)} total text lines")
        print(f"🔍 DEBUG: Showing all text lines (looking for Dean Martin, etc.):")
        for i, line in enumerate(text_lines):
            line_text = line['text']
            angle = line.get('angle', 0)
            
            # Check for potential upside-down matches and common OCR errors
            upside_down_score = 0
            potential_dean_martin = False
            if FUZZY_AVAILABLE:
                for term in ['Dean Martin', 'Toad the Wet Sprocket', 'James Taylor']:
                    score = try_reverse_text_matching(term, line_text)
                    if score > upside_down_score:
                        upside_down_score = score
                
                # Special check for Dean Martin fragments (common OCR errors)
                dean_fragments = ['dean', 'martin', 'naeđ', 'nitram', 'ɴɪʇɹɐɯ', 'uɐǝp']
                if any(frag in line_text.lower() for frag in dean_fragments):
                    potential_dean_martin = True
            
            # Highlight lines that might contain our search terms or be upside down
            if any(term.lower() in line_text.lower() for term in ['dean', 'martin', 'toad', 'james', 'taylor']):
                print(f"   ⭐ {i+1}: \"{line_text}\" (angle: {angle:.1f}°)")
            elif potential_dean_martin:
                print(f"   🔍 {i+1}: \"{line_text}\" (angle: {angle:.1f}°, dean_fragments)")
            elif upside_down_score > 70:
                print(f"   🔄 {i+1}: \"{line_text}\" (angle: {angle:.1f}°, reverse_match: {upside_down_score:.1f}%)")
            else:
                print(f"   {i+1}: \"{line_text}\" (angle: {angle:.1f}°)")
        
        # Find complete phrase matches only
        phrase_matches = {}
        total_matches = 0
        
        for phrase in search_phrases:
            # Check if this phrase is meaningful (not just common words)
            if not is_meaningful_phrase(phrase):
                print(f"⏭️  Skipping common word phrase: '{phrase}'")
                continue
                
            print(f"\n🔍 Searching for: '{phrase}'")
            matches = find_complete_phrases(phrase, text_lines, full_text, 
                                           threshold)
            if matches:
                phrase_matches[phrase] = matches
                total_matches += len(matches)
                print(f"🎯 Found {len(matches)} complete matches for '{phrase}'")
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
    
    print("🎵 CD DETECTION AND ANNOTATION")
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