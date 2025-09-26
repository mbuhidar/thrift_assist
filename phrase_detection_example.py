"""
Enhanced text detection example that focuses on phrases and text blocks
instead of individual words.
"""

# Set environment variables to suppress warnings BEFORE importing Google Cloud
import os
import warnings
from contextlib import redirect_stderr
from io import StringIO
import re

# Suppress ALTS warnings and gRPC verbose logging
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GRPC_TRACE'] = ''
os.environ['GOOGLE_CLOUD_DISABLE_GRPC_FOR_GAE'] = 'true'
os.environ['GLOG_minloglevel'] = '3'

# Suppress all warnings
warnings.filterwarnings("ignore")


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


def extract_phrases(text, min_words=2, max_words=6):
    """
    Extract phrases from text instead of individual words.
    
    Args:
        text: Input text string
        min_words: Minimum number of words in a phrase
        max_words: Maximum number of words in a phrase
    
    Returns:
        List of phrases
    """
    # Clean and split text into words
    words = re.findall(r'\b\w+\b', text.lower())
    phrases = []
    
    # Generate phrases of different lengths
    for phrase_len in range(min_words, min(max_words + 1, len(words) + 1)):
        for i in range(len(words) - phrase_len + 1):
            phrase = ' '.join(words[i:i + phrase_len])
            phrases.append(phrase)
    
    return phrases


def group_text_by_lines(text_annotations):
    """
    Group detected text into lines and phrases based on proximity.
    
    Args:
        text_annotations: Google Vision API text annotations
        
    Returns:
        List of text lines/phrases
    """
    if not text_annotations:
        return []
    
    # Skip the first annotation (full text) and process word-level annotations
    word_annotations = text_annotations[1:] if len(text_annotations) > 1 else []
    
    if not word_annotations:
        return []
    
    # Group words by their vertical position (y-coordinate)
    lines = {}
    for annotation in word_annotations:
        if annotation.bounding_poly.vertices:
            # Use the top-left y-coordinate as the line identifier
            y_pos = annotation.bounding_poly.vertices[0].y
            
            # Find existing line within tolerance (allow some variation)
            line_key = None
            tolerance = 10  # pixels
            
            for existing_y in lines.keys():
                if abs(y_pos - existing_y) <= tolerance:
                    line_key = existing_y
                    break
            
            if line_key is None:
                line_key = y_pos
                lines[line_key] = []
            
            # Add word with its x-position for sorting
            x_pos = annotation.bounding_poly.vertices[0].x
            lines[line_key].append((x_pos, annotation.description, annotation))
    
    # Sort lines by y-position (top to bottom)
    sorted_lines = []
    for y_pos in sorted(lines.keys()):
        # Sort words in each line by x-position (left to right)
        line_words = sorted(lines[y_pos], key=lambda x: x[0])
        
        # Combine words into line text
        line_text = ' '.join([word[1] for word in line_words])
        line_annotations = [word[2] for word in line_words]
        
        sorted_lines.append({
            'text': line_text,
            'annotations': line_annotations,
            'y_position': y_pos
        })
    
    return sorted_lines


def detect_phrases_in_image(image_path, min_phrase_length=2):
    """
    Detect phrases and text blocks in an image.
    
    Args:
        image_path: Path to the image file
        min_phrase_length: Minimum number of words in detected phrases
    
    Returns:
        Dictionary with full text, lines, and phrases
    """
    try:
        # Instantiate client with warnings suppressed
        with suppress_stderr_warnings():
            client = vision.ImageAnnotatorClient()

        # Check if local image exists
        if os.path.exists(image_path):
            # Read the image file as bytes for local images
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            print(f"Using local image: {image_path}")
        else:
            print(f"❌ Image file not found: {image_path}")
            return None

        # Perform text detection
        response = client.text_detection(image=image)
        text_annotations = response.text_annotations

        if not text_annotations:
            print("No text detected in the image.")
            return None

        # Get full text (first annotation contains all detected text)
        full_text = text_annotations[0].description
        
        # Group text into lines
        text_lines = group_text_by_lines(text_annotations)
        
        # Extract phrases from full text
        phrases = extract_phrases(full_text, min_words=min_phrase_length)
        
        results = {
            'full_text': full_text,
            'lines': text_lines,
            'phrases': phrases,
            'word_count': len(text_annotations) - 1  # Subtract 1 for full text annotation
        }
        
        return results

    except Exception as e:
        error_msg = str(e)
        
        # Check if it's a credentials error
        is_creds_error = ("DefaultCredentialsError" in error_msg or
                          "credentials" in error_msg.lower())
        if is_creds_error:
            print("❌ Google Cloud Authentication Error!")
            print("\nTo use Google Vision API, set up authentication:")
            print("1. Create a Google Cloud Project at:")
            print("   https://console.cloud.google.com/")
            print("2. Enable the Vision API for your project")
            print("3. Create a service account and download the JSON key file")
            print("4. Set the environment variable:")
            print("   export GOOGLE_APPLICATION_CREDENTIALS=")
            print("   /path/to/your/service-account-key.json")
        else:
            print(f"❌ Error running Google Vision API: {e}")
        
        return None


def run_phrase_detection_example():
    """Run the phrase detection example."""
    image_path = "image/gw_cds.jpg"
    
    print("🔍 Detecting phrases and text blocks in image...")
    print("=" * 60)
    
    results = detect_phrases_in_image(image_path, min_phrase_length=2)
    
    if not results:
        return
    
    print(f"\n📊 DETECTION SUMMARY:")
    print(f"Total words detected: {results['word_count']}")
    print(f"Text lines found: {len(results['lines'])}")
    print(f"Phrases extracted: {len(results['phrases'])}")
    
    print(f"\n📝 FULL TEXT:")
    print(f'"{results["full_text"]}"')
    
    print(f"\n📋 TEXT LINES (grouped by position):")
    for i, line in enumerate(results['lines'], 1):
        print(f"{i:2}. {line['text']}")
    
    print(f"\n🔤 EXTRACTED PHRASES (2+ words):")
    # Show unique phrases only
    unique_phrases = list(set(results['phrases']))
    unique_phrases.sort(key=len, reverse=True)  # Sort by length, longest first
    
    for i, phrase in enumerate(unique_phrases[:20], 1):  # Show top 20
        print(f"{i:2}. {phrase}")
        
    if len(unique_phrases) > 20:
        print(f"    ... and {len(unique_phrases) - 20} more phrases")


if __name__ == "__main__":
    run_phrase_detection_example()