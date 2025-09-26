"""
Advanced phrase detection with fuzzy matching and search capabilities.
This script focuses on meaningful phrases and provides search functionality.
"""

# Set environment variables to suppress warnings BEFORE importing Google Cloud
import os
import warnings
from contextlib import redirect_stderr
from io import StringIO
import re
from collections import Counter

# Suppress ALTS warnings and gRPC verbose logging
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GRPC_TRACE'] = ''
os.environ['GOOGLE_CLOUD_DISABLE_GRPC_FOR_GAE'] = 'true'
os.environ['GLOG_minloglevel'] = '3'

# Suppress all warnings
warnings.filterwarnings("ignore")

try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️  rapidfuzz not available. Install with: pip install rapidfuzz")


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


def clean_text(text):
    """Clean and normalize text for better phrase extraction."""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep letters, numbers, spaces, and hyphens
    text = re.sub(r'[^\w\s\-\']', ' ', text)
    return text


def extract_meaningful_phrases(text, min_length=3, max_length=8):
    """
    Extract meaningful phrases from text with quality filtering.
    
    Args:
        text: Input text string
        min_length: Minimum number of words in phrase
        max_length: Maximum number of words in phrase
    
    Returns:
        List of high-quality phrases
    """
    cleaned_text = clean_text(text)
    words = cleaned_text.lower().split()
    
    if len(words) < min_length:
        return []
    
    phrases = []
    
    # Generate phrases of different lengths
    for phrase_len in range(min_length, min(max_length + 1, len(words) + 1)):
        for i in range(len(words) - phrase_len + 1):
            phrase_words = words[i:i + phrase_len]
            phrase = ' '.join(phrase_words)
            
            # Quality filters
            if is_meaningful_phrase(phrase_words):
                phrases.append(phrase)
    
    return phrases


def is_meaningful_phrase(words):
    """
    Determine if a phrase is meaningful based on various criteria.
    
    Args:
        words: List of words in the phrase
    
    Returns:
        Boolean indicating if phrase is meaningful
    """
    # Skip if too many single characters
    single_chars = sum(1 for word in words if len(word) == 1)
    if single_chars > len(words) * 0.5:
        return False
    
    # Skip if too many numbers
    numbers = sum(1 for word in words if word.isdigit())
    if numbers > len(words) * 0.7:
        return False
    
    # Skip if all words are very short
    avg_length = sum(len(word) for word in words) / len(words)
    if avg_length < 2:
        return False
    
    # Skip common meaningless patterns
    meaningless_patterns = {
        'thank you for supporting',
        'you for supporting',
        'for supporting',
        'supporting goodwill',
        'goodwill',
        'various artists',
        'greatest hits',
        'best of'
    }
    phrase = ' '.join(words)
    if phrase in meaningless_patterns:
        return False
    
    return True


def search_phrases(phrases, search_terms, threshold=70):
    """
    Search for specific phrases using fuzzy matching.
    
    Args:
        phrases: List of phrases to search in
        search_terms: List of terms to search for
        threshold: Minimum similarity score (0-100)
    
    Returns:
        Dictionary of matches
    """
    if not FUZZY_AVAILABLE:
        # Simple substring matching fallback
        matches = {}
        for term in search_terms:
            term_lower = term.lower()
            matches[term] = [p for p in phrases if term_lower in p.lower()]
        return matches
    
    matches = {}
    for term in search_terms:
        term_matches = []
        for phrase in phrases:
            similarity = fuzz.partial_ratio(term.lower(), phrase.lower())
            if similarity >= threshold:
                term_matches.append((phrase, similarity))
        
        # Sort by similarity score
        term_matches.sort(key=lambda x: x[1], reverse=True)
        matches[term] = term_matches
    
    return matches


def detect_and_search_phrases(image_path, search_terms=None, min_phrase_len=3):
    """
    Detect phrases in image and optionally search for specific terms.
    
    Args:
        image_path: Path to the image file
        search_terms: List of terms to search for (optional)
        min_phrase_len: Minimum phrase length
    
    Returns:
        Dictionary with detection results
    """
    try:
        # Instantiate client with warnings suppressed
        with suppress_stderr_warnings():
            client = vision.ImageAnnotatorClient()

        # Check if local image exists
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None

        # Read and process image
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        print(f"📸 Processing image: {image_path}")

        # Perform text detection
        response = client.text_detection(image=image)
        text_annotations = response.text_annotations

        if not text_annotations:
            print("No text detected in the image.")
            return None

        # Get full text
        full_text = text_annotations[0].description
        
        # Extract meaningful phrases
        phrases = extract_meaningful_phrases(full_text, min_phrase_len)
        
        # Remove duplicates and count occurrences
        phrase_counts = Counter(phrases)
        unique_phrases = list(phrase_counts.keys())
        
        # Search for specific terms if provided
        search_results = {}
        if search_terms:
            search_results = search_phrases(unique_phrases, search_terms)
        
        results = {
            'full_text': full_text,
            'total_phrases': len(phrases),
            'unique_phrases': len(unique_phrases),
            'phrase_counts': phrase_counts,
            'search_results': search_results,
            'top_phrases': phrase_counts.most_common(20)
        }
        
        return results

    except Exception as e:
        error_msg = str(e)
        if "DefaultCredentialsError" in error_msg or "credentials" in error_msg.lower():
            print("❌ Google Cloud Authentication Error!")
            print("Set up authentication or use local OCR scripts instead.")
        else:
            print(f"❌ Error: {e}")
        return None


def run_phrase_search_example():
    """Run the phrase search example with specific search terms."""
    image_path = "image/gw_cds.jpg"
    
    # Define search terms you're looking for
    search_terms = [
        "greatest hits",
        "best of", 
        "original soundtrack",
        "live",
        "compilation",
        "various artists",
        "christmas",
        "classical",
        "jazz",
        "rock",
        "blues"
    ]
    
    print("🔍 PHRASE DETECTION AND SEARCH")
    print("=" * 50)
    
    results = detect_and_search_phrases(image_path, search_terms, min_phrase_len=2)
    
    if not results:
        return
    
    print(f"\n📊 DETECTION SUMMARY:")
    print(f"Total phrases extracted: {results['total_phrases']}")
    print(f"Unique phrases: {results['unique_phrases']}")
    
    print(f"\n🔥 TOP PHRASES (by frequency):")
    for i, (phrase, count) in enumerate(results['top_phrases'], 1):
        print(f"{i:2}. {phrase} ({count}x)")
    
    if results['search_results']:
        print(f"\n🎯 SEARCH RESULTS:")
        for term, matches in results['search_results'].items():
            if matches:
                print(f"\n🔍 '{term}':")
                if FUZZY_AVAILABLE:
                    for phrase, score in matches[:5]:  # Show top 5 matches
                        print(f"   • {phrase} ({score}% match)")
                else:
                    for phrase in matches[:5]:
                        print(f"   • {phrase}")
            else:
                print(f"🔍 '{term}': No matches found")
    
    # Show some interesting phrases
    print(f"\n🎵 INTERESTING MUSIC-RELATED PHRASES:")
    music_phrases = [p for p, _ in results['top_phrases'] 
                     if any(word in p for word in ['music', 'song', 'album', 'artist', 'band', 'records'])]
    for phrase in music_phrases[:10]:
        print(f"   • {phrase}")


if __name__ == "__main__":
    run_phrase_search_example()