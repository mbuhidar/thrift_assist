"""Text processing utilities for OCR."""

import re
from typing import Set

def normalize_text_for_search(text: str) -> str:
    """
    Normalize text for better matching across different orientations.
    Handles OCR artifacts from rotated text and makes matching case-insensitive.
    """
    normalized = re.sub(r'\s+', ' ', text.strip())
    
    char_map = {
        '|': 'I', '¡': '!', '∩': 'n', 'ɹ': 'r', 'ɐ': 'a', 
        'ǝ': 'e', 'ᴍ': 'w', 'rn': 'm', 'cl': 'd', 'ii': 'n',
    }
    
    for ocr_char, standard_char in char_map.items():
        normalized = normalized.replace(ocr_char, standard_char)
    
    return normalized.lower()


def is_meaningful_phrase(phrase: str, common_words: Set[str]) -> bool:
    """
    Check if a phrase is meaningful (not just common words).
    Returns True if the phrase should be searched for.
    """
    words = phrase.lower().strip().split()
    
    if len(words) == 1:
        return words[0] not in common_words
    
    non_common = [w for w in words if w not in common_words]
    return bool(non_common) or len(set(words)) < len(words)
