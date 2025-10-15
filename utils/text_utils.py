"""Text processing utilities for OCR."""

import re
from typing import Set

def normalize_text_for_search(text: str) -> str:
    """Normalize text for searching by converting to lowercase and removing extra whitespace."""
    if not text:
        return ""
    # Convert to lowercase for case-insensitive matching
    normalized = text.lower()
    # Replace multiple spaces with single space
    normalized = ' '.join(normalized.split())
    return normalized


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
    words = phrase.lower().strip().split()
    
    if len(words) == 1:
        return words[0] not in common_words
    
    non_common = [w for w in words if w not in common_words]
    return bool(non_common) or len(set(words)) < len(words)
