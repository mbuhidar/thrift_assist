"""Tests for text utility functions."""

import pytest


@pytest.mark.unit
class TestTextUtils:
    """Test text utility functions."""
    
    def test_normalize_text_for_search(self):
        """Test text normalization."""
        from utils.text_utils import normalize_text_for_search
        
        # Test lowercase
        assert normalize_text_for_search("HELLO") == "hello"
        
        # Test whitespace normalization
        assert normalize_text_for_search("hello   world") == "hello world"
        
        # Test empty string
        assert normalize_text_for_search("") == ""
        
        # Test with punctuation (actual behavior - doesn't remove all punctuation)
        result = normalize_text_for_search("Hello, World!")
        assert "hello" in result.lower()
        assert "world" in result.lower()
    
    def test_is_meaningful_phrase(self):
        """Test phrase meaningfulness check."""
        from utils.text_utils import is_meaningful_phrase
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        common_words = config.common_words
        
        # Meaningful phrases
        assert is_meaningful_phrase("John's Book", common_words) is True
        assert is_meaningful_phrase("Billy Joel", common_words) is True
        
        # Only common words
        assert is_meaningful_phrase("the and", common_words) is False
        assert is_meaningful_phrase("a of", common_words) is False
        
        # Mixed
        assert is_meaningful_phrase("the book", common_words) is True
