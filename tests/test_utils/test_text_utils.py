"""Tests for text utility functions."""

import pytest


@pytest.mark.unit
class TestTextUtils:
    """Test text utility functions."""
    
    def test_normalize_text_for_search_lowercase(self):
        """Test text normalization converts to lowercase."""
        from utils.text_utils import normalize_text_for_search
        
        assert normalize_text_for_search("HELLO") == "hello"
        assert normalize_text_for_search("HeLLo WoRLd") == "hello world"
        assert normalize_text_for_search("ABC123") == "abc123"
    
    def test_normalize_text_for_search_whitespace(self):
        """Test text normalization handles whitespace."""
        from utils.text_utils import normalize_text_for_search
        
        # Multiple spaces
        assert normalize_text_for_search("hello   world") == "hello world"
        assert normalize_text_for_search("a  b  c") == "a b c"
        
        # Leading/trailing spaces
        assert normalize_text_for_search("  hello  ") == "hello"
        assert normalize_text_for_search("\thello\t") == "hello"
        
        # Newlines and mixed whitespace
        assert normalize_text_for_search("hello\nworld") == "hello world"
        assert normalize_text_for_search("hello\t\nworld") == "hello world"
    
    def test_normalize_text_for_search_empty_and_edge_cases(self):
        """Test text normalization edge cases."""
        from utils.text_utils import normalize_text_for_search
        
        # Empty string
        assert normalize_text_for_search("") == ""
        
        # Only whitespace
        assert normalize_text_for_search("   ") == ""
        assert normalize_text_for_search("\n\t  ") == ""
        
        # Single character
        assert normalize_text_for_search("a") == "a"
        assert normalize_text_for_search("A") == "a"
    
    def test_normalize_text_for_search_punctuation(self):
        """Test text normalization with punctuation."""
        from utils.text_utils import normalize_text_for_search
        
        result = normalize_text_for_search("Hello, World!")
        assert "hello" in result.lower()
        assert "world" in result.lower()
        
        # Apostrophes and hyphens
        result2 = normalize_text_for_search("John's Book")
        assert "john" in result2.lower()
        assert "book" in result2.lower()
        
        result3 = normalize_text_for_search("mother-in-law")
        assert "mother" in result3.lower()
    
    def test_normalize_text_for_search_unicode(self):
        """Test text normalization with unicode characters."""
        from utils.text_utils import normalize_text_for_search
        
        # Basic unicode
        result = normalize_text_for_search("café")
        assert "caf" in result.lower()
        
        # Emoji and special characters
        result2 = normalize_text_for_search("hello 😊 world")
        assert "hello" in result2.lower()
        assert "world" in result2.lower()
    
    def test_normalize_text_for_search_numbers(self):
        """Test text normalization with numbers."""
        from utils.text_utils import normalize_text_for_search
        
        assert "123" in normalize_text_for_search("test 123")
        assert "2024" in normalize_text_for_search("Year 2024")
        
        # Mixed alphanumeric
        result = normalize_text_for_search("Room 101B")
        assert "room" in result.lower()
        assert "101" in result
    
    def test_is_meaningful_phrase_meaningful(self):
        """Test phrase meaningfulness check for meaningful phrases."""
        from utils.text_utils import is_meaningful_phrase
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        common_words = config.common_words
        
        # Book titles and authors
        assert is_meaningful_phrase("John's Book", common_words) is True
        assert is_meaningful_phrase("Billy Joel", common_words) is True
        assert is_meaningful_phrase("Harry Potter", common_words) is True
        
        # Single meaningful word
        assert is_meaningful_phrase("Shakespeare", common_words) is True
        assert is_meaningful_phrase("Python", common_words) is True
    
    def test_is_meaningful_phrase_not_meaningful(self):
        """Test phrase meaningfulness check for non-meaningful phrases."""
        from utils.text_utils import is_meaningful_phrase
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        common_words = config.common_words
        
        # Only common words
        assert is_meaningful_phrase("the and", common_words) is False
        assert is_meaningful_phrase("a of", common_words) is False
        assert is_meaningful_phrase("the or and", common_words) is False
    
    def test_is_meaningful_phrase_mixed(self):
        """Test phrase meaningfulness check for mixed phrases."""
        from utils.text_utils import is_meaningful_phrase
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        common_words = config.common_words
        
        # Mixed common and meaningful words
        assert is_meaningful_phrase("the book", common_words) is True
        assert is_meaningful_phrase("a nice day", common_words) is True
        assert is_meaningful_phrase("of mice and men", common_words) is True
    
    def test_is_meaningful_phrase_edge_cases(self):
        """Test phrase meaningfulness check edge cases."""
        from utils.text_utils import is_meaningful_phrase
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        common_words = config.common_words
        
        # Empty string
        assert is_meaningful_phrase("", common_words) is False
        
        # Single common word
        assert is_meaningful_phrase("the", common_words) is False
        assert is_meaningful_phrase("and", common_words) is False
        
        # Very short meaningful phrases
        assert is_meaningful_phrase("AI", common_words) is True
        assert is_meaningful_phrase("Go", common_words) is True

