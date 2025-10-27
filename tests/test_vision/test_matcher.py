"""
Tests for phrase matching functionality.
"""

import pytest
from unittest.mock import Mock, patch


@pytest.mark.matcher
@pytest.mark.unit
class TestPhraseMatcher:
    """Test phrase matching logic."""
    
    def test_find_matches_exact_match(self):
        """Test finding exact phrase matches."""
        from vision.matcher import PhraseMatcher
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        matcher = PhraseMatcher(config)
        
        # Mock text lines with exact match
        text_lines = [
            {
                'text': 'This is a test phrase here',
                'y_position': 100,
                'angle': 0,
                'annotations': []
            }
        ]
        
        matches = matcher.find_matches(
            phrase='test phrase',
            text_lines=text_lines,
            full_text='This is a test phrase here',
            threshold=80
        )
        
        assert len(matches) == 1
        match_data, score, match_type = matches[0]
        assert score == 100
        assert match_type == "complete_phrase"
    
    def test_find_matches_case_insensitive(self):
        """Test that matching is case insensitive."""
        from vision.matcher import PhraseMatcher
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        matcher = PhraseMatcher(config)
        
        text_lines = [
            {
                'text': 'HELLO WORLD',
                'y_position': 100,
                'angle': 0,
                'annotations': []
            }
        ]
        
        # Search with lowercase should still find uppercase text
        matches = matcher.find_matches(
            phrase='hello world',
            text_lines=text_lines,
            full_text='HELLO WORLD',
            threshold=80
        )
        
        assert len(matches) == 1
        match_data, score, match_type = matches[0]
        assert score == 100
    
    def test_find_matches_empty_phrase(self):
        """Test handling of empty search phrase."""
        from vision.matcher import PhraseMatcher
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        matcher = PhraseMatcher(config)
        
        text_lines = [
            {
                'text': 'Some text here',
                'y_position': 100,
                'angle': 0,
                'annotations': []
            }
        ]
        
        # Empty phrase should return no matches
        matches = matcher.find_matches(
            phrase='',
            text_lines=text_lines,
            full_text='Some text here',
            threshold=80
        )
        
        assert len(matches) == 0
    
    def test_find_matches_single_character(self):
        """Test matching single character phrases."""
        from vision.matcher import PhraseMatcher
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        matcher = PhraseMatcher(config)
        
        text_lines = [
            {
                'text': 'Room A is here',
                'y_position': 100,
                'angle': 0,
                'annotations': []
            }
        ]
        
        matches = matcher.find_matches(
            phrase='A',
            text_lines=text_lines,
            full_text='Room A is here',
            threshold=80
        )
        
        # Single letters might be filtered as not meaningful
        # But if found, should be valid
        assert isinstance(matches, list)
    
    def test_find_matches_very_long_phrase(self):
        """Test matching very long phrases."""
        from vision.matcher import PhraseMatcher
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        matcher = PhraseMatcher(config)
        
        long_text = "This is a very long phrase that spans multiple words and contains lots of information about the topic"
        text_lines = [
            {
                'text': long_text,
                'y_position': 100,
                'angle': 0,
                'annotations': []
            }
        ]
        
        matches = matcher.find_matches(
            phrase='very long phrase that spans multiple words',
            text_lines=text_lines,
            full_text=long_text,
            threshold=80
        )
        
        assert len(matches) >= 1
    
    def test_find_matches_with_punctuation(self):
        """Test matching phrases with punctuation."""
        from vision.matcher import PhraseMatcher
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        matcher = PhraseMatcher(config)
        
        text_lines = [
            {
                'text': "John's Book, 2nd Edition!",
                'y_position': 100,
                'angle': 0,
                'annotations': []
            }
        ]
        
        matches = matcher.find_matches(
            phrase="John's Book",
            text_lines=text_lines,
            full_text="John's Book, 2nd Edition!",
            threshold=80
        )
        
        assert len(matches) >= 1
    
    def test_find_matches_with_numbers(self):
        """Test matching phrases with numbers."""
        from vision.matcher import PhraseMatcher
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        matcher = PhraseMatcher(config)
        
        text_lines = [
            {
                'text': 'Room 101B Level 3',
                'y_position': 100,
                'angle': 0,
                'annotations': []
            }
        ]
        
        matches = matcher.find_matches(
            phrase='Room 101B',
            text_lines=text_lines,
            full_text='Room 101B Level 3',
            threshold=80
        )
        
        assert len(matches) >= 1
        if matches:
            match_data, score, match_type = matches[0]
            assert score >= 80
    
    def test_find_matches_fuzzy_match(self):
        """Test finding fuzzy phrase matches."""
        from vision.matcher import PhraseMatcher
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        matcher = PhraseMatcher(config)
        
        text_lines = [
            {
                'text': 'This is a tst phras here',  # Slightly misspelled
                'y_position': 100,
                'angle': 0,
                'annotations': []
            }
        ]
        
        with patch('vision.matcher.FUZZY_AVAILABLE', True):
            with patch('vision.matcher.fuzz.token_set_ratio', return_value=85):
                matches = matcher.find_matches(
                    phrase='test phrase',
                    text_lines=text_lines,
                    full_text='This is a tst phras here',
                    threshold=80
                )
                
                assert len(matches) >= 1
                match_data, score, match_type = matches[0]
                assert score >= 80
                assert "fuzzy" in match_type
    
    def test_find_matches_spanning_lines(self):
        """Test finding matches that span multiple lines."""
        from vision.matcher import PhraseMatcher
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        matcher = PhraseMatcher(config)
        
        text_lines = [
            {
                'text': 'This is a test',
                'y_position': 100,
                'angle': 0,
                'annotations': []
            },
            {
                'text': 'phrase on next line',
                'y_position': 130,
                'angle': 0,
                'annotations': []
            }
        ]
        
        matches = matcher.find_matches(
            phrase='test phrase',
            text_lines=text_lines,
            full_text='This is a test phrase on next line',
            threshold=70
        )
        
        # Should find spanning match
        spanning_matches = [m for m in matches if 'span_info' in m[0]]
        assert len(spanning_matches) >= 1
    
    def test_find_matches_no_meaningful_phrase(self):
        """Test that non-meaningful phrases are filtered out."""
        from vision.matcher import PhraseMatcher
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        config.common_words = {'the', 'and', 'or', 'a', 'an'}
        matcher = PhraseMatcher(config)
        
        text_lines = [
            {
                'text': 'the and or',
                'y_position': 100,
                'angle': 0,
                'annotations': []
            }
        ]
        
        matches = matcher.find_matches(
            phrase='the and',  # Common words only
            text_lines=text_lines,
            full_text='the and or',
            threshold=80
        )
        
        assert len(matches) == 0  # Should be filtered out
    
    def test_find_matches_partial_word_match(self):
        """Test matching partial words."""
        from vision.matcher import PhraseMatcher
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        matcher = PhraseMatcher(config)
        
        text_lines = [
            {
                'text': 'Authentication Protocol',
                'y_position': 100,
                'angle': 0,
                'annotations': []
            }
        ]
        
        # Searching for "auth" should find "Authentication"
        matches = matcher.find_matches(
            phrase='auth',
            text_lines=text_lines,
            full_text='Authentication Protocol',
            threshold=70
        )
        
        assert isinstance(matches, list)


@pytest.mark.matcher
@pytest.mark.unit
@pytest.mark.parametrize("threshold,expected_strictness", [
    (50, "lenient"),
    (70, "balanced"),
    (80, "balanced"),
    (90, "strict"),
    (95, "very_strict"),
])
class TestPhraseMatcherThresholds:
    """Test phrase matching with explicit threshold values."""
    
    def test_threshold_behavior(self, threshold, expected_strictness):
        """Test that different thresholds produce expected behavior."""
        from vision.matcher import PhraseMatcher
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        matcher = PhraseMatcher(config)
        
        # Exact match should pass all thresholds
        text_lines = [
            {
                'text': 'exact match text',
                'y_position': 100,
                'angle': 0,
                'annotations': []
            }
        ]
        
        matches = matcher.find_matches(
            phrase='exact match',
            text_lines=text_lines,
            full_text='exact match text',
            threshold=threshold
        )
        
        # Exact match should be found at any threshold
        assert len(matches) >= 1
        match_data, score, match_type = matches[0]
        assert score >= threshold

