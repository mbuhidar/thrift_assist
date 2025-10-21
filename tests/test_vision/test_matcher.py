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
