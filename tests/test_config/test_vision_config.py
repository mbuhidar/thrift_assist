"""Tests for vision configuration."""

import pytest


@pytest.mark.unit
class TestVisionConfigDetails:
    """Test vision configuration details."""
    
    def test_config_has_common_words(self):
        """Test config includes common words set."""
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        
        assert hasattr(config, 'common_words')
        assert 'the' in config.common_words
        assert 'and' in config.common_words
        assert 'a' in config.common_words
    
    def test_config_thresholds(self):
        """Test config threshold values are reasonable."""
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        
        # Fuzzy threshold should be 0-100
        assert 0 <= config.fuzz_threshold <= 100
        
        # Angle tolerance should be reasonable
        if hasattr(config, 'angle_tolerance'):
            assert 0 <= config.angle_tolerance <= 180
    
    def test_config_text_scale(self):
        """Test config text scale is reasonable."""
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        
        if hasattr(config, 'default_text_scale'):
            assert 50 <= config.default_text_scale <= 200
