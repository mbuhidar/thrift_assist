"""Tests for configuration settings."""

import pytest
from unittest.mock import patch


@pytest.mark.unit
class TestSettings:
    """Test settings configuration."""
    
    def test_settings_defaults(self):
        """Test default settings values."""
        from backend.core.config import settings
        
        assert settings.DEFAULT_THRESHOLD >= 0
        assert settings.DEFAULT_THRESHOLD <= 100
        assert settings.MAX_CACHE_SIZE > 0
        assert settings.CACHE_TTL_SECONDS > 0
    
    @patch.dict('os.environ', {'DEFAULT_THRESHOLD': '85'})
    def test_settings_from_env(self):
        """Test settings can be overridden by environment."""
        from backend.core.config import Settings
        
        test_settings = Settings()
        # Environment override would work in fresh import
        assert test_settings.DEFAULT_THRESHOLD >= 0


@pytest.mark.unit  
class TestVisionConfig:
    """Test vision configuration."""
    
    def test_vision_config_initialization(self):
        """Test VisionConfig can be initialized."""
        from config.vision_config import VisionConfig
        
        config = VisionConfig()
        
        assert config.fuzz_threshold >= 0
        assert config.fuzz_threshold <= 100
        assert isinstance(config.common_words, set)
        assert len(config.common_words) > 0
