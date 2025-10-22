"""Tests for credentials handling."""

import pytest
from unittest.mock import patch, Mock
import os


@pytest.mark.unit
class TestCredentials:
    """Test credential setup and validation."""
    
    @patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': '/path/to/creds.json'})
    def test_credentials_from_env(self):
        """Test credentials path from environment."""
        assert os.getenv('GOOGLE_APPLICATION_CREDENTIALS') == '/path/to/creds.json'
    
    def test_credentials_module_imports(self):
        """Test credentials module can be imported."""
        try:
            from backend.core import credentials
            assert credentials is not None
        except ImportError:
            pytest.fail("backend.core.credentials should be importable")
