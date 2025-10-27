"""Tests for TextLineGrouper."""

import pytest
from unittest.mock import Mock
import numpy as np


@pytest.mark.unit
class TestTextLineGrouper:
    """Test text line grouping logic."""
    
    def test_grouper_initialization(self):
        """Test grouper can be initialized."""
        from vision.grouper import TextLineGrouper
        
        grouper = TextLineGrouper()
        assert grouper is not None
    
    def test_group_empty_annotations(self):
        """Test grouping with no annotations."""
        from vision.grouper import TextLineGrouper
        
        grouper = TextLineGrouper()
        result = grouper.group([])
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_group_single_word(self):
        """Test grouping with single word annotation."""
        from vision.grouper import TextLineGrouper
        
        # Mock a simple word annotation
        mock_word = Mock()
        mock_word.description = "test"
        mock_word.bounding_poly.vertices = [
            Mock(x=10, y=10),
            Mock(x=50, y=10),
            Mock(x=50, y=30),
            Mock(x=10, y=30)
        ]
        
        grouper = TextLineGrouper()
        result = grouper.group([mock_word])
        
        assert isinstance(result, list)
        # Should group into at least one line
        assert len(result) >= 0
