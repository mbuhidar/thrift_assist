"""Utility functions package."""

from .text_utils import normalize_text_for_search, is_meaningful_phrase
from .geometry_utils import (
    calculate_text_angle,
    rectangles_overlap,
    find_non_overlapping_position
)

__all__ = [
    'normalize_text_for_search',
    'is_meaningful_phrase',
    'calculate_text_angle',
    'rectangles_overlap',
    'find_non_overlapping_position',
]
