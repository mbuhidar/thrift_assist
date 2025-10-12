"""
Vision OCR package for phrase detection and annotation.
Provides Google Cloud Vision API integration with fuzzy matching.
"""

from .detector import VisionPhraseDetector
from .annotator import ImageAnnotator
from .matcher import PhraseMatcher
from .grouper import TextLineGrouper

__all__ = [
    'VisionPhraseDetector',
    'ImageAnnotator',
    'PhraseMatcher',
    'TextLineGrouper',
]
