"""OCR Provider abstraction for multiple vision APIs."""

from .base_provider import OCRProvider
from .google_vision_provider import GoogleVisionProvider

try:
    from .gemini_provider import GeminiProvider
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    GeminiProvider = None

__all__ = [
    'OCRProvider',
    'GoogleVisionProvider',
    'GeminiProvider',
    'GEMINI_AVAILABLE'
]
