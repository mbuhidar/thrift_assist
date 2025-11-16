"""OCR Provider abstraction for multiple vision APIs."""

from .base_provider import OCRProvider
from .google_vision_provider import GoogleVisionProvider

try:
    from .deepseek_provider import DeepSeekProvider
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    DeepSeekProvider = None

__all__ = [
    'OCRProvider',
    'GoogleVisionProvider',
    'DeepSeekProvider',
    'DEEPSEEK_AVAILABLE'
]
