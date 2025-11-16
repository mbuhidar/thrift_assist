"""Base OCR provider interface for multiple vision APIs."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class TextAnnotation:
    """Standardized text annotation from any OCR provider."""
    text: str
    confidence: float
    bounding_box: List[Tuple[int, int]]  # List of (x, y) coordinates
    locale: str = None


class OCRProvider(ABC):
    """Abstract base class for OCR providers."""
    
    @abstractmethod
    def detect_text(self, image_path: str) -> Tuple[str, List[TextAnnotation]]:
        """
        Detect text in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (full_text, list of text annotations)
            - full_text: All detected text concatenated
            - annotations: List of TextAnnotation objects with bounding boxes
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this provider is properly configured and available.
        
        Returns:
            True if provider can be used, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass
