"""Google Cloud Vision OCR provider implementation."""

import os
from typing import List, Tuple
from contextlib import contextmanager

from .base_provider import OCRProvider, TextAnnotation

try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    vision = None


@contextmanager
def suppress_stderr_warnings():
    """Suppress GRPC warnings to stderr."""
    import sys
    import io
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


class GoogleVisionProvider(OCRProvider):
    """Google Cloud Vision API provider."""
    
    def __init__(self, credentials_path: str = None):
        """
        Initialize Google Vision provider.
        
        Args:
            credentials_path: Path to Google Cloud credentials JSON file
        """
        self.credentials_path = credentials_path
        self._client = None
        
        # Set credentials if provided
        if credentials_path and os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    
    @property
    def name(self) -> str:
        """Return provider name."""
        return "Google Cloud Vision"
    
    def is_available(self) -> bool:
        """Check if Google Vision is available and configured."""
        if not GOOGLE_VISION_AVAILABLE:
            return False
        
        # Check for credentials
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            return False
        
        creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        return os.path.exists(creds_path)
    
    def _get_client(self):
        """Get or create Vision API client."""
        if self._client is None:
            with suppress_stderr_warnings():
                self._client = vision.ImageAnnotatorClient()
        return self._client
    
    def detect_text(self, image_path: str) -> Tuple[str, List[TextAnnotation]]:
        """
        Detect text using Google Cloud Vision API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (full_text, list of text annotations)
        """
        if not GOOGLE_VISION_AVAILABLE:
            raise ImportError("Google Cloud Vision library not available. Install with: pip install google-cloud-vision")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        with open(image_path, 'rb') as f:
            content = f.read()
        
        image = vision.Image(content=content)
        client = self._get_client()
        
        # Try document text detection first (better for structured text)
        response = client.document_text_detection(image=image)
        
        if not response.text_annotations:
            print("📐 No text found with document detection, trying basic detection...")
            response = client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")
        
        # Convert to standardized format
        annotations = []
        full_text = ""
        
        if response.text_annotations:
            # First annotation contains all text
            full_text = response.text_annotations[0].description
            
            # Skip first annotation (it's the full text), process individual annotations
            for annotation in response.text_annotations[1:]:
                # Extract bounding box vertices
                vertices = annotation.bounding_poly.vertices
                bounding_box = [(v.x, v.y) for v in vertices]
                
                text_ann = TextAnnotation(
                    text=annotation.description,
                    confidence=1.0,  # Google Vision doesn't provide per-word confidence
                    bounding_box=bounding_box,
                    locale=annotation.locale if hasattr(annotation, 'locale') else None
                )
                annotations.append(text_ann)
        
        return full_text, annotations
