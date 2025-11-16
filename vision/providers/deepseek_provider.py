"""DeepSeek-OCR provider implementation using Google Cloud AI."""

import os
import base64
from typing import List, Tuple

from .base_provider import OCRProvider, TextAnnotation

try:
    from google.cloud import aiplatform
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    aiplatform = None


class DeepSeekProvider(OCRProvider):
    """DeepSeek-OCR provider using Google Cloud Vertex AI."""
    
    def __init__(self, project_id: str = None, location: str = "global", endpoint: str = None):
        """
        Initialize DeepSeek provider with Google Cloud.
        
        Args:
            project_id: Google Cloud project ID (or set GOOGLE_CLOUD_PROJECT)
            location: Google Cloud region (default: global)
            endpoint: API endpoint (default: aiplatform.googleapis.com)
        """
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        self.location = location or os.getenv('GOOGLE_CLOUD_LOCATION', 'global')
        self.endpoint = endpoint or os.getenv('GOOGLE_CLOUD_ENDPOINT', 'aiplatform.googleapis.com')
        self._client = None
    
    @property
    def name(self) -> str:
        """Return provider name."""
        return "DeepSeek-OCR (Google Cloud)"
    
    def is_available(self) -> bool:
        """Check if DeepSeek on Google Cloud is available and configured."""
        if not GOOGLE_AI_AVAILABLE:
            return False
        
        return self.project_id is not None
    
    def _get_client(self):
        """Get or create Vertex AI client."""
        if self._client is None:
            aiplatform.init(
                project=self.project_id,
                location=self.location,
                api_endpoint=self.endpoint
            )
        return self._client
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def detect_text(self, image_path: str) -> Tuple[str, List[TextAnnotation]]:
        """
        Detect text using DeepSeek-OCR via Google Cloud Vertex AI.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (full_text, list of text annotations)
            
        Note:
            Uses Google Cloud's Vertex AI to access DeepSeek vision model.
        """
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError(
                "google-cloud-aiplatform library not available. "
                "Install with: pip install google-cloud-aiplatform"
            )
        
        if not self.project_id:
            raise ValueError(
                "Google Cloud project ID not set. Set GOOGLE_CLOUD_PROJECT "
                "environment variable or pass project_id parameter."
            )
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Encode image
        image_base64 = self._encode_image(image_path)
        
        # Initialize Vertex AI
        self._get_client()
        
        # Prepare the prediction request
        from vertexai.preview.vision_models import ImageTextModel
        
        try:
            # Load the DeepSeek model from Vertex AI Model Garden
            model = ImageTextModel.from_pretrained("deepseek-ai/deepseek-vl2")
            
            # Create prompt for OCR with structured output
            prompt = """Extract all text from this image. 
For each text element, provide:
1. The text content
2. Bounding box coordinates (x1, y1, x2, y2) as percentages (0-100)

Format as JSON:
{
  "full_text": "all text concatenated",
  "text_elements": [
    {"text": "word or phrase", "bbox": [x1, y1, x2, y2]}
  ]
}"""
            
            # Get predictions
            response = model.predict(
                prompt=prompt,
                image=f"data:image/jpeg;base64,{image_base64}",
                temperature=0.1,
                max_output_tokens=4096
            )
            
            # Parse response
            import json
            import re
            
            content = response.text
            
            # Extract JSON from markdown code blocks if present
            json_match = re.search(
                r'```json\s*(.*?)\s*```', content, re.DOTALL
            )
            if json_match:
                content = json_match.group(1)
            
            ocr_result = json.loads(content)
            
            # Extract full text
            full_text = ocr_result.get('full_text', '')
            
            # Convert to standardized annotations
            annotations = []
            for element in ocr_result.get('text_elements', []):
                text = element.get('text', '')
                bbox_pct = element.get('bbox', [0, 0, 100, 100])
                
                # Convert percentage coordinates
                bounding_box = [
                    (bbox_pct[0], bbox_pct[1]),  # top-left
                    (bbox_pct[2], bbox_pct[1]),  # top-right
                    (bbox_pct[2], bbox_pct[3]),  # bottom-right
                    (bbox_pct[0], bbox_pct[3])   # bottom-left
                ]
                
                text_ann = TextAnnotation(
                    text=text,
                    confidence=0.9,
                    bounding_box=bounding_box,
                    locale=None
                )
                annotations.append(text_ann)
            
            return full_text, annotations
            
        except Exception as e:
            raise Exception(f"DeepSeek OCR via Google Cloud error: {e}")
