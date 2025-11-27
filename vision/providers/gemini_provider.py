"""Gemini Vision provider implementation using Google Cloud Vertex AI."""

import os
import base64
import json
from typing import List, Tuple
from PIL import Image

from .base_provider import OCRProvider, TextAnnotation

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    vertexai = None
    GenerativeModel = None
    Part = None


class GeminiProvider(OCRProvider):
    """Gemini Vision provider using Google Cloud Vertex AI."""
    
    def __init__(self, project_id: str = None, location: str = "us-central1",
                 model_name: str = None):
        """
        Initialize Gemini Vision provider with Google Cloud.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region (default: us-central1)
            model_name: Gemini model (default: gemini-1.5-flash-001)
        """
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        self.location = location or os.getenv('GOOGLE_CLOUD_LOCATION', 
                                             'us-central1')
        self.model_name = model_name or os.getenv('GEMINI_MODEL',
                                                   'gemini-1.5-flash-001')
        self._model = None
    
    @property
    def name(self) -> str:
        """Return provider name."""
        return "Gemini Vision (Google Cloud)"
    
    def is_available(self) -> bool:
        """Check if Gemini on Google Cloud is available and configured."""
        if not VERTEX_AI_AVAILABLE:
            return False
        
        return self.project_id is not None
    
    def _get_model(self):
        """Get or create Gemini model."""
        if self._model is None:
            vertexai.init(project=self.project_id, location=self.location)
            self._model = GenerativeModel(self.model_name)
        return self._model
    
    def _prepare_image(self, image_path: str) -> Tuple[bytes, int, int]:
        """
        Prepare image for Gemini API.
        
        Returns:
            Tuple of (image_bytes, width, height)
        """
        # Open image to get dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            
            # Resize if too large (Gemini has size limits)
            max_dimension = 3072
            if max(img_width, img_height) > max_dimension:
                if img_width > img_height:
                    new_width = max_dimension
                    new_height = int(img_height * (max_dimension / img_width))
                else:
                    new_height = max_dimension
                    new_width = int(img_width * (max_dimension / img_height))
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_width, img_height = new_width, new_height
                print(f"📐 Resized image to {img_width}×{img_height}")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
        
        print(f"✅ Prepared image: {len(image_bytes):,} bytes")
        return image_bytes, img_width, img_height

    def detect_text(self, image_path: str) -> Tuple[str, List[TextAnnotation]]:
        """
        Detect text using Gemini Vision via Google Cloud Vertex AI.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (full_text, annotations)
        """
        if not self.is_available():
            raise RuntimeError(
                "Gemini Vision not available. Install: "
                "pip install google-cloud-aiplatform"
            )
        
        try:
            print(f"🔧 Gemini config: project_id='{self.project_id}', "
                  f"location='{self.location}', model='{self.model_name}'")
            print(f"✅ Using OCR provider: {self.name}")
            
            # Get image dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size
                print(f"📐 Original image: {img_width}x{img_height}")
            
            # Prepare image
            print(f"📷 Gemini reading image: {image_path}")
            file_size = os.path.getsize(image_path)
            print(f"   File size: {file_size:,} bytes")
            
            image_bytes, processing_width, processing_height = (
                self._prepare_image(image_path)
            )
            
            # Create image part for Gemini
            image_part = Part.from_data(
                data=image_bytes,
                mime_type="image/jpeg"
            )
            
            # Construct detailed OCR prompt
            prompt = (
                "Extract ALL visible text from this image using OCR. "
                "This is a bookshelf with book spines. Extract:\n"
                "- Book titles (even if rotated, vertical, or at angles)\n"
                "- Author names on spines\n"
                "- Publisher names\n"
                "- Any labels, price stickers, or signs\n"
                "- All text regardless of size, orientation, or prominence\n\n"
                "Return ONLY a JSON array with this exact format:\n"
                '{"texts": [{"text": "extracted text 1"}, {"text": "extracted text 2"}, ...]}\n\n'
                "Extract every piece of text you can see. Be thorough."
            )
            
            print(f"📨 Sending request to Gemini {self.model_name}...")
            
            # Get model and generate content
            model = self._get_model()
            response = model.generate_content(
                [prompt, image_part],
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 8192,
                }
            )
            
            # Extract response text
            if not response or not response.text:
                print("❌ No response from Gemini")
                return "", []
            
            content = response.text.strip()
            print(f"✅ Gemini response length: {len(content)} characters")
            print(f"🔍 First 500 chars: {content[:500]}")
            
            # Parse JSON response
            annotations = []
            try:
                # Try to extract JSON from response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    data = json.loads(json_str)
                    
                    # Extract texts from various possible formats
                    texts = []
                    if 'texts' in data:
                        texts = data['texts']
                    elif 'text_elements' in data:
                        texts = data['text_elements']
                    elif 'items' in data:
                        texts = data['items']
                    
                    # Convert to annotations
                    seen_texts = set()
                    y_offset = 0
                    for item in texts:
                        text = item.get('text', '').strip()
                        if text and text not in seen_texts and len(text) > 1:
                            seen_texts.add(text)
                            # Create simple bounding box (Gemini doesn't provide coords)
                            bounding_box = [
                                (0, y_offset), (100, y_offset),
                                (100, y_offset + 5), (0, y_offset + 5)
                            ]
                            annotations.append(TextAnnotation(
                                text=text,
                                confidence=0.95,
                                bounding_box=bounding_box,
                                locale=None
                            ))
                            y_offset += 10
                    
                    print(f"✅ Parsed {len(annotations)} text elements from JSON")
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parse error: {e}")
                print("ℹ️ Falling back to plain text parsing")
            
            # Fallback: parse as plain text
            if not annotations:
                import re
                lines = content.split('\n')
                seen_texts = set()
                y_offset = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Remove common prefixes (bullets, numbers)
                    cleaned = re.sub(r'^\d+\.\s*', '', line)
                    cleaned = re.sub(r'^[-*•]\s*', '', cleaned)
                    cleaned = cleaned.strip()
                    
                    # Skip markdown, code blocks, etc.
                    if cleaned.startswith('#') or cleaned.startswith('```'):
                        continue
                    
                    # Skip if too short or already seen
                    if len(cleaned) < 2 or cleaned in seen_texts:
                        continue
                    
                    seen_texts.add(cleaned)
                    bounding_box = [
                        (0, y_offset), (100, y_offset),
                        (100, y_offset + 5), (0, y_offset + 5)
                    ]
                    annotations.append(TextAnnotation(
                        text=cleaned,
                        confidence=0.95,
                        bounding_box=bounding_box,
                        locale=None
                    ))
                    y_offset += 10
                
                print(f"✅ Extracted {len(annotations)} unique text elements")
            
            # Combine all text
            full_text = '\n'.join(ann.text for ann in annotations)
            
            return full_text, annotations
            
        except Exception as e:
            error_msg = f"Gemini Vision error: {str(e)}"
            print(f"❌ OCR detection error: {error_msg}")
            raise Exception(error_msg)
