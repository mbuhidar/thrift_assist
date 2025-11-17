"""DeepSeek-OCR provider implementation using Google Cloud Vertex AI."""

import os
import base64
from typing import List, Tuple

from .base_provider import OCRProvider, TextAnnotation

try:
    from google.cloud import aiplatform
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    aiplatform = None


class DeepSeekProvider(OCRProvider):
    """DeepSeek-OCR provider using Google Cloud Vertex AI."""
    
    def __init__(self, project_id: str = None, location: str = "global",
                 endpoint: str = None, model_name: str = None):
        """
        Initialize DeepSeek OCR provider with Google Cloud.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region (default: global)
            endpoint: API endpoint (default: aiplatform.googleapis.com)
            model_name: DeepSeek model (default: deepseek-ai/deepseek-ocr-maas)
        """
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        self.location = location or os.getenv('GOOGLE_CLOUD_LOCATION', 'global')
        self.endpoint = endpoint or os.getenv('GOOGLE_CLOUD_ENDPOINT',
                                             'aiplatform.googleapis.com')
        model_default = 'deepseek-ai/deepseek-ocr-maas'
        self.model_name = model_name or os.getenv('DEEPSEEK_MODEL',
                                                   model_default)
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
        
        # Get image file size for verification
        image_size = os.path.getsize(image_path)
        print(f"📷 DeepSeek reading image: {image_path}")
        print(f"   File size: {image_size:,} bytes")
        
        # Verify this is the same image by checking first few bytes
        with open(image_path, 'rb') as f:
            first_bytes = f.read(16)
            import hashlib
            f.seek(0)
            img_hash = hashlib.md5(f.read()).hexdigest()
        print(f"   MD5 hash: {img_hash}")
        print(f"   First bytes: {first_bytes[:8].hex()}")
        
        # Initialize Vertex AI (not needed for OpenAPI endpoint)
        # OpenAPI endpoint uses different authentication
        
        try:
            # Use OpenAPI chat completions endpoint for DeepSeek OCR
            import subprocess
            import requests
            import tempfile  # noqa: F401
            
            # Get access token using gcloud
            result = subprocess.run(
                ['gcloud', 'auth', 'print-access-token'],
                capture_output=True,
                text=True,
                check=True
            )
            access_token = result.stdout.strip()
            
            # Resize image if too large for API
            # DeepSeek OCR may have size limits, resize to max 2048px
            from PIL import Image
            import io
            
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"📐 Original image: {width}x{height}")
                
                # Resize if larger than 2048px on longest side
                max_size = 2048
                if max(width, height) > max_size:
                    ratio = max_size / max(width, height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    print(f"📉 Resized to: {new_width}x{new_height}")
                else:
                    print(f"✓ Image size OK, no resize needed")
                
                # Convert to JPEG with quality 85 to reduce size
                img_buffer = io.BytesIO()
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Convert to RGB for JPEG
                    img = img.convert('RGB')
                img.save(img_buffer, format='JPEG', quality=85, optimize=True)
                image_bytes = img_buffer.getvalue()
            
            print(f"✅ Prepared image: {len(image_bytes):,} bytes")
            
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            print(f"✅ Encoded to base64: {len(image_b64):,} chars")
            
            # Determine image type for proper MIME type
            if image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            elif (image_path.lower().endswith('.jpg') or
                  image_path.lower().endswith('.jpeg')):
                mime_type = 'image/jpeg'
            else:
                mime_type = 'image/jpeg'
            
            # Create properly formatted data URL for vision models
            image_url = f"data:{mime_type};base64,{image_b64}"
            
            # Prepare request to OpenAPI endpoint
            url = (
                f"https://{self.endpoint}/v1/projects/{self.project_id}/"
                f"locations/{self.location}/endpoints/openapi/chat/completions"
            )
            
            print(f"🌐 API endpoint: {url}")
            print(f"📤 Sending {mime_type} as data URL")
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Match OpenAI vision API format exactly
            # Reference: https://platform.openai.com/docs/guides/vision
            # Add detailed OCR instructions to extract ALL text
            payload = {
                "model": self.model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ("Please perform OCR on this image. "
                                   "Extract ALL visible text including: "
                                   "book titles, author names, labels, "
                                   "signs, prices, and any other text. "
                                   "List every piece of text you can see, "
                                   "no matter how small.")
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high"
                            }
                        }
                    ]
                }],
                "max_tokens": 4096,
                "temperature": 0.1
            }
            
            print(f"📨 Request: model={self.model_name}, "
                  f"detail=high, temp=0.1")
            
            # Make request
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result_data = response.json()
            
            # Parse OpenAPI chat completion response
            # Response format: {"choices": [{"message": {"content": "..."}}]}
            if 'choices' not in result_data or not result_data['choices']:
                raise ValueError("No response from DeepSeek OCR model")
            
            content = result_data['choices'][0]['message']['content']
            
            # Debug: print what DeepSeek returned
            print("🔍 DeepSeek OCR raw response:")
            print(f"   Content length: {len(content)} chars")
            print(f"   First 500 chars: {content[:500]}")
            
            # The model returns the extracted text directly
            # DeepSeek OCR returns plain text, not JSON structured data
            full_text = content.strip()
            
            print(f"📝 Full text extracted: {len(full_text)} chars")
            print(f"   Preview: {full_text[:200]}")
            
            # Create simple annotations from the extracted text
            # Split into words/phrases for annotation
            
            annotations = []
            
            # Split by whitespace and newlines to get individual text elements
            # Create annotations for both full lines AND individual words
            # to improve phrase matching
            lines = content.strip().split('\n')
            
            # Give each line a unique Y-coordinate so grouper treats as separate
            y_offset = 0
            y_increment = 10  # Pixels between lines
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Create annotation with unique Y-coordinate for each line
                bounding_box = [
                    (0, y_offset),           # top-left
                    (100, y_offset),         # top-right
                    (100, y_offset + 5),     # bottom-right
                    (0, y_offset + 5)        # bottom-left
                ]
                
                text_ann = TextAnnotation(
                    text=line,
                    confidence=0.95,
                    bounding_box=bounding_box,
                    locale=None
                )
                annotations.append(text_ann)
                
                # Also create word-level annotations for better matching
                import re
                words = re.findall(r'\S+', line)
                x_offset = 0
                x_increment = 100 / max(len(words), 1)
                
                for word in words:
                    word_bbox = [
                        (x_offset, y_offset),
                        (x_offset + x_increment, y_offset),
                        (x_offset + x_increment, y_offset + 5),
                        (x_offset, y_offset + 5)
                    ]
                    word_ann = TextAnnotation(
                        text=word,
                        confidence=0.95,
                        bounding_box=word_bbox,
                        locale=None
                    )
                    annotations.append(word_ann)
                    x_offset += x_increment
                
                y_offset += y_increment
            
            num_lines = len(lines)
            num_ann = len(annotations)
            print(f"✅ Created {num_ann} annotations from {num_lines} lines")
            
            return full_text, annotations
            
        except Exception as e:
            raise Exception(f"DeepSeek OCR via Google Cloud error: {e}")
