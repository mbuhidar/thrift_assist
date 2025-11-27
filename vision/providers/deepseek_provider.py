"""DeepSeek-OCR provider implementation using Google Cloud Vertex AI."""

import os
import base64
import json
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
    
    def _convert_normalized_bbox(
        self, quad: List[int], img_width: int, img_height: int
    ) -> List[Tuple[int, int]]:
        """
        Convert DeepSeek normalized bounding box from [0-999] to pixels.
        
        Args:
            quad: List of 8 integers [x1,y1,x2,y2,x3,y3,x4,y4] in [0-999]
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            List of 4 (x, y) tuples in pixel coordinates
        """
        # DeepSeek returns normalized coordinates in [0-999] range
        # Need to convert to actual pixel coordinates
        pixels = []
        for i in range(0, 8, 2):
            x_norm = quad[i]
            y_norm = quad[i + 1]
            
            # Convert from [0-999] to [0-width] and [0-height]
            x_pixel = int((x_norm / 999.0) * img_width)
            y_pixel = int((y_norm / 999.0) * img_height)
            
            pixels.append((x_pixel, y_pixel))
        
        return pixels
    
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
        
        # Get image dimensions first
        from PIL import Image
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            print(f"📐 Original image: {original_width}x{original_height}")
        
        # Get image file size for verification
        image_size = os.path.getsize(image_path)
        print(f"📷 DeepSeek reading image: {image_path}")
        print(f"   File size: {image_size:,} bytes")
        
        # Initialize Vertex AI (not needed for OpenAPI endpoint)
        # OpenAPI endpoint uses different authentication
        
        try:
            # Use OpenAPI chat completions endpoint for DeepSeek OCR
            import subprocess
            import requests
            import tempfile  # noqa: F401
            
            # Get access token using gcloud with service account
            service_account = (
                'first-project-service-acct@'
                'direct-bonsai-473201-t2.iam.gserviceaccount.com'
            )
            result = subprocess.run(
                ['gcloud', 'auth', 'print-access-token',
                 f'--account={service_account}'],
                capture_output=True,
                text=True,
                check=True
            )
            access_token = result.stdout.strip()
            
            # Resize image if too large for API
            # DeepSeek OCR may have size limits, resize to max 2048px
            import io
            
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Resize if larger than 2048px on longest side
                max_size = 2048
                if max(width, height) > max_size:
                    ratio = max_size / max(width, height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    print(f"📉 Resized to: {new_width}x{new_height}")
                    # Update dimensions for coordinate conversion
                    processing_width = new_width
                    processing_height = new_height
                else:
                    print(f"✓ Image size OK, no resize needed")
                    processing_width = width
                    processing_height = height
                
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
            
            # Verify image data integrity
            import hashlib
            image_hash = hashlib.md5(image_bytes).hexdigest()
            print(f"🔐 Image hash: {image_hash}")
            print(f"📊 First 50 base64 chars: {image_b64[:50]}...")
            print(f"📊 Last 50 base64 chars: ...{image_b64[-50:]}")
            
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
            print(f"📦 Data URL length: {len(image_url):,} chars")
            print(f"📦 Data URL preview: {image_url[:100]}...")
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Request OCR with explicit instructions for comprehensive text
            # DeepSeek needs very specific prompting to extract ALL text
            ocr_prompt = (
                "Please read and extract ALL text visible in this image. "
                "Include:\n"
                "- Book titles on spines (even if rotated or vertical)\n"
                "- Author names\n"
                "- Publisher names\n"
                "- Any labels, stickers, or signs\n"
                "- Small text and large text equally\n"
                "Extract EVERY piece of text you can see, not just the "
                "most prominent elements. List each text element separately."
            )
            payload = {
                "model": self.model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ocr_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high"
                            }
                        }
                ],
                "max_tokens": 8191,  # Max allowed by DeepSeek API
                "temperature": 0.0   # Most deterministic
            }
            ]
            }
            
            print(f"📨 Request: model={self.model_name}, "
                  f"detail=high, format=json")
            
            # Make request
            response = requests.post(url, headers=headers, json=payload)
            
            # Better error handling for common issues
            if response.status_code == 403:
                error_msg = (
                    f"403 Forbidden: DeepSeek API access denied. "
                    f"Possible causes:\n"
                    f"  1. DeepSeek OCR not enabled for project "
                    f"{self.project_id}\n"
                    f"  2. Insufficient permissions for service account\n"
                    f"  3. API endpoint may not be available in region "
                    f"{self.location}\n"
                    f"  4. Check if you need to enable the DeepSeek API "
                    f"in Google Cloud Console"
                )
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            # Print detailed error for 400 Bad Request
            if response.status_code == 400:
                try:
                    error_details = response.json()
                    print(f"❌ 400 Bad Request - API rejected the request")
                    print(f"   Error details: {json.dumps(error_details, indent=2)}")
                except:
                    print(f"❌ 400 Bad Request - Response: {response.text[:500]}")
            
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
            print(f"   First 1000 chars: {content[:1000]}")
            is_json = content.strip().startswith('{')
            print(f"   Appears to be JSON: {is_json}")
            
            # Parse the JSON response with OCR results and bounding boxes
            try:
                ocr_data = json.loads(content)
                print("✅ Successfully parsed JSON response")
            except json.JSONDecodeError as e:
                # Fallback if response is not JSON
                print(f"⚠️ JSON parse error: {e}")
                print(f"⚠️ Error at position {e.pos}")
                pos_start = max(0, e.pos - 50)
                pos_end = e.pos + 50
                print(f"⚠️ Around: ...{content[pos_start:pos_end]}...")
                print("⚠️ Response not JSON, using text-only mode")
                
                # Parse plain text more intelligently
                # DeepSeek may list items with numbers, bullets, etc.
                lines = content.strip().split('\n')
                seen_texts = set()  # Deduplicate
                annotations = []
                y_offset = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Remove common prefixes (numbers, bullets, dashes)
                    cleaned = line
                    # Remove leading numbers like "1. ", "23. "
                    import re
                    cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
                    cleaned = re.sub(r'^[-*•]\s*', '', cleaned)  # bullets
                    cleaned = cleaned.strip()
                    
                    # Skip if empty after cleaning, or if we've seen it
                    if not cleaned or cleaned in seen_texts:
                        continue
                    
                    # Skip very short text unless it looks meaningful
                    if len(cleaned) < 2 and not cleaned.isalnum():
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
                
                full_text = '\n'.join(seen_texts)
                print(f"✅ Extracted {len(annotations)} unique text elements")
                return full_text, annotations
            
            # Extract text and bounding boxes from structured response
            annotations = []
            full_text_parts = []
            
            # Try different possible response formats
            if 'text_elements' in ocr_data:
                # Format: Our requested format with text_elements array
                for item in ocr_data['text_elements']:
                    text = item.get('text', '')
                    bbox_data = item.get('bbox', item.get('box',
                                         item.get('quad')))
                    if bbox_data and len(bbox_data) >= 8:
                        # Normalized [0-999] coordinates
                        bbox = self._convert_normalized_bbox(
                            bbox_data, processing_width, processing_height
                        )
                        annotations.append(TextAnnotation(
                            text=text,
                            confidence=item.get('confidence', 0.95),
                            bounding_box=bbox,
                            locale=None
                        ))
                        full_text_parts.append(text)
            
            elif 'text' in ocr_data and 'boxes' in ocr_data:
                # Format 1: Separate text and boxes arrays
                texts = ocr_data['text']
                boxes = ocr_data['boxes']
                for i, text in enumerate(texts):
                    if i < len(boxes):
                        quad = boxes[i]  # Normalized [0-999] coords
                        bbox = self._convert_normalized_bbox(
                            quad, processing_width, processing_height
                        )
                        annotations.append(TextAnnotation(
                            text=text,
                            confidence=0.95,
                            bounding_box=bbox,
                            locale=None
                        ))
                        full_text_parts.append(text)
            
            elif 'results' in ocr_data:
                # Format 2: Results array with text and box per item
                for item in ocr_data['results']:
                    text = item.get('text', '')
                    quad = item.get('quad', item.get('box',
                                    item.get('bbox')))
                    if quad:
                        bbox = self._convert_normalized_bbox(
                            quad, processing_width, processing_height
                        )
                        annotations.append(TextAnnotation(
                            text=text,
                            confidence=item.get('confidence', 0.95),
                            bounding_box=bbox,
                            locale=None
                        ))
                        full_text_parts.append(text)
            
            elif 'lines' in ocr_data or 'words' in ocr_data:
                # Format 3: Lines or words with embedded boxes
                items = ocr_data.get('lines', ocr_data.get('words', []))
                for item in items:
                    text = item.get('text', '')
                    quad = item.get('quad', item.get('box',
                                    item.get('bbox')))
                    if quad:
                        bbox = self._convert_normalized_bbox(
                            quad, processing_width, processing_height
                        )
                        annotations.append(TextAnnotation(
                            text=text,
                            confidence=item.get('confidence', 0.95),
                            bounding_box=bbox,
                            locale=None
                        ))
                        full_text_parts.append(text)
            
            else:
                # Unknown format, extract any text
                print(f"⚠️ Unknown format: {list(ocr_data.keys())}")
                print(f"⚠️ Full OCR data structure: {ocr_data}")
                
                def extract_text(obj):
                    if isinstance(obj, str):
                        return [obj]
                    elif isinstance(obj, list):
                        result = []
                        for item in obj:
                            result.extend(extract_text(item))
                        return result
                    elif isinstance(obj, dict):
                        result = []
                        for value in obj.values():
                            result.extend(extract_text(value))
                        return result
                    return []
                
                full_text_parts = extract_text(ocr_data)
                
                # Create simple annotations from extracted text
                y_offset = 0
                for text_line in full_text_parts:
                    if text_line and text_line.strip():
                        bounding_box = [
                            (0, y_offset), (100, y_offset),
                            (100, y_offset + 5), (0, y_offset + 5)
                        ]
                        annotations.append(TextAnnotation(
                            text=text_line.strip(),
                            confidence=0.95,
                            bounding_box=bounding_box,
                            locale=None
                        ))
                        y_offset += 10
            
            full_text = '\n'.join(full_text_parts) if full_text_parts else ''
            
            print(f"✅ Created {len(annotations)} annotations")
            print(f"📝 Full text extracted: {len(full_text)} chars")
            
            # If we got text but no annotations, create simple ones
            if full_text and not annotations:
                print("⚠️ Creating fallback annotations from full text")
                lines = full_text.split('\n')
                y_offset = 0
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    bounding_box = [
                        (0, y_offset), (100, y_offset),
                        (100, y_offset + 5), (0, y_offset + 5)
                    ]
                    annotations.append(TextAnnotation(
                        text=line,
                        confidence=0.95,
                        bounding_box=bounding_box,
                        locale=None
                    ))
                    y_offset += 10
                print(f"✅ Created {len(annotations)} fallback annotations")
            
            return full_text, annotations
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"DeepSeek OCR via Google Cloud error: {e}")
