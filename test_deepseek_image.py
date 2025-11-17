#!/usr/bin/env python3
"""
Test DeepSeek OCR API with any image.
Usage: python test_deepseek_image.py path/to/image.jpg
"""

import os
import sys
import base64
import subprocess
import json
from PIL import Image
import io

def test_deepseek_ocr(image_path):
    """Send an image to DeepSeek OCR API and print the response."""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("DeepSeek OCR API Test")
    print("=" * 70)
    print(f"📷 Image: {image_path}")
    
    # Configuration
    project_id = "direct-bonsai-473201-t2"
    location = "global"
    endpoint = "aiplatform.googleapis.com"
    model = "deepseek-ai/deepseek-ocr-maas"
    
    print(f"🔧 Project: {project_id}")
    print(f"🌍 Location: {location}")
    print(f"🤖 Model: {model}")
    print()
    
    # Get access token
    print("🔑 Getting access token...")
    result = subprocess.run(
        ['gcloud', 'auth', 'print-access-token'],
        capture_output=True,
        text=True,
        check=True
    )
    access_token = result.stdout.strip()
    print("✅ Access token obtained")
    print()
    
    # Load and prepare image
    print("📐 Processing image...")
    with Image.open(image_path) as img:
        width, height = img.size
        print(f"   Original size: {width}x{height}")
        
        # Resize if too large
        max_size = 2048
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            print(f"   Resized to: {new_width}x{new_height}")
        
        # Convert to JPEG
        img_buffer = io.BytesIO()
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        img.save(img_buffer, format='JPEG', quality=85, optimize=True)
        image_bytes = img_buffer.getvalue()
    
    file_size = len(image_bytes)
    print(f"   Prepared: {file_size:,} bytes")
    
    # Encode to base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    print(f"   Base64: {len(image_b64):,} chars")
    print()
    
    # Create data URL
    image_url = f"data:image/jpeg;base64,{image_b64}"
    
    # Prepare API request
    url = (
        f"https://{endpoint}/v1/projects/{project_id}/"
        f"locations/{location}/endpoints/openapi/chat/completions"
    )
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": model,
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
    
    print("📤 Sending request to DeepSeek OCR API...")
    print(f"   URL: {url}")
    print(f"   Payload size: {len(json.dumps(payload)):,} bytes")
    print()
    
    # Make request
    import requests
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        print("=" * 70)
        print("✅ SUCCESS - API Response:")
        print("=" * 70)
        print()
        
        # Print full response for debugging
        print("📋 Full JSON Response:")
        print(json.dumps(result, indent=2))
        print()
        
        # Extract and display the OCR text
        if 'choices' in result and result['choices']:
            content = result['choices'][0]['message']['content']
            print("=" * 70)
            print("📝 Extracted Text:")
            print("=" * 70)
            print(content)
            print()
            print(f"📊 Text length: {len(content)} characters")
            print(f"📊 Lines: {len(content.splitlines())}")
        else:
            print("⚠️  No text content in response")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print("=" * 70)
        print("❌ ERROR - Request Failed:")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print()
            print("Response status:", e.response.status_code)
            print("Response body:", e.response.text)
        return None
    except Exception as e:
        print("=" * 70)
        print("❌ ERROR - Unexpected:")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_deepseek_image.py <image_path>")
        print()
        print("Example:")
        print("  python test_deepseek_image.py /path/to/your/image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = test_deepseek_ocr(image_path)
    
    if result:
        print()
        print("=" * 70)
        print("✅ Test completed successfully!")
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("❌ Test failed")
        print("=" * 70)
        sys.exit(1)
