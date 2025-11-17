#!/usr/bin/env python3
"""
Test DeepSeek OCR with actual image using OpenAPI endpoint.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.vision_config import VisionConfig
from vision.providers.deepseek_provider import DeepSeekProvider

# Find a test image
test_image = None
for path in ['temp/test_image.jpg', 'image/test.jpg', 'public/sample.jpg']:
    if os.path.exists(path):
        test_image = path
        break

if not test_image:
    print("❌ No test image found. Please provide an image path.")
    print("Usage: python test_deepseek_ocr.py [image_path]")
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        sys.exit(1)

if not os.path.exists(test_image):
    print(f"❌ Image not found: {test_image}")
    sys.exit(1)

print("=" * 60)
print("Testing DeepSeek OCR with OpenAPI Endpoint")
print("=" * 60)
print(f"Image: {test_image}")
print()

# Load config
config = VisionConfig()

# Create provider
provider = DeepSeekProvider(
    project_id=config.google_cloud_project,
    location=config.google_cloud_location,
    endpoint=config.google_cloud_endpoint,
    model_name=config.deepseek_model
)

print(f"Provider: {provider.name}")
print(f"Model: {provider.model_name}")
print(f"Location: {provider.location}")
print(f"Project: {provider.project_id}")
print()

if not provider.is_available():
    print("❌ Provider not available - check configuration")
    sys.exit(1)

print("✅ Provider is available")
print()
print("Calling DeepSeek OCR API...")
print()

try:
    full_text, annotations = provider.detect_text(test_image)
    
    print("=" * 60)
    print("SUCCESS! OCR Results:")
    print("=" * 60)
    print()
    print(f"Full text ({len(full_text)} characters):")
    print("-" * 60)
    print(full_text)
    print("-" * 60)
    print()
    print(f"Detected {len(annotations)} text elements")
    print()
    print("First 10 elements:")
    for i, ann in enumerate(annotations[:10], 1):
        print(f"  {i}. '{ann.text}' (confidence: {ann.confidence:.2f})")
    
    if len(annotations) > 10:
        print(f"  ... and {len(annotations) - 10} more")
    
    print()
    print("✅ DeepSeek OCR is working correctly!")
    
except Exception as e:
    print("=" * 60)
    print("ERROR calling DeepSeek OCR:")
    print("=" * 60)
    print(f"{type(e).__name__}: {e}")
    print()
    
    import traceback
    traceback.print_exc()
    
    print()
    print("Possible issues:")
    print("1. Check that 'gcloud auth' is configured")
    print("2. Verify the model is enabled in your project")
    print("3. Ensure you have the correct permissions")
    sys.exit(1)
