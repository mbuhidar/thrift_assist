"""
Test client for OCR Phrase Detection Web Service
"""

import requests
import base64
import json
import os

# Configuration
BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "image/iCloud_Photos/IMG_4918.JPEG"  # Update with your test image path

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_detect_phrases():
    """Test phrase detection with base64 image."""
    print("🔍 Testing phrase detection...")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
        return
    
    # Encode image
    image_base64 = encode_image_to_base64(TEST_IMAGE_PATH)
    
    # Prepare request
    payload = {
        "search_phrases": ["Homecoming", "Lee Child", "Circle of Three"],
        "threshold": 80,
        "image_base64": image_base64
    }
    
    # Send request
    response = requests.post(f"{BASE_URL}/detect-phrases", json=payload)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Success: {result.get('success')}")
    print(f"Total matches: {result.get('total_matches')}")
    print(f"Processing time: {result.get('processing_time_ms')}ms")
    
    if result.get('matches'):
        print("Matches:")
        for phrase, matches in result['matches'].items():
            print(f"  {phrase}: {len(matches)} matches")
            for match in matches:
                print(f"    - {match['text']} ({match['score']:.1f}% {match['match_type']})")
    print()

def test_upload_and_detect():
    """Test file upload and phrase detection."""
    print("🔍 Testing file upload and detection...")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
        return
    
    # Prepare form data
    with open(TEST_IMAGE_PATH, "rb") as image_file:
        files = {"file": (os.path.basename(TEST_IMAGE_PATH), image_file, "image/jpeg")}
        data = {
            "search_phrases": json.dumps(["Homecoming", "Lee Child", "Circle of Three"]),
            "threshold": 80
        }
        
        # Send request
        response = requests.post(f"{BASE_URL}/upload-and-detect", files=files, data=data)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Success: {result.get('success')}")
    print(f"Total matches: {result.get('total_matches')}")
    print(f"Processing time: {result.get('processing_time_ms')}ms")
    print(f"Filename: {result.get('filename')}")
    print()

def test_detect_with_annotation():
    """Test phrase detection with annotated image return."""
    print("🔍 Testing phrase detection with annotation...")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
        return
    
    # Encode image
    image_base64 = encode_image_to_base64(TEST_IMAGE_PATH)
    
    # Prepare request
    payload = {
        "search_phrases": ["Homecoming", "Lee Child"],
        "threshold": 80,
        "image_base64": image_base64
    }
    
    # Send request
    response = requests.post(f"{BASE_URL}/detect-phrases-with-annotation", json=payload)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Success: {result.get('success')}")
    print(f"Total matches: {result.get('total_matches')}")
    print(f"Has annotated image: {'annotated_image_base64' in result}")
    print(f"Processing time: {result.get('processing_time_ms')}ms")
    print()

if __name__ == "__main__":
    print("🧪 Testing OCR Phrase Detection Web Service")
    print("=" * 50)
    
    try:
        test_health_check()
        test_detect_phrases()
        test_upload_and_detect()
        test_detect_with_annotation()
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Make sure the web service is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")
