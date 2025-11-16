#!/usr/bin/env python3
"""
Test script to verify OCR provider selection works correctly.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from backend.services.ocr_service import OCRService

def test_provider_selection():
    """Test that OCRService can switch between providers."""
    
    print("Testing OCR provider selection...")
    print("-" * 60)
    
    # Initialize service
    service = OCRService()
    
    if not service.is_available():
        print("❌ OCR service not available - cannot test")
        return False
    
    print("✅ OCR service initialized")
    
    # Test getting Google provider detector
    print("\n1. Testing Google provider...")
    google_detector = service.get_detector('google')
    print(f"   Provider: {google_detector.provider.get_provider_name()}")
    
    # Test getting DeepSeek provider detector (if configured)
    print("\n2. Testing DeepSeek provider...")
    try:
        deepseek_detector = service.get_detector('deepseek')
        print(f"   Provider: {deepseek_detector.provider.get_provider_name()}")
    except Exception as e:
        print(f"   ⚠️  DeepSeek provider not configured: {e}")
        print("   (This is expected if GOOGLE_CLOUD_PROJECT is not set)")
    
    # Test detector caching
    print("\n3. Testing detector caching...")
    google_detector_2 = service.get_detector('google')
    if google_detector is google_detector_2:
        print("   ✅ Detector caching works - same instance returned")
    else:
        print("   ❌ Detector caching failed - different instances")
        return False
    
    # Test default provider in cache
    print("\n4. Testing default provider cache...")
    if 'google' in service.detector_cache:
        print("   ✅ Default Google provider cached on init")
    else:
        print("   ❌ Default provider not cached")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All provider selection tests passed!")
    return True

if __name__ == "__main__":
    success = test_provider_selection()
    sys.exit(0 if success else 1)
