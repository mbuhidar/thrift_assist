"""Test script for OCR provider functionality."""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set OCR provider before importing
os.environ['OCR_PROVIDER'] = sys.argv[1] if len(sys.argv) > 1 else 'google'

from config.vision_config import VisionConfig
from vision.detector import VisionPhraseDetector


def test_provider_initialization():
    """Test that providers initialize correctly."""
    print(f"\n{'='*60}")
    print(f"Testing OCR Provider: {os.getenv('OCR_PROVIDER', 'google')}")
    print(f"{'='*60}\n")
    
    try:
        # Create configuration
        config = VisionConfig()
        print(f"✓ Created VisionConfig")
        print(f"  - Provider: {config.ocr_provider}")
        print(f"  - Google credentials: {config.google_credentials_path}")
        
        # Create detector
        detector = VisionPhraseDetector(config)
        print(f"\n✓ Created VisionPhraseDetector")
        print(f"  - Provider name: {detector.provider.name}")
        print(f"  - Provider available: {detector.provider.is_available()}")
        
        if not detector.provider.is_available():
            print(f"\n⚠️  Warning: Provider is not available!")
            print(f"   Check credentials/API keys")
            return False
        
        print(f"\n✅ Provider initialized successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing provider: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_detection(image_path=None):
    """Test OCR detection with the provider."""
    if not image_path:
        print("\nℹ️  Skipping detection test (no image provided)")
        return True
    
    if not os.path.exists(image_path):
        print(f"\n⚠️  Image not found: {image_path}")
        return True
    
    try:
        print(f"\n{'='*60}")
        print(f"Testing OCR Detection")
        print(f"{'='*60}\n")
        print(f"Image: {image_path}")
        
        config = VisionConfig()
        detector = VisionPhraseDetector(config)
        
        # Test basic text detection
        print("\nRunning OCR detection...")
        results = detector.detect(
            image_path=image_path,
            search_phrases=["test"],
            threshold=75,
            show_plot=False
        )
        
        if results:
            print(f"\n✅ Detection completed!")
            print(f"  - Total text: {len(results.get('all_text', ''))} characters")
            print(f"  - Matches found: {results.get('total_matches', 0)}")
            
            # Show sample of detected text
            all_text = results.get('all_text', '')
            if all_text:
                preview = all_text[:200].replace('\n', ' ')
                print(f"  - Text preview: {preview}...")
        else:
            print(f"\n⚠️  No results returned")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run provider tests."""
    print("\n" + "="*60)
    print("ThriftAssist OCR Provider Test")
    print("="*60)
    
    # Test initialization
    init_ok = test_provider_initialization()
    
    # Test detection if image provided
    if len(sys.argv) > 2:
        detect_ok = test_provider_detection(sys.argv[2])
    else:
        detect_ok = True
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"Provider initialization: {'✅ PASS' if init_ok else '❌ FAIL'}")
    print(f"OCR detection: {'✅ PASS' if detect_ok else '❌ FAIL'}")
    print()
    
    return init_ok and detect_ok


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("\nUsage: python test_providers.py [provider] [image_path]")
        print("\nProviders:")
        print("  google    - Google Cloud Vision (default)")
        print("  deepseek  - DeepSeek-OCR")
        print("\nExamples:")
        print("  python test_providers.py google")
        print("  python test_providers.py deepseek")
        print("  python test_providers.py google image/test.jpg")
        print("\nEnvironment variables:")
        print("  OCR_PROVIDER              - Override provider selection")
        print("  GOOGLE_APPLICATION_CREDENTIALS - Google Cloud credentials path")
        print("  DEEPSEEK_API_KEY          - DeepSeek API key")
        print()
        sys.exit(0)
    
    success = main()
    sys.exit(0 if success else 1)
