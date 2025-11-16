# OCR Provider Architecture

ThriftAssist supports multiple OCR providers through a plugin architecture. This document explains how to configure and use different providers.

## Supported Providers

### 1. Google Cloud Vision (default)
- **Provider ID**: `google`
- **Cost**: Pay-per-use (free tier available)
- **Strengths**: Very accurate, handles complex layouts well
- **Requirements**: Google Cloud account and service account credentials

### 2. DeepSeek-OCR
- **Provider ID**: `deepseek`
- **Cost**: Pay-per-use based on API tokens
- **Strengths**: AI-powered, good at understanding context
- **Requirements**: DeepSeek API key

## Configuration

### Environment Variables

- `OCR_PROVIDER`: Set to `google` or `deepseek` (default: `google`)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud credentials JSON file
- `DEEPSEEK_API_KEY`: DeepSeek API key

### Example Configurations

**Google Cloud Vision:**
```bash
export OCR_PROVIDER=google
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

**DeepSeek-OCR:**
```bash
export OCR_PROVIDER=deepseek
export DEEPSEEK_API_KEY="sk-..."
```

## Provider Selection Logic

1. The system checks the `OCR_PROVIDER` environment variable
2. If not set, it uses the value from `VisionConfig.ocr_provider` (default: `google`)
3. If the selected provider is unavailable or misconfigured, it falls back to Google Cloud Vision

## Adding New Providers

To add a new OCR provider:

1. Create a new file in `vision/providers/` (e.g., `my_provider.py`)
2. Implement the `OCRProvider` interface from `vision/providers/base_provider.py`
3. Add the provider to `vision/providers/__init__.py`
4. Update `VisionPhraseDetector._setup_provider()` to handle the new provider
5. Update `VisionConfig` with any provider-specific configuration

### Example Provider Implementation

```python
from .base_provider import OCRProvider, TextAnnotation
from typing import List, Tuple

class MyProvider(OCRProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    @property
    def name(self) -> str:
        return "My Provider"
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def detect_text(self, image_path: str) -> Tuple[str, List[TextAnnotation]]:
        # Implement OCR logic here
        # Return (full_text, list of TextAnnotation objects)
        pass
```

## Architecture Overview

```
VisionPhraseDetector
    ├── OCRProvider (interface)
    │   ├── GoogleVisionProvider
    │   ├── DeepSeekProvider
    │   └── [Your Provider]
    ├── TextLineGrouper
    ├── PhraseMatcher
    └── ImageAnnotator
```

The `VisionPhraseDetector` orchestrates the OCR workflow:
1. Provider detects text → returns standardized `TextAnnotation` objects
2. `TextLineGrouper` groups annotations into logical text lines
3. `PhraseMatcher` finds phrases using fuzzy matching
4. `ImageAnnotator` draws bounding boxes on the image

## Testing Providers

To test a specific provider:

```bash
# Test with Google Cloud Vision
OCR_PROVIDER=google python main.py --image test.jpg --phrases "hello world"

# Test with DeepSeek
OCR_PROVIDER=deepseek python main.py --image test.jpg --phrases "hello world"
```

## Troubleshooting

### Provider Not Available

If you see "Provider not available" errors:

1. **Google Vision**: 
   - Check `GOOGLE_APPLICATION_CREDENTIALS` points to valid JSON file
   - Verify credentials file has Vision API enabled
   - Ensure billing is enabled on your Google Cloud project

2. **DeepSeek**:
   - Verify `DEEPSEEK_API_KEY` is set correctly
   - Check API key is valid and has available credits
   - Ensure `requests` library is installed: `pip install requests`

### Fallback Behavior

The system will automatically fall back to Google Cloud Vision if:
- Selected provider is not available
- Provider initialization fails
- Required dependencies are missing

You'll see a warning message indicating the fallback occurred.
