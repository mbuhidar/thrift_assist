# ThriftAssist OCR - Architecture Documentation

## Overview

ThriftAssist is a modular OCR system for detecting and annotating phrases in images using Google Cloud Vision API. The application has been fully refactored from a monolithic script into a clean, layered architecture with FastAPI backend integration.

## Architecture Status

### ✅ Migration Complete: Modular Architecture with FastAPI Backend

**Current (Production):**
- `backend/` - FastAPI web application with service layer architecture
- `vision/` - Modular OCR package (core implementation)
- `utils/` - Shared utilities
- `config/` - Configuration management

**Legacy (Compatibility Layer):**
- `thriftassist_googlevision.py` - Original monolithic implementation
  - **Status**: Legacy reference implementation
  - **Usage**: Backend services now use modular `VisionPhraseDetector`
  - **Deprecation**: Maintained for reference, backend uses vision package

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Web Frontend (public/web_app.html)                 │   │
│  │ • REST API Clients                                   │   │
│  │ • Command Line Interface (vision_demo.py)            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ API Layer (backend/api/)                             │   │
│  │  ├─ main.py - FastAPI app configuration              │   │
│  │  ├─ routes/ocr.py - OCR endpoints                    │   │
│  │  ├─ routes/health.py - Health check                  │   │
│  │  └─ routes/cache.py - Cache management               │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Service Layer (backend/services/)                    │   │
│  │  ├─ ocr_service.py - OCR operations                  │   │
│  │  ├─ cache_service.py - Result caching                │   │
│  │  └─ image_service.py - Image processing              │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Core Layer (backend/core/)                           │   │
│  │  ├─ config.py - Application settings                 │   │
│  │  └─ credentials.py - Authentication                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Vision Package                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ VisionPhraseDetector (vision/detector.py)            │   │
│  │                                                      │   │
│  │ Public Interface:                                    │   │
│  │  ├─ detect(image_path, phrases, threshold)           │   │
│  │  ├─ _detect_text() - Google Cloud Vision API        │   │
│  │  ├─ _print_orientation_info()                        │   │
│  │  └─ _show_results() - Matplotlib display            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        ┌───────────────────┐   ┌───────────────────┐
        │  TextLineGrouper  │   │  PhraseMatcher    │
        │  (grouper.py)     │   │  (matcher.py)     │
        │                   │   │                   │
        │ • group()         │   │ • find_matches()  │
        │ • _group_by_angle │   │ • _search_in_     │
        │ • _group_by_      │   │   lines()         │
        │   position        │   │ • _search_        │
        │ • _convert_to_    │   │   spanning()      │
        │   text_lines      │   │ • _deduplicate_   │
        └───────────────────┘   │   matches()       │
                                └───────────────────┘
                                          │
                                          ▼
                                ┌───────────────────┐
                                │  ImageAnnotator   │
                                │  (annotator.py)   │
                                │                   │
                                │ • draw_           │
                                │   annotations()   │
                                │ • _extract_bbox() │
                                │ • _draw_label()   │
                                │ • _find_label_    │
                                │   position()      │
                                └───────────────────┘
```

## Directory Structure

```
thrift_assist/
├── main.py                           # Render.com deployment entry point
├── run_api.py                        # Development server runner
│
├── backend/                          # FastAPI Application
│   ├── __init__.py                   # Backend package root
│   ├── api/                          # API layer
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app configuration
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py             # Health check endpoints
│   │   │   ├── ocr.py                # OCR processing endpoints
│   │   │   └── cache.py              # Cache management endpoints
│   │   └── models/                   # Request/response models
│   │       ├── __init__.py
│   │       ├── requests.py           # Pydantic request models
│   │       └── responses.py          # Pydantic response models
│   ├── services/                     # Business logic layer
│   │   ├── __init__.py
│   │   ├── ocr_service.py            # OCR operations using VisionPhraseDetector
│   │   ├── cache_service.py          # Result caching
│   │   └── image_service.py          # Image processing utilities
│   └── core/                         # Core functionality
│       ├── __init__.py
│       ├── config.py                 # Application settings
│       └── credentials.py            # Authentication setup
│
├── vision/                           # Vision OCR package
│   ├── __init__.py                   # Public API exports
│   ├── detector.py                   # Main detector class
│   ├── grouper.py                    # Text line grouping
│   ├── matcher.py                    # Phrase matching logic
│   └── annotator.py                  # Image annotation
│
├── utils/                            # Utility functions
│   ├── __init__.py                   # Utility exports
│   ├── text_utils.py                 # Text processing
│   └── geometry_utils.py             # Bounding box operations
│
├── config/                           # Configuration
│   ├── __init__.py                   # Config exports
│   └── vision_config.py              # Settings dataclass
│
├── public/                           # Web frontend assets
│   └── web_app.html                  # Frontend interface
│
├── credentials/                      # API credentials (gitignored)
│   └── *.json                        # Google Cloud credentials
│
├── thriftassist_googlevision.py      # Legacy monolithic implementation
├── vision_demo.py                    # Standalone demo application
├── requirements.txt                  # Python dependencies
├── ARCHITECTURE.md                   # This file
└── README.md                         # User documentation
```

## Backend Service Integration

### OCR Service Layer (`backend/services/ocr_service.py`)

**Current Implementation:**
```python
from vision.detector import VisionPhraseDetector
from config.vision_config import VisionConfig

class OCRService:
    def __init__(self):
        if self.ocr_available:
            config = VisionConfig()
            config.fuzz_threshold = settings.DEFAULT_THRESHOLD
            config.default_text_scale = settings.DEFAULT_TEXT_SCALE
            self.detector = VisionPhraseDetector(config)
    
    def detect_phrases(self, image_path, search_phrases, threshold=None, 
                      text_scale=None, show_plot=False):
        return self.detector.detect(
            image_path=image_path,
            search_phrases=search_phrases,
            threshold=threshold,
            text_scale=text_scale,
            show_plot=show_plot
        )
```

**Previous Implementation (Deprecated):**
```python
# Old approach - no longer used
from thriftassist_googlevision import detect_and_annotate_phrases

results = detect_and_annotate_phrases(
    image_path=image_path,
    search_phrases=search_phrases,
    threshold=threshold,
    text_scale=text_scale,
    show_plot=show_plot
)
```

### API Endpoints

**OCR Processing:**
- `POST /ocr/upload` - Upload image and detect phrases
- `POST /ocr/detect` - Process with search parameters

**System:**
- `GET /health` - Service health check
- `GET /cache/status` - Cache statistics

**Legacy Support:**
- `POST /upload-and-detect` - Backward compatibility endpoint

## Migration Benefits

### From Monolithic to Modular Backend

**Before (thriftassist_googlevision.py):**
- Single 2000+ line file
- Direct function calls
- Hardcoded configuration
- No API interface
- Limited reusability

**After (Vision Package + FastAPI Backend):**
- Separation of concerns
- Dependency injection
- Configurable via API
- RESTful interface
- Scalable architecture
- Proper error handling
- Response caching

### Service Layer Advantages

1. **Abstraction**: API routes don't directly interact with Google Cloud Vision
2. **Configuration**: Backend settings integrate with VisionConfig
3. **Error Handling**: Graceful failures with proper HTTP responses
4. **Caching**: Service layer can cache results for repeated requests
5. **Testing**: Service methods can be unit tested independently
6. **Scaling**: Service instances can be distributed across workers

## Data Flow Updates

### FastAPI Request Flow

```
1. Client Request (POST /ocr/upload)
   ├─ File upload via multipart/form-data
   ├─ Search phrases in form fields
   └─ Optional threshold/text_scale parameters
        │
        ▼
2. API Route Handler (backend/api/routes/ocr.py)
   ├─ Request validation (Pydantic models)
   ├─ File processing (save to temp location)
   └─ Call OCR Service
        │
        ▼
3. OCR Service (backend/services/ocr_service.py)
   ├─ Configure VisionPhraseDetector with settings
   ├─ Call detector.detect() method
   └─ Format results for API response
        │
        ▼
4. VisionPhraseDetector (vision/detector.py)
   ├─ Google Cloud Vision API integration
   ├─ Text grouping and phrase matching
   └─ Image annotation generation
        │
        ▼
5. API Response
   ├─ JSON response with match data
   ├─ Base64 encoded annotated image
   └─ HTTP status codes and error handling
```

### Configuration Integration

**Backend Settings → Vision Config:**
```python
# backend/core/config.py
class Settings(BaseSettings):
    DEFAULT_THRESHOLD: int = 75
    DEFAULT_TEXT_SCALE: int = 100
    
# backend/services/ocr_service.py  
config = VisionConfig()
config.fuzz_threshold = settings.DEFAULT_THRESHOLD
config.default_text_scale = settings.DEFAULT_TEXT_SCALE
```

## Deployment Architecture

### Render.com Production Setup

**Entry Point:**
```python
# main.py - Render.com entry point
from backend.api.main import app

# Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Environment Variables:**
```bash
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/key.json
API_TITLE="ThriftAssist OCR API"
DEFAULT_THRESHOLD=75
DEFAULT_TEXT_SCALE=100
```

### Development Setup

**Local Development:**
```bash
# Use enhanced development runner
python run_api.py

# Direct uvicorn (alternative)
uvicorn backend.api.main:app --reload --port 8000
```

## API Response Format Updates

### Enhanced Response Structure

```python
{
    "success": true,
    "message": "OCR processing completed",
    "data": {
        "total_matches": 2,
        "processing_time_ms": 1547,
        "matches": {
            "Billy Joel": [
                {
                    "text": "BILLY JOEL", 
                    "score": 100.0,
                    "match_type": "complete_phrase",
                    "angle": 0.0,
                    "is_spanning": false,
                    "explanation": {
                        "phrase_searched": "Billy Joel",
                        "text_found": "BILLY JOEL",
                        "overall_score": 100.0,
                        "confidence_level": "Very High",
                        "reasoning": ["Exact substring match found (case-insensitive)"]
                    }
                }
            ]
        },
        "annotated_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        "image_info": {
            "width": 1024,
            "height": 768,
            "orientations_detected": ["horizontal", "upside-down"]
        }
    }
}
```

## Performance Optimizations

### Service Layer Improvements

1. **Connection Pooling**: Reuse Google Cloud Vision client instances
2. **Result Caching**: Cache results by image hash and search parameters  
3. **Async Processing**: FastAPI async endpoints for better concurrency
4. **Image Optimization**: Resize large images before processing
5. **Background Tasks**: Process annotations asynchronously

### Scaling Considerations

- **Horizontal Scaling**: Multiple FastAPI workers with shared cache
- **Load Balancing**: Distribute requests across service instances
- **Database Integration**: Store results in persistent storage
- **CDN Integration**: Serve annotated images from CDN

## Testing Strategy Updates

### Backend Testing Structure

```python
# tests/unit/services/test_ocr_service.py
def test_ocr_service_initialization()
def test_detect_phrases_success()
def test_detect_phrases_with_invalid_image()
def test_format_matches_for_api()

# tests/unit/api/test_ocr_routes.py  
def test_upload_endpoint_success()
def test_upload_endpoint_validation_errors()
def test_detect_endpoint_with_parameters()

# tests/integration/test_end_to_end.py
def test_full_api_workflow()
def test_vision_detector_integration()
```

## Migration Guide

### For Existing Code Using Legacy Implementation

**Old Pattern:**
```python
from thriftassist_googlevision import detect_and_annotate_phrases

results = detect_and_annotate_phrases(
    image_path="image.jpg",
    search_phrases=["Billy Joel", "U2"], 
    threshold=75,
    show_plot=False
)
```

**New Pattern (Direct Vision Package):**
```python
from vision import VisionPhraseDetector
from config import VisionConfig

config = VisionConfig(fuzz_threshold=75)
detector = VisionPhraseDetector(config)

results = detector.detect(
    image_path="image.jpg",
    search_phrases=["Billy Joel", "U2"],
    show_plot=False
)
```

**New Pattern (Via FastAPI Backend):**
```python
import requests

files = {"file": open("image.jpg", "rb")}
data = {
    "search_phrases": "Billy Joel,U2",
    "threshold": 75
}

response = requests.post("http://localhost:8000/ocr/upload", 
                        files=files, data=data)
results = response.json()
```

## Changelog

### v2.0.0 (Current) - FastAPI Backend Integration

**Backend Changes:**
- ✅ FastAPI application with modular architecture
- ✅ Service layer using VisionPhraseDetector instead of legacy functions  
- ✅ RESTful API endpoints for OCR processing
- ✅ Pydantic models for request/response validation
- ✅ Async processing support
- ✅ Health check and cache management endpoints
- ✅ Render.com deployment configuration

**OCR Service Migration:**
- ✅ Replaced direct `thriftassist_googlevision` imports
- ✅ Integrated `VisionPhraseDetector` with backend configuration
- ✅ Enhanced error handling and response formatting
- ✅ Added explainability data to API responses

**Deployment Updates:**
- ✅ Production-ready main.py for Render.com
- ✅ Development server with enhanced logging
- ✅ Environment variable configuration
- ✅ Static file serving for web frontend

### Migration Benefits Summary

| Aspect | Before | After |
|--------|---------|-------|
| **Architecture** | Monolithic script | Modular FastAPI backend |
| **OCR Integration** | Direct function calls | Service layer with VisionPhraseDetector |
| **Configuration** | Hardcoded values | Environment variables + VisionConfig |
| **Interface** | Command line only | REST API + CLI |
| **Error Handling** | Print statements | HTTP status codes + structured responses |
| **Deployment** | Manual execution | Cloud platform ready |
| **Scaling** | Single process | Multi-worker support |
| **Testing** | Manual verification | Unit/integration test framework |

---

## License

MIT License - See LICENSE file for details
### Text Grouping Flow

```
TextAnnotations from Google API
        │
        ▼
TextLineGrouper.group()
        │
        ├─ 1. Calculate angles
        │   └─ atan2(p2.y-p1.y, p2.x-p1.x)
        │
        ├─ 2. Group by angle (±15° tolerance)
        │   ├─ 0° → horizontal
        │   ├─ 90° → vertical up
        │   ├─ -90° → vertical down
        │   └─ 180° → upside-down
        │
        ├─ 3. For each angle group:
        │   ├─ Calculate position key
        │   │   ├─ Horizontal: use y_position
        │   │   ├─ Vertical: use x_position
        │   │   └─ Diagonal: use center point
        │   │
        │   └─ Group by position (±20px tolerance)
        │
        └─ 4. Convert to text lines
            ├─ Sort words by position
            ├─ Join into line text
            └─ Attach metadata (angle, position, annotations)
```

---

### Phrase Matching Flow

```
PhraseMatcher.find_matches()
        │
        ├─ 1. Check if meaningful
        │   └─ Skip if only common words
        │
        ├─ 2. Normalize phrase
        │   ├─ Lowercase
        │   └─ Fix OCR artifacts
        │
        ├─ 3. Search in single lines
        │   ├─ Exact match → 100% score
        │   ├─ Fuzzy match → RapidFuzz score
        │   └─ Reverse match → upside-down text
        │
        ├─ 4. Search spanning lines (if multi-word)
        │   ├─ Find words in current line
        │   ├─ Look in adjacent lines (±3 lines)
        │   ├─ Check compatibility:
        │   │   ├─ Y-distance ≤ 100px
        │   │   └─ Angle diff ≤ 30°
        │   ├─ Calculate match percentage
        │   └─ Create spanning match if ≥70%
        │
        └─ 5. Deduplicate
            ├─ Generate unique keys
            ├─ Keep highest scores
            └─ Return sorted matches
```

---

## API Response Format

### Detection Results

```python
{
    'image': np.ndarray,              # Original image
    'annotated_image': np.ndarray,    # Annotated image
    'matches': {
        'Billy Joel': [
            (
                {
                    'text': 'BILLY JOEL',
                    'annotations': [<Annotation>, ...],
                    'y_position': 245.0,
                    'angle': 0.0
                },
                100.0,                # score
                'complete_phrase'     # match_type
            )
        ],
        'U2': [...]
    },
    'total_matches': 2,
    'all_text': 'BILLY JOEL\nGreatest Hits\nU2\nThe Joshua Tree...'
}
```

---

## Configuration Options

### VisionConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `credentials_path` | str | `"credentials/..."` | Google Cloud credentials file |
| `fuzz_threshold` | int | 75 | Fuzzy match threshold (0-100) |
| `angle_tolerance` | int | 15 | Text angle grouping tolerance (degrees) |
| `line_proximity_tolerance` | int | 20 | Line grouping distance (pixels) |
| `default_text_scale` | int | 100 | Annotation text size (50-200%) |
| `common_words` | Set[str] | {...} | Words to filter when alone |

### Threshold Guide

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 90-100 | Very strict, near-exact matches | High precision needed |
| 75-89 | Balanced, allows minor OCR errors | General use (recommended) |
| 50-74 | Lenient, more false positives | Poor image quality |

---

## Performance Characteristics

### Detection Speed

| Component | Typical Time | Notes |
|-----------|--------------|-------|
| Google Vision API | 1000-2500ms | Network dependent |
| Text grouping | 10-50ms | Local processing |
| Phrase matching | 50-200ms | Depends on phrase count |
| Annotation drawing | 100-300ms | Depends on match count |
| **Total** | **1200-3000ms** | Per image |

### Memory Usage

- Text annotations: ~1-5 MB per image
- Image arrays: Width × Height × 3 bytes
- Cache overhead: ~10 KB per entry

---

## Error Handling

### Common Errors and Responses

**Authentication Failure:**
```python
DefaultCredentialsError: Could not automatically determine credentials
```
→ Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

**Invalid Image:**
```python
cv2.error: Image is empty or corrupt
```
→ Validate image file before processing

**No Text Detected:**
```python
Returns: None (graceful failure)
```
→ Check image quality and text visibility

**Import Errors:**
```python
ImportError: attempted relative import beyond top-level package
```
→ Run from correct directory or install package

---

## Extension Points

### Future Enhancements

1. **REST API Layer**
   - FastAPI endpoints
   - Image upload/download
   - Session management
   - WebSocket for real-time updates

2. **Advanced Caching**
   - Redis/Memcached integration
   - Distributed cache
   - Pre-computed results database

3. **Batch Processing**
   - Multiple image processing
   - Parallel execution
   - Progress tracking

4. **Additional OCR Engines**
   - Tesseract fallback
   - AWS Textract integration
   - Azure Computer Vision

5. **Enhanced Matching**
   - Regex pattern support
   - Soundex phonetic matching
   - Multi-language support

---

## Testing Strategy

### Unit Tests (Future)

```python
# tests/test_grouper.py
def test_horizontal_text_grouping()
def test_vertical_text_grouping()
def test_upside_down_text_grouping()

# tests/test_matcher.py
def test_exact_phrase_match()
def test_fuzzy_phrase_match()
def test_spanning_phrase_match()
def test_common_word_filtering()

# tests/test_annotator.py
def test_bbox_extraction()
def test_label_positioning()
def test_overlap_avoidance()

# tests/test_text_utils.py
def test_text_normalization()
def test_ocr_artifact_correction()

# tests/test_geometry_utils.py
def test_angle_calculation()
def test_rectangle_overlap()
```

### Integration Tests

```python
# tests/integration/test_end_to_end.py
def test_full_detection_pipeline()
def test_multi_angle_detection()
def test_visualization_output()
```

---

## Deployment

### Package Installation

```bash
# Development install
cd /home/mbuhidar/Code/mbuhidar/thrift_assist
pip install -e .

# Production install
pip install git+https://github.com/yourusername/thrift_assist.git
```

### Environment Setup

```bash
# Required
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# Optional
export VISION_THRESHOLD=75
export VISION_TEXT_SCALE=100
```

### Docker Deployment (Future)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY thrift_assist/ ./thrift_assist/
COPY credentials/ ./credentials/

ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/key.json

CMD ["python", "-m", "thrift_assist.vision_demo"]
```

---

## Changelog

### v1.0.0 (Current) - Modular Refactoring

**Changes:**
- ✅ Refactored monolithic script into modular packages
- ✅ Separated concerns: vision, utils, config
- ✅ Added proper package structure with `__init__.py`
- ✅ Implemented dataclass configuration
- ✅ Absolute imports for better compatibility
- ✅ Created demo application
- ✅ Updated documentation

**Migration from v0.x:**
```python
# Old (v0.x)
from thriftassist_googlevision import detect_and_annotate_phrases

# New (v1.0)
from vision import VisionPhraseDetector
from config import VisionConfig

detector = VisionPhraseDetector(VisionConfig())
results = detector.detect(image_path, phrases)
```

---

## License

MIT License - See LICENSE file for details
