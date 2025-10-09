# ThriftAssist OCR API - Architecture Documentation

## Overview

ThriftAssist is a REST API service for detecting and annotating phrases in images using Google Cloud Vision API. The application follows a clean, layered architecture separating concerns into distinct modules.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend Layer                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ web_app.html (Static HTML/JS/CSS)                    │   │
│  │ - Image upload UI                                     │   │
│  │ - Search phrase input                                 │   │
│  │ - Results display with zoom/pan                       │   │
│  │ - Threshold controls                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                          API Layer                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ main.py (FastAPI Application)                        │   │
│  │                                                       │   │
│  │ Routes:                                              │   │
│  │  ├─ /health         (Health check)                  │   │
│  │  ├─ /ocr/upload     (Upload & detect)               │   │
│  │  ├─ /ocr/detect     (Base64 detect)                 │   │
│  │  ├─ /cache/status   (Cache info)                    │   │
│  │  └─ /cache/clear    (Clear cache)                   │   │
│  │                                                       │   │
│  │ Middleware:                                          │   │
│  │  ├─ CORS (allow origins)                            │   │
│  │  ├─ Static files (/static)                          │   │
│  │  └─ Error handling                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Service Layer                         │
│  ┌─────────────────┬──────────────────┬──────────────────┐  │
│  │  OCR Service    │  Cache Service   │  Image Service   │  │
│  │                 │                  │                  │  │
│  │ • Phrase        │ • LRU cache     │ • Base64 ↔      │  │
│  │   detection     │ • Hash keys     │   Image array    │  │
│  │ • Google Cloud  │ • Expiry        │ • Validation     │  │
│  │   Vision API    │   (1 hour)      │ • Temp files     │  │
│  │ • Result        │ • Max 100       │                  │  │
│  │   formatting    │   entries       │                  │  │
│  └─────────────────┴──────────────────┴──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Core OCR Module                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ thriftassist_googlevision.py                         │   │
│  │                                                       │   │
│  │ • detect_and_annotate_phrases()                      │   │
│  │ • group_text_into_lines()                            │   │
│  │ • find_complete_phrases()                            │   │
│  │ • draw_phrase_annotations()                          │   │
│  │ • normalize_text_for_search()                        │   │
│  │ • try_reverse_text_matching()                        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   External Services                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Google Cloud Vision API                              │   │
│  │ • document_text_detection()                          │   │
│  │ • text_detection()                                   │   │
│  │ • Handles multi-angle text                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
thrift_assist/
├── backend/                          # Backend services (future refactoring)
│   ├── api/                          # API layer
│   │   ├── routes/                   # API endpoints
│   │   │   ├── health.py            # Health check routes
│   │   │   ├── ocr.py               # OCR routes
│   │   │   └── cache.py             # Cache management routes
│   │   ├── models/                   # Pydantic models
│   │   │   ├── requests.py          # Request schemas
│   │   │   └── responses.py         # Response schemas
│   │   └── main.py                   # FastAPI app initialization
│   ├── services/                     # Business logic
│   │   ├── ocr_service.py           # OCR operations
│   │   ├── cache_service.py         # Cache management
│   │   └── image_service.py         # Image processing
│   ├── core/                         # Core configuration
│   │   ├── config.py                # Settings management
│   │   ├── credentials.py           # Google Cloud auth
│   │   └── logging.py               # Logging config
│   └── utils/                        # Utility functions
│       ├── image_utils.py           # Image helpers
│       └── text_utils.py            # Text processing
├── frontend/                         # Frontend application
│   └── public/                       # Static files
│       ├── web_app.html             # Main web interface
│       └── favicon.ico              # Site icon
├── credentials/                      # API credentials (gitignored)
│   └── *.json                       # Google Cloud credentials
├── main.py                          # Current monolithic API (legacy)
├── thriftassist_googlevision.py    # Core OCR module
├── run_api.py                       # Development server launcher
├── requirements.txt                 # Python dependencies
├── ARCHITECTURE.md                  # This file
└── README.md                        # Project documentation
```

## Component Descriptions

### 1. Frontend Layer

**Location:** `frontend/public/web_app.html`

**Responsibilities:**
- User interface for image upload
- Search phrase input and configuration
- Interactive result display with zoom/pan
- Real-time threshold adjustment
- Mobile-responsive design

**Technologies:**
- HTML5, CSS3, JavaScript (Vanilla)
- Canvas API for image display
- Fetch API for REST communication

### 2. API Layer

**Current:** `main.py`  
**Future:** `backend/api/main.py`

**Responsibilities:**
- HTTP request handling
- Request validation
- Response formatting
- CORS management
- Static file serving
- Error handling

**Key Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/ocr/upload` | POST | Upload image & detect phrases |
| `/ocr/detect` | POST | Detect from base64 image |
| `/cache/status` | GET | View cache statistics |
| `/cache/clear` | POST | Clear OCR cache |
| `/` | GET | Serve web application |
| `/docs` | GET | API documentation (Swagger) |

**Technologies:**
- FastAPI framework
- Pydantic for validation
- Uvicorn ASGI server

### 3. Service Layer

**Future Location:** `backend/services/`

**OCR Service** (`ocr_service.py`):
- Wrapper for Google Cloud Vision API
- Phrase detection logic
- Result formatting for API responses

**Cache Service** (`cache_service.py`):
- LRU cache implementation
- Image hash generation
- Cache expiry management (1-hour TTL)
- Maximum 100 cached entries

**Image Service** (`image_service.py`):
- Base64 ↔ NumPy array conversion
- Image validation
- Temporary file management

### 4. Core OCR Module

**Location:** `thriftassist_googlevision.py`

**Key Functions:**

```python
detect_and_annotate_phrases(image_path, search_phrases, threshold, text_scale)
  ├─ Loads image from file
  ├─ Calls Google Cloud Vision API
  ├─ Groups text into lines (multi-angle support)
  ├─ Finds phrase matches (case-insensitive)
  ├─ Draws annotations on image
  └─ Returns results dict

group_text_into_lines(text_annotations)
  ├─ Analyzes text orientation
  ├─ Groups by proximity and angle
  └─ Returns structured text lines

find_complete_phrases(phrase, text_lines, full_text, threshold)
  ├─ Exact substring matching (100% score)
  ├─ Fuzzy matching (RapidFuzz)
  ├─ Multi-line phrase spanning
  └─ Returns match list with scores

draw_phrase_annotations(image, phrase_matches, text_scale)
  ├─ Draws bounding boxes
  ├─ Adds labeled text
  ├─ Smart label placement (avoid overlaps)
  └─ Returns annotated image
```

**Features:**
- Multi-angle text detection (0°, 90°, 180°, 270°, etc.)
- Case-insensitive matching
- Fuzzy matching with configurable threshold (50-100)
- Spanning phrase detection across multiple lines
- Scalable annotation text size

### 5. Configuration & Settings

**Environment Variables:**

```bash
# Google Cloud Authentication
GOOGLE_APPLICATION_CREDENTIALS        # Path to credentials JSON
GOOGLE_APPLICATION_CREDENTIALS_JSON   # Inline credentials JSON
GOOGLE_CREDENTIALS_BASE64             # Base64 encoded credentials

# Server Configuration
PORT=8000                             # Server port
HOST=0.0.0.0                         # Server host

# Cache Settings
MAX_CACHE_SIZE=100                    # Maximum cached entries
CACHE_EXPIRY_SECONDS=3600            # 1 hour TTL
```

**Settings Management** (Future: `backend/core/config.py`):
```python
class Settings(BaseSettings):
    API_TITLE: str
    API_VERSION: str
    CORS_ORIGINS: list
    MAX_CACHE_SIZE: int
    CACHE_EXPIRY_SECONDS: int
    DEFAULT_THRESHOLD: int = 75
    DEFAULT_TEXT_SCALE: int = 100
```

## Data Flow

### 1. Image Upload & Detection

```
User Upload → Frontend
              │
              ▼
         FormData (file, phrases, threshold, text_scale)
              │
              ▼ POST /ocr/upload
         API Layer (main.py)
              │
              ├─ Validate file type
              ├─ Calculate image hash
              └─ Check cache
                   │
                   ├─ Cache Hit  → Use cached image
                   │                    │
                   │                    ▼
                   └─ Cache Miss → Process image
                                        │
                                        ├─ Decode image
                                        ├─ Save temp file
                                        └─ Cache result
                                             │
                                             ▼
                                   OCR Service
                                        │
                                        ├─ Call Google Cloud Vision
                                        ├─ Group text into lines
                                        ├─ Find phrase matches
                                        └─ Draw annotations
                                             │
                                             ▼
                                   Format Response
                                        │
                                        ├─ Convert to JSON
                                        ├─ Base64 encode image
                                        └─ Add metadata
                                             │
                                             ▼
                                   Return to Frontend
                                        │
                                        ▼
                                   Display Results
```

### 2. Threshold Update (Using Cache)

```
Threshold Change → Frontend
                        │
                        ▼ POST /ocr/upload (same image hash)
                   Check Cache
                        │
                        ├─ Found → Retrieve cached image path
                        │          │
                        │          ▼
                        │     Re-run detection with new threshold
                        │          │
                        │          └─ Fast response (~500ms)
                        │
                        └─ Not Found → Full OCR process (~2-5s)
```

## Caching Strategy

### Cache Key Generation

```python
image_hash = MD5(image_bytes + text_scale)
```

**Why include text_scale?**
- Different text scales produce different annotated images
- Ensures cache invalidation when annotation size changes

### Cache Storage

```python
ocr_cache = OrderedDict()  # LRU cache

cache_entry = {
    'timestamp': time.time(),
    'ocr_data': {
        'image_path': '/tmp/xyz.jpg',
        'image_dimensions': [1920, 1080],
        'all_text': 'detected text...',
        'filename': 'upload.jpg',
        'text_scale': 100
    }
}
```

### Cache Eviction

- **LRU (Least Recently Used):** Oldest entries removed when cache is full
- **TTL (Time To Live):** Entries expire after 1 hour
- **Size Limit:** Maximum 100 entries

### Cache Benefits

1. **Performance:** Threshold updates are 5-10x faster
2. **Cost Savings:** Reduces Google Cloud Vision API calls
3. **User Experience:** Near-instant threshold adjustments
4. **Resource Efficiency:** Reuses OCR results

## API Request/Response Schemas

### Upload & Detect Request

```json
{
  "file": "<multipart/form-data>",
  "search_phrases": "[\"Billy Joel\", \"U2\", \"Jewel\"]",
  "threshold": 75,
  "text_scale": 100
}
```

### Detection Response

```json
{
  "success": true,
  "total_matches": 3,
  "matches": {
    "Billy Joel": [
      {
        "text": "BILLY JOEL",
        "score": 100.0,
        "match_type": "complete_phrase",
        "angle": 0,
        "is_spanning": false
      }
    ]
  },
  "processing_time_ms": 1250.5,
  "image_dimensions": [1920, 1080],
  "annotated_image_base64": "data:image/jpeg;base64,...",
  "all_detected_text": "full OCR text...",
  "filename": "image.jpg",
  "cached": false
}
```

## Error Handling

### API Layer Errors

```python
try:
    # Process request
    return success_response
except HTTPException:
    # Re-raise HTTP exceptions
    raise
except Exception as e:
    # Log error and return formatted response
    return {
        "success": false,
        "error_message": str(e),
        "processing_time_ms": elapsed_time
    }
```

### OCR Module Errors

```python
try:
    # Call Google Cloud Vision API
    results = detect_and_annotate_phrases(...)
except DefaultCredentialsError:
    # Authentication error
    return None
except Exception as e:
    # General error
    logger.error(f"OCR failed: {e}")
    return None
```

## Performance Characteristics

| Operation | Cold Start | Cached | Notes |
|-----------|-----------|--------|-------|
| OCR Detection | 2-5s | N/A | Google API call |
| Threshold Update | 2-5s | 300-800ms | With cache |
| Image Upload | 100-300ms | N/A | File I/O |
| Base64 Conversion | 50-150ms | N/A | CPU bound |

## Security Considerations

### Authentication
- Google Cloud credentials via environment variables
- Multiple credential loading methods (JSON, base64, file path)
- Credentials not exposed in API responses

### Input Validation
- File type checking (images only)
- File size limits (10MB default)
- JSON schema validation (Pydantic)
- Image format validation

### CORS Policy
```python
CORSMiddleware(
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

## Deployment Architecture

### Development
```bash
# Local development server
python run_api.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production (Render.com)
```yaml
services:
  - type: web
    name: thriftassist-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GOOGLE_CREDENTIALS_BASE64
        sync: false
```

## Future Improvements

### Planned Refactoring

1. **Complete Service Layer Migration**
   - Move from monolithic `main.py` to modular `backend/` structure
   - Separate concerns into distinct service modules

2. **Enhanced Caching**
   - Redis integration for distributed caching
   - Cache warming strategies
   - Configurable TTL per use case

3. **Authentication & Authorization**
   - API key authentication
   - Rate limiting per user/key
   - Usage tracking and quotas

4. **Monitoring & Observability**
   - Structured logging (JSON format)
   - Metrics collection (Prometheus)
   - Distributed tracing (OpenTelemetry)
   - Error tracking (Sentry)

5. **Testing**
   - Unit tests for services
   - Integration tests for API
   - E2E tests for workflows
   - Performance benchmarks

6. **Database Integration**
   - Store OCR results persistently
   - User preferences and history
   - Analytics and usage patterns

## Technology Stack

### Backend
- **Framework:** FastAPI 0.100+
- **Server:** Uvicorn (ASGI)
- **Validation:** Pydantic
- **OCR:** Google Cloud Vision API
- **Fuzzy Matching:** RapidFuzz
- **Image Processing:** OpenCV (cv2), NumPy

### Frontend
- **Core:** HTML5, CSS3, ES6 JavaScript
- **No Framework:** Vanilla JS for simplicity
- **Image Display:** Canvas API
- **HTTP Client:** Fetch API

### Infrastructure
- **Deployment:** Render.com
- **Storage:** Temporary file system (ephemeral)
- **Cache:** In-memory OrderedDict (LRU)

## Configuration Files

### requirements.txt
```txt
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.0.6
google-cloud-vision>=3.4.0
opencv-python>=4.8.0
numpy>=1.24.0
rapidfuzz>=3.0.0
pydantic>=2.0.0
```

### .env (Example)
```bash
GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account.json
PORT=8000
HOST=0.0.0.0
MAX_CACHE_SIZE=100
CACHE_EXPIRY_SECONDS=3600
```

## Development Workflow

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up credentials
export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Run development server
python run_api.py
```

### 2. Testing
```bash
# Access web UI
open http://localhost:8000

# Access API docs
open http://localhost:8000/docs

# Health check
curl http://localhost:8000/health
```

### 3. Deployment
```bash
# Commit changes
git add .
git commit -m "feature: description"
git push origin main

# Render.com auto-deploys from main branch
```

## Contact & Support

For questions about this architecture:
- Review this documentation
- Check API docs at `/docs`
- Examine code comments in source files

---

**Last Updated:** 2024-01-15  
**Version:** 1.0.0  
**Maintainer:** ThriftAssist Development Team
