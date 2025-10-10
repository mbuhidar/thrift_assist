# ThriftAssist OCR API - Architecture Documentation

## Overview

ThriftAssist is a REST API service for detecting and annotating phrases in images using Google Cloud Vision API. The application follows a clean, layered architecture separating concerns into distinct modules.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend Layer                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ web_app.html (Static HTML/JS/CSS)                    │   │
│  │ - Image upload UI                                    │   │
│  │ - Search phrase input                                │   │
│  │ - Results display with zoom/pan                      │   │
│  │ - Threshold controls                                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                          API Layer                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ main.py (FastAPI Application)                        │   │
│  │                                                      │   │
│  │ Routes:                                              │   │
│  │  ├─ /health         (Health check)                   │   │
│  │  ├─ /ocr/upload     (Upload & detect)                │   │
│  │  ├─ /ocr/detect     (Base64 detect)                  │   │
│  │  ├─ /cache/status   (Cache info)                     │   │
│  │  └─ /cache/clear    (Clear cache)                    │   │
│  │                                                      │   │
│  │ Middleware:                                          │   │
│  │  ├─ CORS (allow origins)                             │   │
│  │  ├─ Static files (/static)                           │   │
│  │  └─ Error handling                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Service Layer                        │
│  ┌─────────────────┬──────────────────┬──────────────────┐  │
│  │  OCR Service    │  Cache Service   │  Image Service   │  │
│  │                 │                  │                  │  │
│  │ • Phrase        │ • LRU cache     │ • Base64 ↔        │  │
│  │   detection     │ • Hash keys     │   Image array     │  │
│  │ • Google Cloud  │ • Expiry        │ • Validation      │  │
│  │   Vision API    │   (1 hour)      │ • Temp files      │  │
│  │ • Result        │ • Max 100       │                   │  │
│  │   formatting    │   entries       │                   │  │
│  └─────────────────┴──────────────────┴──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Core OCR Module                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ thriftassist_googlevision.py                         │   │
│  │                                                      │   │
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
│                   External Services                         │
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

## Inter-Service Communication Examples

### Example 1: Complete Image Upload Flow

#### 1. Frontend → API Route (`/ocr/upload`)

**HTTP Request:**
```http
POST /ocr/upload HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="file"; filename="cd_collection.jpg"
Content-Type: image/jpeg

[binary image data]
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="search_phrases"

["Billy Joel", "U2", "Jewel"]
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="threshold"

75
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="text_scale"

100
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

#### 2. API Route → Image Service

**Function Call:**
```python
# In backend/api/routes/ocr.py

image_data = await file.read()  # bytes

# Validate image
is_valid = image_service.validate_image_data(
    image_data=image_data,
    max_size_mb=10
)
# Returns: True

# Convert to array
image_array = image_service.base64_to_array(base64_string)
# Returns: numpy.ndarray shape (1080, 1920, 3)

# Save temp file
temp_path = image_service.save_temp_image(image_array)
# Returns: "/tmp/tmpxyz123.jpg"
```

#### 3. API Route → Cache Service

**Function Call:**
```python
# Generate cache key
image_hash = cache_service.get_image_hash(
    image_data=image_data,
    text_scale=100
)
# Returns: "a1b2c3d4e5f6g7h8i9j0"

# Check cache
cached_result = cache_service.get_cached_result(image_hash)
# Returns: None (cache miss) or cached data dict
```

**Cache Miss Response:**
```python
None
```

**Cache Hit Response:**
```python
{
    'image_path': '/tmp/tmpxyz123.jpg',
    'image_dimensions': [1920, 1080],
    'all_text': 'BILLY JOEL Greatest Hits U2 The Joshua Tree...',
    'filename': 'cd_collection.jpg',
    'text_scale': 100
}
```

#### 4. Cache Service → OCR Service (Cache Miss)

**Function Call:**
```python
# In backend/api/routes/ocr.py (after cache miss)

results = ocr_service.detect_phrases(
    image_path='/tmp/tmpxyz123.jpg',
    search_phrases=['Billy Joel', 'U2', 'Jewel'],
    threshold=75,
    text_scale=100,
    show_plot=False
)
```

#### 5. OCR Service → Core OCR Module

**Function Call:**
```python
# In backend/services/ocr_service.py

from thriftassist_googlevision import detect_and_annotate_phrases

results = detect_and_annotate_phrases(
    image_path='/tmp/tmpxyz123.jpg',
    search_phrases=['Billy Joel', 'U2', 'Jewel'],
    threshold=75,
    text_scale=100,
    show_plot=False
)
```

**Core OCR Module Response:**
```python
{
    'image': numpy.ndarray,  # Original image
    'annotated_image': numpy.ndarray,  # Image with bounding boxes
    'matches': {
        'Billy Joel': [
            (
                {
                    'text': 'BILLY JOEL',
                    'annotations': [<Annotation objects>],
                    'y_position': 245,
                    'angle': 0
                },
                100.0,  # score
                'complete_phrase'  # match_type
            )
        ],
        'U2': [
            (
                {
                    'text': 'U2',
                    'annotations': [<Annotation object>],
                    'y_position': 380,
                    'angle': 0
                },
                100.0,
                'complete_phrase'
            )
        ]
    },
    'total_matches': 2,
    'all_text': 'BILLY JOEL Greatest Hits Volume I & II\nU2 The Joshua Tree\nJEWEL Pieces of You...'
}
```

#### 6. OCR Service → API Route (Formatted Results)

**Function Call:**
```python
# In backend/services/ocr_service.py

serializable_matches = ocr_service.format_matches_for_api(results)
```

**Formatted Response:**
```python
{
    'Billy Joel': [
        {
            'text': 'BILLY JOEL',
            'score': 100.0,
            'match_type': 'complete_phrase',
            'angle': 0,
            'is_spanning': False
        }
    ],
    'U2': [
        {
            'text': 'U2',
            'score': 100.0,
            'match_type': 'complete_phrase',
            'angle': 0,
            'is_spanning': False
        }
    ]
}
```

#### 7. API Route → Image Service (Convert to Base64)

**Function Call:**
```python
annotated_base64 = image_service.array_to_base64(
    results['annotated_image']
)
```

**Response:**
```python
'/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a...'
```

#### 8. API Route → Cache Service (Store Result)

**Function Call:**
```python
cache_service.cache_result(
    image_hash='a1b2c3d4e5f6g7h8i9j0',
    ocr_data={
        'image_path': '/tmp/tmpxyz123.jpg',
        'image_dimensions': [1920, 1080],
        'all_text': results.get('all_text', ''),
        'filename': 'cd_collection.jpg',
        'text_scale': 100
    }
)
```

#### 9. API Route → Frontend (Final Response)

**HTTP Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "success": true,
  "total_matches": 2,
  "matches": {
    "Billy Joel": [
      {
        "text": "BILLY JOEL",
        "score": 100.0,
        "match_type": "complete_phrase",
        "angle": 0,
        "is_spanning": false
      }
    ],
    "U2": [
      {
        "text": "U2",
        "score": 100.0,
        "match_type": "complete_phrase",
        "angle": 0,
        "is_spanning": false
      }
    ]
  },
  "processing_time_ms": 1250.5,
  "image_dimensions": [1920, 1080],
  "annotated_image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a...",
  "all_detected_text": "BILLY JOEL Greatest Hits Volume I & II\nU2 The Joshua Tree\nJEWEL Pieces of You...",
  "filename": "cd_collection.jpg",
  "cached": false
}
```

---

### Example 2: Cached Threshold Update Flow

#### 1. Frontend → API Route (Same Image, Different Threshold)

**HTTP Request:**
```http
POST /ocr/upload HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data

[Same image data]
search_phrases: ["Billy Joel", "U2"]
threshold: 85  # Changed from 75
text_scale: 100
```

#### 2. API Route → Cache Service

**Function Call:**
```python
image_hash = cache_service.get_image_hash(image_data, text_scale=100)
# Returns: "a1b2c3d4e5f6g7h8i9j0" (same hash)

cached_result = cache_service.get_cached_result(image_hash)
# Returns: {cached data} (cache hit!)
```

**Cache Hit Response:**
```python
{
    'image_path': '/tmp/tmpxyz123.jpg',  # Existing temp file
    'image_dimensions': [1920, 1080],
    'all_text': 'BILLY JOEL Greatest Hits...',
    'filename': 'cd_collection.jpg',
    'text_scale': 100
}
```

#### 3. API Route → OCR Service (Skip Google API, Use Cached Image)

**Function Call:**
```python
# Use cached image path - NO Google Cloud Vision API call!
results = ocr_service.detect_phrases(
    image_path='/tmp/tmpxyz123.jpg',  # From cache
    search_phrases=['Billy Joel', 'U2'],
    threshold=85,  # New threshold
    text_scale=100,
    show_plot=False
)
```

**Processing Time Comparison:**
- Without cache: ~2500ms (Google API call)
- With cache: ~500ms (local processing only)

#### 4. API Route → Frontend (Fast Response)

**HTTP Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "success": true,
  "total_matches": 1,  # Fewer matches due to stricter threshold
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
  "processing_time_ms": 485.2,  # Much faster!
  "image_dimensions": [1920, 1080],
  "annotated_image_base64": "...",
  "all_detected_text": "BILLY JOEL Greatest Hits...",
  "filename": "cd_collection.jpg",
  "cached": true  # Indicates cache was used
}
```

---

### Example 3: Health Check Flow

#### 1. Frontend → API Route

**HTTP Request:**
```http
GET /health HTTP/1.1
Host: localhost:8000
```

#### 2. API Route → OCR Service

**Function Call:**
```python
# In backend/api/routes/health.py

is_available = ocr_service.is_available()
```

**Response:**
```python
True  # or False if OCR module not loaded
```

#### 3. API Route → Frontend

**HTTP Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.123456",
  "service": "ThriftAssist OCR API",
  "ocr_available": true
}
```

---

### Example 4: Cache Status Query

#### 1. Frontend → API Route

**HTTP Request:**
```http
GET /cache/status HTTP/1.1
Host: localhost:8000
```

#### 2. API Route → Cache Service

**Function Call:**
```python
# In backend/api/routes/cache.py

status = cache_service.get_cache_status()
```

**Response:**
```python
{
    'cache_size': 5,
    'max_cache_size': 100,
    'entries': [
        {
            'hash': 'a1b2c3d4...',
            'timestamp': 1705318200.0,
            'age_seconds': 120.5
        },
        {
            'hash': 'e5f6g7h8...',
            'timestamp': 1705318150.0,
            'age_seconds': 170.3
        },
        # ... more entries
    ]
}
```

#### 3. API Route → Frontend

**HTTP Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "success": true,
  "cache_size": 5,
  "max_cache_size": 100,
  "entries": [
    {
      "hash": "a1b2c3d4...",
      "timestamp": 1705318200.0,
      "age_seconds": 120.5
    },
    {
      "hash": "e5f6g7h8...",
      "timestamp": 1705318150.0,
      "age_seconds": 170.3
    }
  ]
}
```

---

### Example 5: Error Handling Flow

#### 1. Frontend → API Route (Invalid Image)

**HTTP Request:**
```http
POST /ocr/upload HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data

[corrupted image data]
search_phrases: ["Billy Joel"]
threshold: 75
text_scale: 100
```

#### 2. API Route → Image Service

**Function Call:**
```python
is_valid = image_service.validate_image_data(image_data, max_size_mb=10)
```

**Response:**
```python
False  # Invalid image
```

#### 3. API Route → Frontend (Error Response)

**HTTP Response:**
```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "detail": "Invalid or oversized image"
}
```

---

### Example 6: Google Cloud Vision API Communication

#### Core OCR Module → Google Cloud Vision

**API Call:**
```python
# In thriftassist_googlevision.py

from google.cloud import vision

client = vision.ImageAnnotatorClient()

with open(image_path, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.document_text_detection(image=image)
```

**Google Cloud Vision Response:**
```python
{
    'text_annotations': [
        {
            'description': 'BILLY JOEL Greatest Hits Volume I & II\nU2 The Joshua Tree\nJEWEL Pieces of You',
            'bounding_poly': {
                'vertices': [
                    {'x': 10, 'y': 20},
                    {'x': 890, 'y': 20},
                    {'x': 890, 'y': 650},
                    {'x': 10, 'y': 650}
                ]
            }
        },
        {
            'description': 'BILLY',
            'bounding_poly': {
                'vertices': [
                    {'x': 45, 'y': 235},
                    {'x': 125, 'y': 235},
                    {'x': 125, 'y': 265},
                    {'x': 45, 'y': 265}
                ]
            }
        },
        {
            'description': 'JOEL',
            'bounding_poly': {
                'vertices': [
                    {'x': 135, 'y': 235},
                    {'x': 195, 'y': 235},
                    {'x': 195, 'y': 265},
                    {'x': 135, 'y': 265}
                ]
            }
        },
        # ... more text annotations
    ],
    'error': {
        'message': ''  # Empty if no error
    }
}
```

---

## Message Flow Sequence Diagrams

### Diagram 1: New Image Upload (Cache Miss)

```
Frontend         API Route       Cache Service    OCR Service      Core OCR       Google Vision
   │                │                  │               │               │                │
   ├─POST /upload──>│                  │               │               │                │
   │                ├─get_hash()──────>│               │               │                │
   │                │<─hash────────────┤               │               │                │
   │                ├─get_cached()────>│               │               │                │
   │                │<─None (miss)─────┤               │               │                │
   │                ├─detect_phrases()─────────────────>│               │                │
   │                │                  │               ├─annotate()────>│                │
   │                │                  │               │               ├─detect()───────>│
   │                │                  │               │               │<─text_data─────┤
   │                │                  │               │<─results──────┤                │
   │                │<─formatted────────────────────────┤               │                │
   │                ├─cache_result()──>│               │               │                │
   │                │<─ok──────────────┤               │               │                │
   │<─JSON response─┤                  │               │               │                │
   │                │                  │               │               │                │
```

### Diagram 2: Threshold Update (Cache Hit)

```
Frontend         API Route       Cache Service    OCR Service      Core OCR
   │                │                  │               │               │
   ├─POST /upload──>│                  │               │               │
   │  (same image)  ├─get_hash()──────>│               │               │
   │                │<─hash────────────┤               │               │
   │                ├─get_cached()────>│               │               │
   │                │<─cached_data─────┤               │               │
   │                ├─detect_phrases()─────────────────>│               │
   │                │  (cached path)   │               ├─annotate()────>│
   │                │                  │               │  (skip API)   │
   │                │                  │               │<─results──────┤
   │                │<─formatted────────────────────────┤               │
   │<─JSON response─┤                  │               │               │
   │  (fast! ~500ms)│                  │               │               │
```
