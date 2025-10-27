# ThriftAssist OCR - Architecture

ThriftAssist is a FastAPI-based OCR and phrase-matching system. It detects phrases in images using Google Cloud Vision API, applies robust matching (exact, fuzzy, multi-line spanning, upside-down), annotates matches, and returns a JSON + base64 image response. The codebase is organized as a layered backend (API → Services → Vision/Utils), backed by configuration and tests.

## Repository Structure (authoritative)

```
thrift_assist/
├── main.py                         # FastAPI app entrypoint (importable as main:app)
├── requirements.txt
├── pytest.ini
├── ARCHITECTURE.md                 # This document
├── README.md
│
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── ocr.py             # /ocr/upload endpoint: upload + detect pipeline
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── cache_service.py       # LRU cache (in-memory), MD5 keys, TTL checks
│   │   ├── image_service.py       # Validation, base64 decode, temp file IO
│   │   └── ocr_service.py         # Orchestrates VisionPhraseDetector, formats results
│   │
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py         # JPEG quality, PIL resize (display-safe)
│
├── vision/
│   ├── __init__.py
│   ├── detector.py                # VisionPhraseDetector: Vision API + text line extraction
│   └── matcher.py                 # PhraseMatcher: exact/fuzzy/spanning/upside-down matching
│
├── utils/
│   ├── __init__.py
│   └── text_utils.py              # Text normalization, phrase meaningfulness checks
│
├── config/
│   ├── __init__.py
│   ├── settings.py                # Pydantic/Env-driven settings used by services/routes
│   └── vision_config.py           # VisionConfig used by detector/matcher
│
├── credentials/                   # GCP credentials (gitignored)
│
└── tests/
    ├── conftest.py
    ├── test_api/
    │   └── test_ocr_routes.py
    ├── test_backend/
    │   └── test_main.py
    ├── test_services/
    │   ├── test_cache_service.py
    │   ├── test_image_service.py
    │   └── test_ocr_service.py
    ├── test_utils/
    │   └── test_image_utils.py
    └── test_vision/
        └── test_matcher.py
```

## Architecture Block Diagram (updated)

```
Client (multipart POST /ocr/upload)
        │
        ▼
FastAPI Route (backend/api/routes/ocr.py :: upload_and_detect)
  • Parse params (search_phrases, threshold, text_scale, max_image_width)
  • Validate file (ImageService.validate_image_data)
  • Build cache key (CacheService.get_image_hash)
  • Cache hit? → return cached JSON quickly
  • Save temp image (ImageService.save_temp_image)
  • Call OCR (OCRService.detect_phrases)
  • Format matches for API (OCRService.format_matches_for_api)
  • Encode annotated image (base64 via OpenCV/PIL pipeline)
  • Cache result (CacheService.cache_result)
        │
        ▼
Services Layer
  ├─ ImageService: base64 decode, temp img save, input validation
  ├─ CacheService: LRU in-memory cache with TTL and MD5 keys
  └─ OCRService: orchestrates detector + formatting
        │
        ▼
Vision Layer
  ├─ VisionPhraseDetector.detect
  │   • Read image (cv2.imread)
  │   • Google Vision API (document_text_detection)
  │   • Extract text lines (y-position, angle, annotations)
  │   • For each search phrase → PhraseMatcher.find_matches
  │   • Aggregate matches + draw annotations (np image)
  └─ PhraseMatcher
      • Exact, fuzzy (RapidFuzz), upside-down, multi-line spanning
      • Deduplicate and rank matches
        │
        ▼
Response JSON
  { success, matches, image (base64), all_text, processing_time, cache_hit }
```

## Major Components and Function Responsibilities

### 1) Entry Point (main.py)
- app: FastAPI instance configured to include OCR routes.
- If executed as script, runs uvicorn. Importable as "main:app" (tests rely on this).

Purpose:
- Provide a single importable ASGI app.
- Support CLI execution in development/production.

### 2) API Route (backend/api/routes/ocr.py)
- upload_and_detect(file, search_phrases, threshold, text_scale, max_image_width)
  - Purpose: End-to-end handling of file upload → OCR → annotate → respond.
  - Steps:
    1) Validate file type/size via ImageService.validate_image_data.
    2) Derive hash (CacheService.get_image_hash(image_bytes, text_scale)).
    3) Check cache (CacheService.get_cached_result). On hit, short-circuit.
    4) Save image to temp file for Vision API (ImageService.save_temp_image).
    5) OCR: OCRService.detect_phrases(image_path, search_phrases, threshold, text_scale).
    6) Format matches (OCRService.format_matches_for_api).
    7) Encode annotated image (OpenCV → JPEG quality via image_utils → base64).
    8) Cache final result (CacheService.cache_result).
    9) Return JSON with matches, image, timing, cache_hit.

### 3) Services (backend/services)

- OCRService (ocr_service.py)
  - __init__():
    - Builds VisionPhraseDetector using VisionConfig and settings (threshold, text_scale).
  - detect_phrases(image_path, search_phrases, threshold=None, text_scale=None, show_plot=False)
    - Purpose: Invoke detector for phrases, measure runtime, return raw OCR results:
      - keys: matches (phrase → [(match_data, score, match_type), ...]),
              annotated_image (np array),
              all_detected_text (str)
  - format_matches_for_api(ocr_results: dict) -> dict
    - Purpose: Convert raw match tuples into API-serializable dicts per phrase:
      - [{'text', 'score', 'match_type', 'angle', 'bounding_box', ...}, ...]

- CacheService (cache_service.py)
  - get_image_hash(image_bytes: bytes, text_scale: int) -> str
    - Purpose: Generate cache key (MD5(image_bytes + text_scale)), 32-hex chars.
  - cache_result(image_hash: str, ocr_data: dict) -> None
    - Purpose: Insert into OrderedDict cache with timestamp; evict LRU if size ≥ MAX_CACHE_SIZE.
  - get_cached_result(image_hash: str) -> Optional[dict]
    - Purpose: Return cached data if present and not expired (based on CACHE_TTL_SECONDS); update LRU order.

- ImageService (image_service.py)
  - validate_image_data(file_bytes: bytes, max_size_mb=10) -> bool
    - Purpose: Ensure payload size and decodability (PIL sanity checks) before processing.
  - base64_to_array(base64_str: str) -> np.ndarray
    - Purpose: Decode base64 → bytes → np.ndarray via cv2.imdecode (BGR).
  - save_temp_image(image_array: np.ndarray) -> str
    - Purpose: Write np image to a temp .jpg file using cv2.imwrite; return path.

### 4) Vision (vision)

- VisionPhraseDetector (detector.py)
  - detect(image_path: str, search_phrases: list[str], threshold: int, text_scale: int, show_plot: bool=False) -> dict|None
    - Purpose: Central OCR pipeline:
      1) Load image (cv2.imread).
      2) Invoke Google Vision API (document_text_detection).
      3) Extract text lines (compute y-position, angle, word annotations, vertices).
      4) For each phrase, call PhraseMatcher.find_matches(...).
      5) Merge results, annotate image, produce 'matches', 'annotated_image', 'all_detected_text'.
  - Internal helpers (representative):
    - _extract_text_lines(full_text_annotation) -> list[dict]
      - Purpose: Flatten hierarchical Vision response (page/block/paragraph/word) to line dicts with geometry.
    - _build_annotations(...):
      - Purpose: Package word-level metadata (bounding boxes, vertices) for downstream annotation.

- PhraseMatcher (matcher.py)
  - find_matches(phrase: str, text_lines: list[dict], full_text: str, threshold: int) -> list[tuple]
    - Purpose: Return list of (match_data, score, match_type):
      - Uses single-line exact/fuzzy, upside-down checks, then multi-line spanning if needed.
  - _search_in_lines(...):
    - Purpose: For each line, attempt exact match or fuzz.token_set_ratio variants; assign match_type and score.
  - _search_spanning(...):
    - Purpose: If multi-word phrase, search adjacent lines (next 3, previous 2), respect y-distance/angle tolerances; generate spanning matches (exact_spanning/fuzzy_spanning).
  - _deduplicate_matches(...):
    - Purpose: Sort by score(desc), line-count(asc), spanning(last); keep best unique normalized-text matches.

### 5) Utilities

- backend/utils/image_utils.py
  - calculate_optimal_jpeg_quality(width: int) -> int
    - Purpose: Adaptive quality for encoding (small → higher quality; large → lower).
  - resize_image_for_display(image: PIL.Image, max_width: int) -> PIL.Image
    - Purpose: Maintain aspect ratio; never upscale; memory-friendly resize for visualization/response.

- utils/text_utils.py
  - normalize_text_for_search(text: str) -> str
    - Purpose: Lowercase, strip punctuation/extra whitespace; normalize for matching.
  - is_meaningful_phrase(phrase: str, common_words: set[str]) -> bool
    - Purpose: Filter phrases composed only of common stop-words.

### 6) Configuration (config)

- settings.py (env-driven; consumed by routes/services)
  - Representative fields (observed in code/tests):
    - API_TITLE
    - MAX_CACHE_SIZE
    - CACHE_TTL_SECONDS
    - DEFAULT_THRESHOLD
    - DEFAULT_TEXT_SCALE
    - MAX_UPLOAD_SIZE_MB
    - ALLOWED_IMAGE_TYPES
    - ENABLE_MEMORY_OPTIMIZATION
    - GC_AFTER_REQUEST
- vision_config.py
  - VisionConfig dataclass/obj: contains fuzz_threshold, default_text_scale, common_words, angle/line tolerances.

## Request/Processing/Data Flow

1) Client POST /ocr/upload (multipart form)
2) Route validates and hashes input (ImageService + CacheService)
3) Cache hit → format and return response
4) Cache miss → temp save → OCRService.detect_phrases
5) VisionPhraseDetector + PhraseMatcher compute matches
6) Annotate + encode image; format matches for API
7) Cache and return JSON

Response (typical):
```
{
  "success": true,
  "matches": { "some phrase": [ { "text": "...", "score": 95.5, "match_type": "fuzzy_phrase", ... } ] },
  "image": "data:image/jpeg;base64,...",
  "all_text": "full text ...",
  "processing_time": 1234.5,
  "cache_hit": false
}
```

## Testing Notes (high level)

- Tests validate:
  - CacheService: MD5 keying, LRU behavior, TTL checks.
  - ImageService: validation, base64 decode, temp file saves.
  - OCRService: tuple-to-JSON formatting; detector delegation (mocked).
  - API route: end-to-end flow with patched services; real temp file for cv2.imread in tests.
  - Vision/Matcher: exact, fuzzy, spanning, upside-down; deduplication.

## Operational Characteristics

- Caching: In-memory LRU (MAX_CACHE_SIZE; default 100). TTL in seconds (CACHE_TTL_SECONDS; default ~300s). Hash includes image bytes + text_scale.
- Performance: Vision API dominates latency; matching is linear in lines with limited spanning windows.
- Image Encoding: JPEG quality adapts to width; never upscale for display method; ensure memory-optimized path.

## Deployment

- Importable ASGI: main:app
- Example run: uvicorn main:app --host 0.0.0.0 --port 8000
- Requires GOOGLE_APPLICATION_CREDENTIALS for Vision API (file path or equivalent credentials setup).
