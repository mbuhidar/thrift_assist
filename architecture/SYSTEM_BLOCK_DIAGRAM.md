# ThriftAssist System Block Diagram

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE LAYER                                │
│                           (public/web_app.html)                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐                 │
│  │   Input Form   │  │  Gallery View   │  │  Results Display │                 │
│  │                │  │                 │  │                  │                 │
│  │ • File Upload  │  │ • Thumbnails    │  │ • Match Summary  │                 │
│  │ • Batch Mode   │  │ • Selection     │  │ • Confidence     │                 │
│  │ • Phrases      │  │ • Preview       │  │ • Explanations   │                 │
│  │ • Threshold    │  │ • Multi-select  │  │ • Annotated Img  │                 │
│  │ • Text Scale   │  │                 │  │ • Zoom/Pan       │                 │
│  └────────┬───────┘  └────────┬────────┘  └────────┬─────────┘                 │
│           │                   │                     │                           │
│           └───────────────────┴─────────────────────┘                           │
│                               │                                                 │
│                               │ FormData POST                                   │
│                               ▼                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                │
                                │ HTTP/JSON
                                │
┌───────────────────────────────▼─────────────────────────────────────────────────┐
│                              API ENDPOINT LAYER                                  │
│                        (backend/api/routes/ocr.py)                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      /upload-and-detect                                   │   │
│  │                                                                           │   │
│  │  1. Validate Input (file type, size, parameters)                        │   │
│  │  2. Generate Cache Key (MD5 hash of image + text_scale)                 │   │
│  │  3. Check Cache ──► Cache Hit? ──Yes──► Return Cached Result            │   │
│  │         │                                        │                       │   │
│  │         No                                       └──────────────────┐    │   │
│  │         ▼                                                          │    │   │
│  │  4. Save Temp Image                                               │    │   │
│  │  5. Resize Image (if > max_image_width)                           │    │   │
│  │  6. Process OCR ──────────────────────────┐                       │    │   │
│  │  7. Format Results                        │                       │    │   │
│  │  8. Encode Image (base64)                 │                       │    │   │
│  │  9. Cache Result                          │                       │    │   │
│  │  10. Return JSON Response ◄───────────────┴───────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                │                                                 │
└────────────────────────────────┼─────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            SERVICES LAYER                                        │
│                         (backend/services/)                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌───────────────────┐  ┌──────────────────┐  ┌────────────────────┐           │
│  │  Image Service    │  │  Cache Service   │  │   OCR Service      │           │
│  │                   │  │                  │  │                    │           │
│  │ • Validate Format │  │ • MD5 Hashing    │  │ • Orchestrate      │           │
│  │ • Base64 Decode   │  │ • LRU Cache      │  │ • Format Results   │           │
│  │ • Temp File Save  │  │ • TTL Management │  │ • Call Detector    │           │
│  │ • File Cleanup    │  │ • Cache Eviction │  │ • Measure Time     │           │
│  └─────────┬─────────┘  └──────────────────┘  └──────────┬─────────┘           │
│            │                                              │                     │
│            └──────────────────────┬───────────────────────┘                     │
│                                   │                                             │
└───────────────────────────────────┼─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        IMAGE PRE-PROCESSING                                      │
│                         (OpenCV/PIL Pipeline)                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  1. Load Image (cv2.imread)                                              │   │
│  │  2. Resize if Width > max_image_width                                    │   │
│  │        • Maintain aspect ratio                                           │   │
│  │        • Optimize for display performance                                │   │
│  │  3. Prepare for Vision API (save as temp file)                           │   │
│  └──────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                                │
└─────────────────────────────────┼────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        AI MODEL PROCESSING                                       │
│                    (Google Cloud Vision API)                                    │
│                    (vision/detector.py)                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │              VisionPhraseDetector.detect()                                │   │
│  │                                                                           │   │
│  │  1. Call Google Vision API                                               │   │
│  │     • document_text_detection()                                          │   │
│  │     • Extract full text annotations                                      │   │
│  │     • Get bounding boxes & vertices                                      │   │
│  │                                                                           │   │
│  │  2. Parse Text Lines                                                     │   │
│  │     • Extract text content                                               │   │
│  │     • Calculate Y-position (vertical location)                           │   │
│  │     • Determine rotation angle                                           │   │
│  │     • Store bounding box coordinates                                     │   │
│  │                                                                           │   │
│  │  3. Build Text Structure                                                 │   │
│  │     • Sort lines by Y-position                                           │   │
│  │     • Group multi-line text                                              │   │
│  │     • Preserve spatial relationships                                     │   │
│  └──────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                                │
└─────────────────────────────────┼────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      PHRASE MATCHING ENGINE                                      │
│                       (vision/matcher.py)                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │              PhraseMatcher.find_matches()                                 │   │
│  │                                                                           │   │
│  │  For each search phrase:                                                 │   │
│  │                                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │  A. EXACT MATCHING                                               │    │   │
│  │  │     • Case-insensitive comparison                                │    │   │
│  │  │     • Direct string match                                        │    │   │
│  │  │     • 100% confidence score                                      │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  │           │                                                               │   │
│  │           ▼                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │  B. FUZZY MATCHING (RapidFuzz)                                   │    │   │
│  │  │     • Levenshtein distance                                       │    │   │
│  │  │     • Partial ratio scoring                                      │    │   │
│  │  │     • Token-based matching                                       │    │   │
│  │  │     • Similarity score (0-100)                                   │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  │           │                                                               │   │
│  │           ▼                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │  C. MULTI-LINE SPANNING                                          │    │   │
│  │  │     • Combine adjacent text lines                                │    │   │
│  │  │     • Match phrases split across lines                           │    │   │
│  │  │     • Window-based search (2-3 lines)                            │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  │           │                                                               │   │
│  │           ▼                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │  D. UPSIDE-DOWN DETECTION                                        │    │   │
│  │  │     • Check rotation angles (±180°)                              │    │   │
│  │  │     • Reverse text comparison                                    │    │   │
│  │  │     • Match inverted text                                        │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  │           │                                                               │   │
│  │           ▼                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │  E. THRESHOLD FILTERING                                          │    │   │
│  │  │     • Apply user-defined threshold (50-100%)                     │    │   │
│  │  │     • Filter low-confidence matches                              │    │   │
│  │  │     • Rank by similarity score                                   │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  │           │                                                               │   │
│  │           ▼                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │  F. DEDUPLICATION                                                │    │   │
│  │  │     • Remove duplicate matches                                   │    │   │
│  │  │     • Merge overlapping bounding boxes                           │    │   │
│  │  │     • Keep highest-confidence match                              │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                           │   │
│  └──────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                                │
└─────────────────────────────────┼────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        IMAGE ANNOTATION                                          │
│                    (vision/detector.py + OpenCV)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  1. Draw Bounding Boxes                                                  │   │
│  │     • Green rectangles around matches                                    │   │
│  │     • Line thickness based on confidence                                 │   │
│  │     • Color intensity varies by score                                    │   │
│  │                                                                           │   │
│  │  2. Add Text Labels                                                      │   │
│  │     • Match phrase text                                                  │   │
│  │     • Confidence percentage                                              │   │
│  │     • Font size scaled by text_scale parameter                           │   │
│  │     • Positioned above/below bounding box                                │   │
│  │     • Mobile boost: 4x text scale for mobile devices                     │   │
│  │                                                                           │   │
│  │  3. Handle Overlaps                                                      │   │
│  │     • Adjust label positions to avoid overlap                            │   │
│  │     • Layer annotations by confidence                                    │   │
│  │                                                                           │   │
│  │  4. Maintain Image Quality                                               │   │
│  │     • Preserve original aspect ratio                                     │   │
│  │     • Anti-aliasing for smooth rendering                                 │   │
│  └──────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                                │
└─────────────────────────────────┼────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      IMAGE ENCODING & OPTIMIZATION                               │
│                    (backend/utils/image_utils.py)                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  1. Convert to PIL Image                                                 │   │
│  │     • OpenCV BGR → RGB conversion                                        │   │
│  │     • Maintain numpy array structure                                     │   │
│  │                                                                           │   │
│  │  2. Calculate Optimal JPEG Quality                                       │   │
│  │     • Based on image dimensions                                          │   │
│  │     • Larger images: lower quality (60-75)                               │   │
│  │     • Smaller images: higher quality (85-95)                             │   │
│  │     • Balance file size vs visual quality                                │   │
│  │                                                                           │   │
│  │  3. Encode to Base64                                                     │   │
│  │     • JPEG compression with calculated quality                           │   │
│  │     • Optimization enabled                                               │   │
│  │     • Base64 encoding for JSON transport                                 │   │
│  │                                                                           │   │
│  │  4. Memory Management                                                    │   │
│  │     • Immediate cleanup of temporary objects                             │   │
│  │     • Garbage collection after encoding                                  │   │
│  └──────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                                │
└─────────────────────────────────┼────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          RESPONSE ASSEMBLY                                       │
│                      (backend/api/routes/ocr.py)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  JSON Response Structure:                                                │   │
│  │                                                                           │   │
│  │  {                                                                        │   │
│  │    "success": true,                                                      │   │
│  │    "total_matches": <count>,                                             │   │
│  │    "matches": {                                                          │   │
│  │      "phrase1": [                                                        │   │
│  │        {                                                                 │   │
│  │          "text": "matched text",                                         │   │
│  │          "score": 95.5,                                                  │   │
│  │          "match_type": "exact|fuzzy|spanning|upside_down",              │   │
│  │          "angle": 0,                                                     │   │
│  │          "bounding_box": [[x1,y1], [x2,y2], ...],                       │   │
│  │          "explanation": {                                                │   │
│  │            "confidence_level": "Very High|High|Medium|Low",             │   │
│  │            "reasoning": ["reason1", "reason2"],                          │   │
│  │            "recommendation": "explanation text"                          │   │
│  │          }                                                                │   │
│  │        }                                                                 │   │
│  │      ]                                                                   │   │
│  │    },                                                                    │   │
│  │    "processing_time_ms": 1234.5,                                         │   │
│  │    "image_dimensions": [width, height],                                  │   │
│  │    "annotated_image_base64": "base64_encoded_jpeg",                      │   │
│  │    "cached": false                                                       │   │
│  │  }                                                                        │   │
│  └──────────────────────────────┬───────────────────────────────────────────┘   │
│                                 │                                                │
└─────────────────────────────────┼────────────────────────────────────────────────┘
                                  │ JSON/HTTP
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     RESULTS PRESENTATION (UI)                                    │
│                        (public/web_app.html)                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  A. RESULTS SUMMARY                                                      │   │
│  │     • Total matches found                                                │   │
│  │     • Processing time                                                    │   │
│  │     • Average confidence                                                 │   │
│  │     • Image dimensions                                                   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  B. MATCH DETAILS (Collapsible)                                          │   │
│  │     • Grouped by search phrase                                           │   │
│  │     • Individual match cards                                             │   │
│  │     • Matched text preview                                               │   │
│  │     • Confidence badge (Very High, High, Medium, Low)                    │   │
│  │     • Similarity score percentage                                        │   │
│  │     • Rotation angle indicator                                           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  C. EXPLAINABILITY ("Why?" Button)                                       │   │
│  │     • Confidence level explanation                                       │   │
│  │     • Reasoning factors list                                             │   │
│  │     • Match type description                                             │   │
│  │     • Recommendation for user                                            │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  D. ANNOTATED IMAGE DISPLAY                                              │   │
│  │                                                                           │   │
│  │     1. Image Rendering                                                   │   │
│  │        • Decode base64 → data URL                                        │   │
│  │        • Display in <img> element                                        │   │
│  │        • Responsive sizing (mobile/tablet/desktop)                       │   │
│  │        • Maintain aspect ratio                                           │   │
│  │                                                                           │   │
│  │     2. Zoom & Pan Controls (Panzoom.js)                                  │   │
│  │        • Click to zoom in                                                │   │
│  │        • Drag to pan when zoomed                                         │   │
│  │        • Mouse wheel zoom (desktop)                                      │   │
│  │        • Pinch to zoom (mobile/tablet)                                   │   │
│  │        • Zoom buttons (+, -, reset)                                      │   │
│  │        • Zoom percentage indicator                                       │   │
│  │        • Touch gestures support                                          │   │
│  │                                                                           │   │
│  │     3. Device-Specific Optimizations                                     │   │
│  │        • Mobile: Full-width display, touch-optimized                     │   │
│  │        • Tablet: 90vw width, gesture support                             │   │
│  │        • Desktop: Card-constrained, mouse controls                       │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  E. BATCH MODE RESULTS (Multiple Images)                                 │   │
│  │     • Batch summary (total images, success count)                        │   │
│  │     • Individual image results cards                                     │   │
│  │     • Per-image annotated display                                        │   │
│  │     • Click to enlarge (modal view)                                      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

### Single Image Processing Flow
```
User Upload → API Validation → Cache Check → Image Pre-processing → 
Google Vision API → Text Extraction → Phrase Matching → Threshold Filtering → 
Image Annotation → JPEG Encoding → Base64 → JSON Response → UI Display
```

### Batch Processing Flow
```
Multiple Images Selected → Gallery Preview → User Selection → 
For Each Selected Image:
  → Same as Single Image Flow → 
Aggregate Results → Batch Summary Display → Individual Image Cards
```

## Key Performance Features

1. **Caching Layer**: MD5-based LRU cache with TTL to avoid redundant OCR processing
2. **Image Optimization**: Automatic resizing and JPEG quality adjustment for optimal transfer
3. **Memory Management**: Aggressive cleanup and garbage collection throughout pipeline
4. **Responsive Design**: Device-specific rendering for mobile, tablet, and desktop
5. **Batch Processing**: Sequential processing with progress tracking for multiple images
6. **Explainability**: Detailed reasoning for each match with confidence metrics

## Technology Stack

- **Frontend**: Vanilla JavaScript, HTML5, CSS3, Panzoom.js
- **Backend**: FastAPI (Python), Uvicorn ASGI server
- **AI/ML**: Google Cloud Vision API, RapidFuzz (fuzzy matching)
- **Image Processing**: OpenCV, PIL/Pillow
- **Caching**: In-memory LRU cache with OrderedDict
- **Data Transport**: Base64-encoded JPEG images in JSON
