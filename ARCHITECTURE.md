# ThriftAssist OCR - Architecture Documentation

## Overview

ThriftAssist is a modular OCR system for detecting and annotating phrases in images using Google Cloud Vision API. The application has been fully refactored from a monolithic script into a clean, layered architecture.

## Architecture Status

### ✅ Migration Complete: Modular Architecture Active

**Current (Recommended):**
- `vision/` - Modular OCR package (primary implementation)
- `utils/` - Shared utilities
- `config/` - Configuration management

**Legacy (Compatibility Layer):**
- `thriftassist_googlevision.py` - Wrapper providing backward compatibility
  - **Status**: Wrapper only - redirects to modular implementation
  - **All functionality**: Now powered by vision package
  - **Deprecation**: Shows warnings, will be removed in future release

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Applications                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Command Line Interface (vision_demo.py)            │   │
│  │ • Python Scripts (import vision package)             │   │
│  │ • Future: REST API / Web Interface                   │   │
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
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        ┌───────────────────┐               ┌───────────────────┐
        │  Utils Package    │               │  Config Package   │
        │                   │               │                   │
        │ • text_utils.py   │               │ • vision_config.py│
        │   - normalize()   │               │   - VisionConfig  │
        │   - is_meaningful │               │   - credentials   │
        │                   │               │   - thresholds    │
        │ • geometry_utils  │               │   - common_words  │
        │   - calc_angle()  │               └───────────────────┘
        │   - overlap()     │
        │   - find_pos()    │
        └───────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   External Services                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Google Cloud Vision API                              │   │
│  │ • document_text_detection()                          │   │
│  │ • text_detection()                                   │   │
│  │ • Multi-angle text recognition                       │   │
│  │ • Bounding box coordinates                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
thrift_assist/
├── __init__.py                       # Package root
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
├── credentials/                      # API credentials (gitignored)
│   └── *.json                        # Google Cloud credentials
│
├── vision_demo.py                    # Demo/CLI application
├── setup.py                          # Package installation
├── requirements.txt                  # Python dependencies
├── ARCHITECTURE.md                   # This file
└── README.md                         # User documentation
```

## Component Descriptions

### 1. Vision Package (`vision/`)

The core OCR functionality organized into specialized modules.

#### **VisionPhraseDetector** (`vision/detector.py`)

**Responsibilities:**
- Main entry point for phrase detection
- Google Cloud Vision API integration
- Orchestrates grouping, matching, and annotation
- Results visualization

**Key Methods:**
```python
class VisionPhraseDetector:
    def __init__(self, config: VisionConfig = None)
    
    def detect(
        image_path: str,
        search_phrases: List[str],
        threshold: int = None,
        show_plot: bool = True,
        text_scale: int = None
    ) -> Optional[Dict]
    
    def _detect_text(image_path: str) -> TextAnnotations
    def _print_orientation_info(text_lines: List[Dict])
    def _show_results(annotated, phrase_matches)
```

**Dependencies:**
- `VisionConfig` - Configuration settings
- `TextLineGrouper` - Line grouping logic
- `PhraseMatcher` - Phrase detection logic
- `ImageAnnotator` - Visual annotation
- `google.cloud.vision` - Google Cloud API

---

#### **TextLineGrouper** (`vision/grouper.py`)

**Responsibilities:**
- Groups individual word annotations into logical text lines
- Handles multi-angle text (horizontal, vertical, diagonal, upside-down)
- Position-based proximity grouping

**Key Methods:**
```python
class TextLineGrouper:
    def __init__(self, angle_tolerance: int = 15)
    
    def group(text_annotations) -> List[Dict]
    
    def _group_by_angle(annotations) -> Dict[float, List]
    def _group_by_position(annotations, angle: float) -> Dict[float, List]
    def _calculate_position(annotation, angle: float) -> Tuple[float, float]
    def _convert_to_text_lines(lines: Dict, angle: float) -> List[Dict]
```

**Algorithm:**
1. Calculate text angle for each annotation
2. Group by normalized angle (±tolerance)
3. Within angle groups, cluster by position
4. Convert clusters to structured text lines

**Output Format:**
```python
[
    {
        'text': 'Billy Joel',
        'annotations': [<Annotation>, <Annotation>],
        'y_position': 245.0,
        'angle': 0.0
    },
    ...
]
```

---

#### **PhraseMatcher** (`vision/matcher.py`)

**Responsibilities:**
- Finds exact and fuzzy phrase matches
- Handles multi-line spanning phrases
- Case-insensitive matching with OCR error correction
- Deduplicates matches

**Key Methods:**
```python
class PhraseMatcher:
    def __init__(self, config: VisionConfig)
    
    def find_matches(
        phrase: str,
        text_lines: List[Dict],
        full_text: str,
        threshold: int = 85
    ) -> List[Tuple[Dict, float, str]]
    
    def _search_in_lines(...) -> List
    def _search_spanning(...) -> List
    def _calculate_similarity(phrase: str, text: str) -> float
    def _deduplicate_matches(matches: List) -> List
```

**Matching Types:**
- `complete_phrase` - Exact substring match (100%)
- `fuzzy_phrase` - Fuzzy match (threshold-dependent)
- `upside_down` - Reversed text match (>95%)
- `exact_spanning` - Multi-line exact match (100%)
- `fuzzy_spanning` - Multi-line fuzzy match (70-99%)

**Fuzzy Matching Features:**
- Uses RapidFuzz library
- `token_set_ratio` for word-order flexibility
- Reverse matching for upside-down text
- Character reversal detection

---

#### **ImageAnnotator** (`vision/annotator.py`)

**Responsibilities:**
- Draws bounding boxes around detected phrases
- Smart label placement avoiding overlaps
- Leader lines when labels can't be placed nearby
- Scalable text sizing

**Key Methods:**
```python
class ImageAnnotator:
    def __init__(self, text_scale: int = 100)
    
    def draw_annotations(
        image,
        phrase_matches: Dict,
        phrase_colors: Dict = None
    ) -> np.ndarray
    
    def _extract_bbox(match_data: Dict, phrase: str) -> Tuple
    def _draw_label(image, phrase, score, bbox, color, positions)
    def _find_label_position(bbox, text_w, text_h, img_shape, labels) -> Tuple
```

**Annotation Features:**
- Color-coded bounding boxes per phrase
- Tight bbox extraction (exact word boundaries)
- Label collision avoidance
- Automatic leader line generation
- Configurable text scale (50-200%)

---

### 2. Utils Package (`utils/`)

#### **text_utils.py**

**Functions:**
```python
def normalize_text_for_search(text: str) -> str
    """
    Normalize text for matching:
    - Lowercase conversion
    - Whitespace normalization
    - OCR artifact correction ('rn' → 'm', '|' → 'I', etc.)
    """

def is_meaningful_phrase(phrase: str, common_words: Set[str]) -> bool
    """
    Filter common words:
    - Single words: reject if common
    - Multi-word: require at least one non-common word
    - Allow repeated words (e.g., "The The")
    """
```

**OCR Corrections:**
```python
char_map = {
    '|': 'I',    # Vertical bar → I
    '¡': '!',    # Inverted exclamation
    '∩': 'n',    # Upside-down u
    'ɹ': 'r',    # Upside-down r
    'ɐ': 'a',    # Upside-down a
    'ǝ': 'e',    # Upside-down e
    'ᴍ': 'w',    # Small caps M
    'rn': 'm',   # Two chars misread
    'cl': 'd',   # c+l misread
    'ii': 'n',   # Two i's
}
```

---

#### **geometry_utils.py**

**Functions:**
```python
def calculate_text_angle(vertices) -> float
    """Calculate text rotation angle from bounding box vertices."""

def rectangles_overlap(rect1: Tuple, rect2: Tuple) -> bool
    """Check if two rectangles (x1,y1,x2,y2) overlap."""

def find_non_overlapping_position(
    rect: Tuple,
    existing_positions: List[Tuple],
    image_shape: Tuple
) -> Tuple
    """Find position that doesn't overlap with existing labels."""
```

**Label Positioning Algorithm:**
1. Try 12 candidate positions around bbox
2. Check overlap with bbox and existing labels
3. Adjust position using offsets
4. Fallback to stacked vertical positioning
5. Ensure within image bounds

---

### 3. Config Package (`config/`)

#### **VisionConfig** (`vision_config.py`)

**Dataclass:**
```python
@dataclass
class VisionConfig:
    # API Settings
    credentials_path: str = "credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json"
    
    # Matching Settings
    fuzz_threshold: int = 75           # Similarity threshold (0-100)
    angle_tolerance: int = 15          # Text angle grouping (degrees)
    line_proximity_tolerance: int = 20 # Line grouping distance (pixels)
    
    # Display Settings
    default_text_scale: int = 100      # Label text size (percentage)
    
    # Word Filtering
    common_words: Set[str] = field(default_factory=lambda: {...})
    
    def setup_credentials(self):
        """Set GOOGLE_APPLICATION_CREDENTIALS env var."""
```

**Configuration Usage:**
```python
# Default config
config = VisionConfig()

# Custom config
config = VisionConfig(
    fuzz_threshold=85,
    angle_tolerance=10,
    default_text_scale=120
)

detector = VisionPhraseDetector(config)
```

---

### 4. Demo Application (`vision_demo.py`)

**Purpose:** Command-line interface demonstrating vision package usage

**Features:**
- Multiple example use cases
- Book title detection
- CD/media detection
- Results visualization

**Usage:**
```bash
cd /home/mbuhidar/Code/mbuhidar/thrift_assist
python vision_demo.py
```

**Code Structure:**
```python
from vision import VisionPhraseDetector
from config import VisionConfig

def main():
    config = VisionConfig(
        fuzz_threshold=75,
        angle_tolerance=15
    )
    
    detector = VisionPhraseDetector(config)
    
    # Example 1: Books
    results = detector.detect(
        "image/books.jpg",
        ['Lee Child', 'Tom Clancy'],
        show_plot=True
    )
    
    # Example 2: CDs
    results = detector.detect(
        "image/cds.jpg",
        ['Billy Joel', 'U2'],
        show_plot=True
    )

if __name__ == "__main__":
    main()
```

---

## Data Flow

### Complete Detection Flow

```
1. Client Application
   ├─ Load image path
   ├─ Define search phrases
   └─ Set threshold/text_scale
        │
        ▼
2. VisionPhraseDetector.detect()
   ├─ Load image with cv2
   ├─ Call _detect_text()
   │   └─ Google Cloud Vision API
   │       ├─ document_text_detection()
   │       └─ Returns: TextAnnotations[]
   ├─ TextLineGrouper.group()
   │   ├─ Group by angle
   │   ├─ Group by position
   │   └─ Returns: structured text lines
   ├─ For each search phrase:
   │   └─ PhraseMatcher.find_matches()
   │       ├─ Check if meaningful phrase
   │       ├─ Search in single lines
   │       ├─ Search spanning lines
   │       └─ Returns: matches with scores
   ├─ ImageAnnotator.draw_annotations()
   │   ├─ Extract bboxes
   │   ├─ Draw rectangles
   │   ├─ Find label positions
   │   └─ Draw labels with backgrounds
   └─ Return results dict
        │
        ▼
3. Display/Use Results
   ├─ Show matplotlib plot (optional)
   ├─ Print match statistics
   └─ Access results programmatically
```

---

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
