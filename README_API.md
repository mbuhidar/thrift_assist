# ThriftAssist Web Service API

FastAPI-based web service for OCR phrase detection and annotation using Google Cloud Vision API.

## Quick Start

### Local Development
```bash
# Make startup script executable
chmod +x start_service.sh

# Start the service
./start_service.sh
```

### Direct Python
```bash
# Install dependencies
pip install -r requirements_web.txt

# Start service
python thriftassist_web_service.py
```

### Docker
```bash
cd docker
docker-compose up -d
```

## API Endpoints

### Health Check
- **GET** `/health`
- Returns service status and timestamp

### Phrase Detection (Base64)
- **POST** `/detect-phrases`
- Request body:
```json
{
  "search_phrases": ["Billy Joel", "U2", "Jewel"],
  "threshold": 75,
  "image_base64": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

### Phrase Detection with Annotation
- **POST** `/detect-phrases-with-annotation`
- Same request as above, but returns annotated image as base64

### File Upload Detection
- **POST** `/upload-and-detect`
- Form data:
  - `file`: Image file
  - `search_phrases`: JSON array string `["phrase1", "phrase2"]`
  - `threshold`: Integer (50-100)

## Response Format

```json
{
  "success": true,
  "total_matches": 3,
  "matches": {
    "Billy Joel": [
      {
        "text": "Billy Joel",
        "score": 95.0,
        "match_type": "complete_phrase",
        "angle": 0
      }
    ]
  },
  "processing_time_ms": 1250.5,
  "image_dimensions": [1920, 1080],
  "annotated_image_base64": "data:image/jpeg;base64,..."
}
```

## Interactive Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Environment Variables

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud credentials JSON
