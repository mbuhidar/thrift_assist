# ThriftAssist Text Detection Web Application

A user-friendly web interface for the ThriftAssist Text Detection API that allows users to upload images of physical media like books, CDs, and movies, search for specific phrases, and view annotated results to aid in locating media of interest.

## Features

- 📤 **Image upload** - Support for JPG, PNG, GIF, BMP formats
- 🔍 **Phrase search** - Enter multiple search phrases separated by commas
- 🎛️ **Adjustable threshold** - Fine-tune detection sensitivity (50-100%)
- 📊 **Detailed results** - View match statistics and processing time
- 🖼️ **Annotated images** - See visual annotations highlighting found phrases
- 📱 **Responsive design** - Works on desktop and mobile devices

## Quick Start

1. **Start the API service:**
   ```bash
   ./start_web_service.sh
   ```

2. **Open the web application:**
   - Open `web_app.html` in your web browser
   - Or use a local web server: `python -m http.server 8080`

3. **Use the application:**
   - Upload an image containing text
   - Enter search phrases (comma-separated)
   - Adjust threshold if needed (default: 80%)
   - Click "Analyze Image"
   - View results and annotated image

## Example Usage

**Search Phrases:**
```
Homecoming, Lee Child, Circle of Three, David Foster Wallace
```

**Threshold:** 80% (recommended for most use cases)

## API Integration

The web app communicates with the FastAPI service running on `http://localhost:8000`. Make sure the API service is running before using the web application.

### Endpoints Used:
- `POST /upload-and-detect` - Main endpoint for file upload and phrase detection

## Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Troubleshooting

**"Failed to connect to API" error:**
- Ensure the FastAPI service is running on port 8000
- Check that `start_web_service.sh` completed successfully

**"No matches found" results:**
- Try lowering the threshold (e.g., 60-70%)
- Check spelling of search phrases
- Ensure image quality is sufficient for OCR

**Large image upload issues:**
- Compress images if they're very large (>10MB)
- Use common formats (JPG, PNG)

## Features Explained

### Detection Threshold
- **90-100%:** Very strict matching, requires nearly exact text
- **80-90%:** Recommended range, handles minor OCR errors
- **60-80%:** More lenient, useful for poor image quality
- **50-60%:** Very permissive, may have false positives

### Match Types
- **complete_phrase:** Exact substring match
- **fuzzy_phrase:** Close match with minor differences  
- **partial_phrase:** Partial text match
- **upside_down:** Text detected in rotated/flipped orientation

### Visual Annotations
- Green bounding boxes highlight detected phrases
- Labels show phrase name and confidence score
- Boxes align with text orientation (horizontal, vertical, diagonal)
