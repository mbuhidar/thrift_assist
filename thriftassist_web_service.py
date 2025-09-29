"""
FastAPI Web Service for OCR Phrase Detection
Provides REST API endpoints for phrase detection and annotation using Google Cloud Vision API.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import cv2
import numpy as np
import base64
import io
import os
import tempfile
import json
from datetime import datetime
import uuid

# Import the OCR detection function and its dependencies
from thriftassist_googlevision import (
    detect_and_annotate_phrases,
    suppress_stderr_warnings,
    group_text_into_lines,
    find_complete_phrases,
    draw_phrase_annotations,
    normalize_text_for_search,
    is_meaningful_phrase,
    try_reverse_text_matching,
    calculate_text_angle,
    FUZZY_AVAILABLE,
    COMMON_WORDS
)

app = FastAPI(
    title="ThriftAssist Text Detection API",
    description="REST API for detecting and annotating phrases in images using Google Cloud Vision API",
    version="1.0.0"
)

# Enable CORS for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data models
class PhraseDetectionRequest(BaseModel):
    search_phrases: List[str]
    threshold: Optional[int] = 75
    image_base64: Optional[str] = None


class PhraseDetectionResponse(BaseModel):
    success: bool
    total_matches: int
    matches: Dict[str, Any]
    processing_time_ms: float
    image_dimensions: Optional[tuple] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str


# Helper functions
def image_to_base64(image_array: np.ndarray) -> str:
    """Convert OpenCV image array to base64 string."""
    _, buffer = cv2.imencode('.jpg', image_array)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64


def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image array."""
    # Remove data URL prefix if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def save_temp_image(image: np.ndarray) -> str:
    """Save image to temporary file and return path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_file.name, image)
    return temp_file.name


# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service="OCR Phrase Detection API"
    )


@app.post("/detect-phrases", response_model=PhraseDetectionResponse)
async def detect_phrases_endpoint(request: PhraseDetectionRequest):
    """
    Detect phrases in image provided as base64 string.
    """
    start_time = datetime.now()
    
    try:
        if not request.image_base64:
            raise HTTPException(status_code=400, detail="image_base64 is required")
        
        if not request.search_phrases:
            raise HTTPException(status_code=400, detail="search_phrases list cannot be empty")
        
        # Convert base64 to image
        image = base64_to_image(request.image_base64)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Save to temporary file
        temp_image_path = save_temp_image(image)
        
        try:
            # Run phrase detection
            results = detect_and_annotate_phrases(
                image_path=temp_image_path,
                search_phrases=request.search_phrases,
                threshold=request.threshold,
                show_plot=False  # Disable matplotlib display for web service
            )
            
            if results is None:
                return PhraseDetectionResponse(
                    success=False,
                    total_matches=0,
                    matches={},
                    processing_time_ms=0,
                    error_message="Failed to process image or no text detected"
                )
            
            # Convert matches to serializable format
            serializable_matches = {}
            for phrase, matches in results['matches'].items():
                serializable_matches[phrase] = []
                for match_data, score, match_type in matches:
                    serializable_matches[phrase].append({
                        'text': match_data.get('text', ''),
                        'score': float(score),
                        'match_type': match_type,
                        'angle': match_data.get('angle', 0)
                    })
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return PhraseDetectionResponse(
                success=True,
                total_matches=results['total_matches'],
                matches=serializable_matches,
                processing_time_ms=processing_time,
                image_dimensions=(image.shape[1], image.shape[0])
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
    
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return PhraseDetectionResponse(
            success=False,
            total_matches=0,
            matches={},
            processing_time_ms=processing_time,
            error_message=str(e)
        )


@app.post("/detect-phrases-with-annotation")
async def detect_phrases_with_annotation(request: PhraseDetectionRequest):
    """
    Detect phrases and return both results and annotated image as base64.
    """
    start_time = datetime.now()
    
    try:
        if not request.image_base64:
            raise HTTPException(status_code=400, detail="image_base64 is required")
        
        if not request.search_phrases:
            raise HTTPException(status_code=400, detail="search_phrases list cannot be empty")
        
        # Convert base64 to image
        image = base64_to_image(request.image_base64)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Save to temporary file
        temp_image_path = save_temp_image(image)
        
        try:
            # Run phrase detection
            results = detect_and_annotate_phrases(
                image_path=temp_image_path,
                search_phrases=request.search_phrases,
                threshold=request.threshold,
                show_plot=False
            )
            
            if results is None:
                return JSONResponse({
                    "success": False,
                    "total_matches": 0,
                    "matches": {},
                    "processing_time_ms": 0,
                    "error_message": "Failed to process image or no text detected"
                })
            
            # Convert matches to serializable format
            serializable_matches = {}
            for phrase, matches in results['matches'].items():
                serializable_matches[phrase] = []
                for match_data, score, match_type in matches:
                    serializable_matches[phrase].append({
                        'text': match_data.get('text', ''),
                        'score': float(score),
                        'match_type': match_type,
                        'angle': match_data.get('angle', 0)
                    })
            
            # Convert annotated image to base64
            annotated_image_base64 = image_to_base64(results['annotated_image'])
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return JSONResponse({
                "success": True,
                "total_matches": results['total_matches'],
                "matches": serializable_matches,
                "processing_time_ms": processing_time,
                "image_dimensions": [image.shape[1], image.shape[0]],
                "annotated_image_base64": annotated_image_base64,
                "all_detected_text": results.get('all_text', '')
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
    
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return JSONResponse({
            "success": False,
            "total_matches": 0,
            "matches": {},
            "processing_time_ms": processing_time,
            "error_message": str(e)
        })


@app.post("/upload-and-detect")
async def upload_and_detect_phrases(
    file: UploadFile = File(...),
    search_phrases: str = Form(...),
    threshold: int = Form(75)
):
    """
    Upload image file and detect phrases.
    """
    start_time = datetime.now()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Parse search phrases (expecting JSON string)
        try:
            phrases_list = json.loads(search_phrases)
            if not isinstance(phrases_list, list):
                raise ValueError("search_phrases must be a JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid search_phrases format: {str(e)}")
        
        # Read and process image
        image_data = await file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Save to temporary file
        temp_image_path = save_temp_image(image)
        
        try:
            # Run phrase detection
            results = detect_and_annotate_phrases(
                image_path=temp_image_path,
                search_phrases=phrases_list,
                threshold=threshold,
                show_plot=False
            )
            
            if results is None:
                return JSONResponse({
                    "success": False,
                    "total_matches": 0,
                    "matches": {},
                    "processing_time_ms": 0,
                    "error_message": "Failed to process image or no text detected"
                })
            
            # Convert matches to serializable format
            serializable_matches = {}
            for phrase, matches in results['matches'].items():
                serializable_matches[phrase] = []
                for match_data, score, match_type in matches:
                    serializable_matches[phrase].append({
                        'text': match_data.get('text', ''),
                        'score': float(score),
                        'match_type': match_type,
                        'angle': match_data.get('angle', 0)
                    })
            
            # Convert annotated image to base64
            annotated_image_base64 = image_to_base64(results['annotated_image'])
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return JSONResponse({
                "success": True,
                "total_matches": results['total_matches'],
                "matches": serializable_matches,
                "processing_time_ms": processing_time,
                "image_dimensions": [image.shape[1], image.shape[0]],
                "annotated_image_base64": annotated_image_base64,
                "all_detected_text": results.get('all_text', ''),
                "filename": file.filename
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
    
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return JSONResponse({
            "success": False,
            "total_matches": 0,
            "matches": {},
            "processing_time_ms": processing_time,
            "error_message": str(e)
        })


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "OCR Phrase Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health - Health check",
            "detect_phrases": "POST /detect-phrases - Detect phrases from base64 image",
            "detect_with_annotation": "POST /detect-phrases-with-annotation - Detect phrases and return annotated image",
            "upload_and_detect": "POST /upload-and-detect - Upload file and detect phrases"
        },
        "documentation": "/docs"
    }

if __name__ == "__main__":
    # Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    print("🚀 Starting OCR Phrase Detection API Server")
    print(f"📍 Server will run on http://{HOST}:{PORT}")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Interactive API: http://localhost:8000/redoc")
    
    uvicorn.run(
        "thriftassist_web_service:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )