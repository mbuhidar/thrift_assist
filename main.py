"""
FastAPI Web Service for OCR Phrase Detection
Optimized for Render.com deployment
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
import socket
import subprocess
import signal

# Import the OCR detection function and its dependencies
try:
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
    OCR_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ OCR module not available: {e}")
    # Create stub functions for deployment testing
    def detect_and_annotate_phrases(*args, **kwargs):
        return {
            'total_matches': 0,
            'matches': {},
            'annotated_image': np.zeros((100, 100, 3), dtype=np.uint8),
            'all_text': 'OCR module not available'
        }
    FUZZY_AVAILABLE = False
    COMMON_WORDS = set()
    OCR_MODULE_AVAILABLE = False

app = FastAPI(
    title="ThriftAssist Text Detection API",
    description="REST API for detecting and annotating phrases in images using Google Cloud Vision API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def find_available_port(host: str, start_port: int = 8080) -> int:
    """Find the next available port starting from start_port."""
    for port in range(start_port, start_port + 100):
        if is_port_available(host, port):
            return port
    raise RuntimeError(f"No available ports found starting from {start_port}")


def kill_process_on_port(port: int) -> bool:
    """Kill any process running on the specified port."""
    try:
        # Find process using the port
        result = subprocess.run(['lsof', '-ti', f':{port}'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"💀 Killing process {pid} on port {port}")
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        # Give it a moment to terminate gracefully
                        subprocess.run(['sleep', '1'])
                        # Force kill if still running
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # Process already terminated
                    except (ProcessLookupError, ValueError) as e:
                        print(f"⚠️ Could not kill process {pid}: {e}")
            return True
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"⚠️ Could not check/kill processes on port {port}: {e}")
        return False


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
        "message": "ThriftAssist OCR API",
        "status": "running",
        "version": "1.0.0",
        "platform": "Render.com",
        "ocr_available": OCR_MODULE_AVAILABLE,
        "environment": {
            "port": os.getenv("PORT", "10000"),
            "host": "0.0.0.0",
            "render_service": os.getenv("RENDER_SERVICE_NAME", "thriftassist-api")
        },
        "endpoints": {
            "health": "GET /health",
            "docs": "GET /docs",
            "detect_phrases": "POST /detect-phrases"
        }
    }