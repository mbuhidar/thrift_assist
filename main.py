"""
FastAPI Web Service for OCR Phrase Detection
Optimized for Render.com deployment
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
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
import hashlib
import time
from collections import OrderedDict

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

# Render.com specific configuration
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

# Mount static files directory
try:
    app.mount("/static", StaticFiles(directory="public"), name="static")
except RuntimeError:
    # Directory doesn't exist, create it
    import os
    os.makedirs("public", exist_ok=True)
    app.mount("/static", StaticFiles(directory="public"), name="static")


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


# Configure Google Cloud credentials for web deployment
def setup_google_credentials():
    """Setup Google Cloud credentials from environment variables or files."""
    try:
        # Method 1: Direct JSON credentials from environment variable
        creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if creds_json:
            # Create temporary credentials file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(creds_json)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
            print("✅ Using credentials from GOOGLE_APPLICATION_CREDENTIALS_JSON")
            return True
        
        # Method 2: Base64 encoded credentials
        creds_b64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
        if creds_b64:
            try:
                creds_json = base64.b64decode(creds_b64).decode('utf-8')
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    f.write(creds_json)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
                print("✅ Using credentials from GOOGLE_CREDENTIALS_BASE64")
                return True
            except Exception as e:
                print(f"⚠️ Failed to decode base64 credentials: {e}")
        
        # Method 3: File path from environment
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and os.path.exists(creds_path):
            print(f"✅ Using credentials file: {creds_path}")
            return True
        
        # Method 4: Default local file
        default_path = "credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json"
        if os.path.exists(default_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(default_path)
            print(f"✅ Using default credentials: {default_path}")
            return True
        
        print("⚠️ No Google Cloud credentials found - OCR will use stub functions")
        return False
        
    except Exception as e:
        print(f"❌ Error setting up credentials: {e}")
        return False


# Setup credentials at startup
setup_google_credentials()


# Global cache for OCR results to enable real-time threshold updates
ocr_cache = OrderedDict()
MAX_CACHE_SIZE = 100  # Maximum number of cached results

def get_image_hash(image_data: bytes, text_scale: int = 100) -> str:
    """Generate a hash for image data combined with text_scale to use as cache key."""
    # Include text_scale in the hash so different text sizes are cached separately
    combined_data = image_data + str(text_scale).encode('utf-8')
    return hashlib.md5(combined_data).hexdigest()

def cache_ocr_result(image_hash: str, ocr_data: dict):
    """Cache OCR result with LRU eviction."""
    global ocr_cache
    
    # Remove oldest entries if cache is full
    while len(ocr_cache) >= MAX_CACHE_SIZE:
        ocr_cache.popitem(last=False)
    
    # Cache the OCR data
    ocr_cache[image_hash] = {
        'timestamp': time.time(),
        'ocr_data': ocr_data
    }
    
    print(f"🔄 Cached OCR result for image hash: {image_hash[:8]}...")

def get_cached_ocr_result(image_hash: str) -> Optional[dict]:
    """Retrieve cached OCR result if available."""
    if image_hash in ocr_cache:
        # Move to end (most recently used)
        ocr_cache.move_to_end(image_hash)
        cached_data = ocr_cache[image_hash]
        
        # Check if cache is still valid (1 hour expiry)
        if time.time() - cached_data['timestamp'] < 3600:
            print(f"✅ Using cached OCR result for image hash: {image_hash[:8]}...")
            return cached_data['ocr_data']
        else:
            # Remove expired cache entry
            del ocr_cache[image_hash]
            print(f"⏰ Cache expired for image hash: {image_hash[:8]}...")
    
    return None


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


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon or return 204 No Content."""
    favicon_path = "public/favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    else:
        # Return empty response to prevent 404 errors
        return JSONResponse(content={}, status_code=204)


@app.post("/upload-and-detect")
async def upload_and_detect_phrases(
    file: UploadFile = File(...),
    search_phrases: str = Form(...),
    threshold: int = Form(75),
    text_scale: int = Form(100)
):
    """
    Upload image file and detect phrases with caching support and detailed debugging.
    """
    start_time = datetime.now()
    
    try:
        print(f"🔍 Starting OCR analysis - OCR Available: {OCR_MODULE_AVAILABLE}")
        print(f"📁 File: {file.filename}, Content-Type: {file.content_type}")
        print(f"🔧 Parameters: threshold={threshold}, text_scale={text_scale}")
        print(f"📏 Text scale for annotations: {text_scale}%")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Parse search phrases (expecting JSON string)
        try:
            phrases_list = json.loads(search_phrases)
            if not isinstance(phrases_list, list):
                raise ValueError("search_phrases must be a JSON array")
            print(f"🔤 Search phrases: {phrases_list}")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid search_phrases format: {str(e)}")
        
        # Read image data and calculate hash with text_scale
        image_data = await file.read()
        image_hash = get_image_hash(image_data, text_scale)
        print(f"🔑 Image hash (with text_scale={text_scale}): {image_hash[:8]}...")
        
        # Check if we have cached OCR results for this specific text_scale
        cached_ocr = get_cached_ocr_result(image_hash)
        
        if cached_ocr:
            print(f"🚀 Using cached result with text_scale={text_scale}...")
            # Use cached image path and run detection with same text_scale
            temp_image_path = cached_ocr.get('image_path')
            if temp_image_path and os.path.exists(temp_image_path):
                print(f"📂 Using cached image: {temp_image_path}")
                results = detect_and_annotate_phrases(
                    image_path=temp_image_path,
                    search_phrases=phrases_list,
                    threshold=threshold,
                    text_scale=text_scale,
                    show_plot=False
                )
            else:
                print("❌ Cached image path invalid, processing normally...")
                cached_ocr = None
        
        if not cached_ocr:
            print("🔍 No cache found, performing OCR...")
            # Process image normally
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            print(f"📸 Image loaded: {image.shape}")
            
            # Save to temporary file
            temp_image_path = save_temp_image(image)
            print(f"💾 Temp image saved: {temp_image_path}")
            
            # Check if OCR module is actually available
            if not OCR_MODULE_AVAILABLE:
                print("❌ OCR module not available, returning error")
                return JSONResponse({
                    "success": False,
                    "total_matches": 0,
                    "matches": {},
                    "processing_time_ms": 0,
                    "error_message": "OCR module not available. Please check Google Cloud credentials and dependencies."
                })
            
            # Add detailed Google Cloud Vision debugging
            print("🔍 Running OCR phrase detection with Google Cloud Vision...")
            
            try:
                from google.cloud import vision
                print("✅ Google Cloud Vision client imported successfully")
                
                # Test the client
                client = vision.ImageAnnotatorClient()
                print("✅ Vision API client created successfully")
                
                # Read the image for Vision API
                with open(temp_image_path, 'rb') as image_file:
                    content = image_file.read()
                
                image_vision = vision.Image(content=content)
                
                # Try document text detection first
                print("📄 Attempting document_text_detection...")
                response = client.document_text_detection(image=image_vision)
                text_annotations = response.text_annotations
                
                print(f"📊 Google Cloud Vision Results:")
                print(f"   - Text annotations found: {len(text_annotations) if text_annotations else 0}")
                
                if text_annotations:
                    # Show the full text detected
                    full_text = text_annotations[0].description if text_annotations else ""
                    print(f"   - Full text length: {len(full_text)} characters")
                    print(f"   - First 200 chars: {repr(full_text[:200])}")
                    
                    # Show individual text elements
                    print(f"   - Individual elements: {len(text_annotations) - 1}")
                    for i, annotation in enumerate(text_annotations[1:6]):  # Show first 5
                        vertices = annotation.bounding_poly.vertices
                        coords = [(v.x, v.y) for v in vertices] if vertices else []
                        print(f"     [{i+1}] '{annotation.description}' at {coords}")
                    
                    if len(text_annotations) > 6:
                        print(f"     ... and {len(text_annotations) - 6} more elements")
                        
                    # Show what Google Cloud Vision detected vs what we're searching for
                    print(f"   - Google detected text contains:")
                    for phrase in phrases_list:
                        if phrase.lower() in full_text.lower():
                            print(f"     ✅ '{phrase}' found in raw text")
                        else:
                            print(f"     ❌ '{phrase}' NOT found in raw text")
                            
                else:
                    print("   - No text annotations found, trying basic text_detection...")
                    response_basic = client.text_detection(image=image_vision)
                    text_annotations = response_basic.text_annotations
                    print(f"   - Basic detection results: {len(text_annotations) if text_annotations else 0} annotations")
                
                # Check for errors in the response
                if response.error.message:
                    print(f"❌ Google Cloud Vision API error: {response.error.message}")
                
            except Exception as vision_e:
                print(f"❌ Google Cloud Vision error: {vision_e}")
                import traceback
                traceback.print_exc()
            
            # Run the full phrase detection pipeline
            print("🔍 Running phrase detection pipeline...")
            results = detect_and_annotate_phrases(
                image_path=temp_image_path,
                search_phrases=phrases_list,
                threshold=threshold,
                text_scale=text_scale,
                show_plot=False
            )
            
            print(f"📊 Final Pipeline Results:")
            if results:
                print(f"   - Success: {results is not None}")
                print(f"   - Total matches: {results.get('total_matches', 0)}")
                print(f"   - Phrases found: {list(results.get('matches', {}).keys())}")
                print(f"   - Has annotated image: {'annotated_image' in results}")
                print(f"   - All text length: {len(results.get('all_text', ''))}")
                
                # Show match details
                for phrase, matches in results.get('matches', {}).items():
                    print(f"   - '{phrase}': {len(matches)} matches")
                    for match_data, score, match_type in matches[:2]:  # Show first 2 matches
                        match_text = match_data.get('text', '')
                        print(f"     * '{match_text}' ({score:.1f}% {match_type})")
            else:
                print("   - Results: None (processing failed)")
            
            # Cache the OCR data with text_scale information
            if results:
                cache_ocr_result(image_hash, {
                    'image_path': temp_image_path,
                    'image_dimensions': [image.shape[1], image.shape[0]],
                    'all_text': results.get('all_text', ''),
                    'filename': file.filename,
                    'text_scale': text_scale  # Store text_scale in cache
                })
                print(f"💾 OCR data cached with text_scale={text_scale}")
        
        if results is None:
            print("❌ OCR processing failed")
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
        annotated_image_base64 = None
        if 'annotated_image' in results and results['annotated_image'] is not None:
            try:
                annotated_image_base64 = image_to_base64(results['annotated_image'])
                print("🖼️ Annotated image converted to base64 successfully")
            except Exception as e:
                print(f"❌ Failed to convert annotated image: {e}")
        else:
            print("⚠️ No annotated image in results")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get image dimensions from cache or results
        if cached_ocr and 'image_dimensions' in cached_ocr:
            image_dims = cached_ocr['image_dimensions']
        else:
            image_dims = [results['annotated_image'].shape[1], results['annotated_image'].shape[0]] if 'annotated_image' in results else [0, 0]
        
        response_data = {
            "success": True,
            "total_matches": results['total_matches'],
            "matches": serializable_matches,
            "processing_time_ms": processing_time,
            "image_dimensions": image_dims,
            "annotated_image_base64": annotated_image_base64,
            "all_detected_text": results.get('all_text', ''),
            "filename": file.filename,
            "cached": cached_ocr is not None
        }
        
        print(f"✅ OCR processing completed in {processing_time:.1f}ms")
        print(f"📤 Response: success={response_data['success']}, matches={response_data['total_matches']}")
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        print(f"❌ OCR processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "total_matches": 0,
            "matches": {},
            "processing_time_ms": processing_time,
            "error_message": f"Server error: {str(e)}"
        })


@app.post("/update-threshold")
async def update_threshold_endpoint(
    file: UploadFile = File(...),
    search_phrases: str = Form(...),
    threshold: int = Form(75),
    text_scale: int = Form(100),
    use_cached_ocr: str = Form("false")
):
    """
    Update annotations with new threshold using cached OCR data.
    """
    start_time = datetime.now()
    
    try:
        # Parse search phrases
        try:
            phrases_list = json.loads(search_phrases)
            if not isinstance(phrases_list, list):
                raise ValueError("search_phrases must be a JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid search_phrases format: {str(e)}")
        
        # Get image hash for cache lookup
        image_data = await file.read()
        image_hash = get_image_hash(image_data)
        
        # Try to get cached OCR result
        cached_ocr = get_cached_ocr_result(image_hash)
        
        if not cached_ocr:
            return JSONResponse({
                "success": False,
                "total_matches": 0,
                "matches": {},
                "processing_time_ms": 0,
                "error_message": "No cached OCR data found. Please re-upload the image."
            })
        
        print(f"🔄 Updating threshold to {threshold}% using cached OCR...")
        
        # Use cached image path
        temp_image_path = cached_ocr.get('image_path')
        if not temp_image_path or not os.path.exists(temp_image_path):
            return JSONResponse({
                "success": False,
                "total_matches": 0,
                "matches": {},
                "processing_time_ms": 0,
                "error_message": "Cached image not found. Please re-upload the image."
            })
        
        # Run detection with new threshold
        results = detect_and_annotate_phrases(
            image_path=temp_image_path,
            search_phrases=phrases_list,
            threshold=threshold,
            text_scale=text_scale,
            show_plot=False
        )
        
        if not results:
            return JSONResponse({
                "success": False,
                "total_matches": 0,
                "matches": {},
                "processing_time_ms": 0,
                "error_message": "Failed to process cached OCR data"
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
            "image_dimensions": cached_ocr.get('image_dimensions', [0, 0]),
            "annotated_image_base64": annotated_image_base64,
            "all_detected_text": cached_ocr.get('all_text', ''),
            "cached": True
        })
        
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

@app.get("/cache-status")
async def cache_status():
    """Get cache status information."""
    global ocr_cache
    
    cache_info = []
    for image_hash, data in ocr_cache.items():
        cache_info.append({
            'hash': image_hash[:8] + '...',
            'timestamp': data['timestamp'],
            'age_seconds': time.time() - data['timestamp']
        })
    
    return JSONResponse({
        'success': True,
        'cache_size': len(ocr_cache),
        'max_cache_size': MAX_CACHE_SIZE,
        'entries': cache_info
    })

@app.post("/clear-cache")
async def clear_cache():
    """Clear OCR cache (useful for debugging)."""
    global ocr_cache
    cache_size = len(ocr_cache)
    ocr_cache.clear()
    
    return JSONResponse({
        'success': True,
        'message': f'Cleared {cache_size} cached entries'
    })

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web application."""
    try:
        # Try to serve the web app HTML file
        return FileResponse("public/web_app.html")
    except FileNotFoundError:
        # Fallback to API info if web app not found
        return JSONResponse({
            "message": "ThriftAssist OCR API",
            "status": "running",
            "version": "1.0.0",
            "platform": "Render.com",
            "ocr_available": OCR_MODULE_AVAILABLE,
            "web_app": "Web app not found at /public/web_app.html",
            "endpoints": {
                "health": "GET /health",
                "docs": "GET /docs",
                "web": "GET /web",
                "detect_phrases": "POST /detect-phrases"
            }
        })


@app.get("/web", response_class=HTMLResponse)
async def web_app():
    """Serve the web application interface."""
    try:
        return FileResponse("public/web_app.html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Web application not found")


@app.get("/api", response_class=JSONResponse)
async def api_info():
    """API information endpoint."""
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
            "web": "GET /web",
            "detect_phrases": "POST /detect-phrases"
        }
    }
