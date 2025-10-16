"""
FastAPI application entry point.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import sys
import gc

# Memory optimization: Set environment variables before imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # For Apple Silicon
os.environ['OMP_NUM_THREADS'] = '2'  # Limit OpenMP threads
os.environ['OPENBLAS_NUM_THREADS'] = '2'  # Limit OpenBLAS threads
os.environ['MKL_NUM_THREADS'] = '2'  # Limit MKL threads
os.environ['VECLIB_MAXIMUM_THREADS'] = '2'  # Limit vecLib threads
os.environ['NUMEXPR_NUM_THREADS'] = '2'  # Limit NumExpr threads

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core.config import settings
from backend.core.credentials import setup_google_credentials
from backend.api.routes import health, ocr, cache

# Setup Google Cloud credentials at startup
setup_google_credentials()

# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
public_dir = os.path.join(project_root, "frontend", "public")
if not os.path.exists(public_dir):
    # Try alternate location
    public_dir = os.path.join(project_root, "public")

if os.path.exists(public_dir):
    app.mount("/static", StaticFiles(directory=public_dir), name="static")
else:
    # Create directory if it doesn't exist
    os.makedirs(public_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=public_dir), name="static")

# Include routers
app.include_router(health.router)
app.include_router(ocr.router)
app.include_router(cache.router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup with memory optimization."""
    print("🚀 Starting ThriftAssist API...")
    print("💾 Memory optimization: Enabled")
    
    # Force garbage collection at startup
    gc.collect()
    
    # Lazy load OCR service (don't initialize until first request)
    from backend.services.ocr_service import ocr_service
    print(f"📦 OCR Service: {'Available' if ocr_service.is_available() else 'Not Available'}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🛑 Shutting down ThriftAssist API...")
    
    # Clear caches and force garbage collection
    from backend.services.cache_service import cache_service
    cache_service.clear_cache()
    
    gc.collect()


@app.middleware("http")
async def cleanup_after_request(request, call_next):
    """Force garbage collection after each request to minimize memory footprint."""
    response = await call_next(request)
    
    # Cleanup after processing
    gc.collect()
    
    return response


# Backward compatibility routes (redirect old endpoints to new ones)
@app.post("/upload-and-detect")
async def upload_and_detect_legacy(
    file: UploadFile = File(...),
    search_phrases: str = Form(...),
    threshold: int = Form(settings.DEFAULT_THRESHOLD),
    text_scale: int = Form(settings.DEFAULT_TEXT_SCALE),
    max_image_width: int = Form(2560)
):
    """Legacy endpoint - redirects to /ocr/upload"""
    from backend.api.routes.ocr import upload_and_detect
    return await upload_and_detect(file, search_phrases, threshold, text_scale, max_image_width)


@app.get("/")
async def root():
    """Serve the web application."""
    web_app_path = os.path.join(public_dir, "web_app.html")
    
    if os.path.exists(web_app_path):
        return FileResponse(web_app_path)
    else:
        # Return API information if web app not found
        from backend.services.ocr_service import ocr_service
        
        return JSONResponse({
            "message": "ThriftAssist OCR API",
            "status": "running",
            "version": settings.API_VERSION,
            "ocr_available": ocr_service.is_available(),
            "endpoints": {
                "health": "GET /health",
                "docs": "GET /docs",
                "upload": "POST /ocr/upload",
                "upload_legacy": "POST /upload-and-detect (deprecated)",
                "detect": "POST /ocr/detect",
                "cache_status": "GET /cache/status"
            }
        })


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon or return 204 No Content."""
    favicon_path = os.path.join(public_dir, "favicon.ico")
    
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    else:
        return JSONResponse(content={}, status_code=204)
