"""
Health check and status endpoints.
"""

from fastapi import APIRouter
from datetime import datetime
from backend.services.ocr_service import ocr_service
from backend.api.models.responses import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service="ThriftAssist OCR API",
        ocr_available=ocr_service.is_available()
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check for container orchestration."""
    is_ready = ocr_service.is_available()
    
    return {
        "ready": is_ready,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ocr": ocr_service.is_available()
        }
    }
