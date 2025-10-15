"""
Cache management endpoints.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from backend.services.cache_service import cache_service
from backend.api.models.responses import CacheStatusResponse

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/status", response_model=CacheStatusResponse)
async def get_cache_status():
    """
    Get current cache status and statistics.
    """
    status = cache_service.get_cache_status()
    
    return CacheStatusResponse(
        success=True,
        cache_size=status['cache_size'],
        max_cache_size=status['max_cache_size'],
        entries=status['entries']
    )


@router.post("/clear")
async def clear_cache():
    """
    Clear all cached OCR results.
    
    This is useful for freeing up memory or debugging.
    """
    cleared_count = cache_service.clear_cache()
    
    return JSONResponse({
        "success": True,
        "message": f"Cleared {cleared_count} cached entries"
    })
