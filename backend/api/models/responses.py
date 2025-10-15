"""
API response models.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class MatchDetail(BaseModel):
    """Details of a single match."""
    text: str
    score: float
    match_type: str
    angle: float = 0
    is_spanning: bool = False


class PhraseDetectionResponse(BaseModel):
    """Response model for phrase detection."""
    
    success: bool
    total_matches: int
    matches: Dict[str, List[MatchDetail]]
    processing_time_ms: float
    image_dimensions: Optional[List[int]] = None
    annotated_image_base64: Optional[str] = None
    all_detected_text: Optional[str] = None
    cached: bool = False
    error_message: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_matches": 3,
                "matches": {
                    "Billy Joel": [
                        {
                            "text": "BILLY JOEL",
                            "score": 100.0,
                            "match_type": "complete_phrase",
                            "angle": 0,
                            "is_spanning": False
                        }
                    ]
                },
                "processing_time_ms": 1250.5,
                "image_dimensions": [1920, 1080],
                "cached": False
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    service: str
    ocr_available: bool
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "service": "ThriftAssist OCR API",
                "ocr_available": True
            }
        }


class CacheStatusResponse(BaseModel):
    """Cache status response."""
    success: bool
    cache_size: int
    max_cache_size: int
    entries: List[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "cache_size": 5,
                "max_cache_size": 100,
                "entries": [
                    {
                        "hash": "a1b2c3d4...",
                        "timestamp": 1705318200.0,
                        "age_seconds": 120.5
                    }
                ]
            }
        }
