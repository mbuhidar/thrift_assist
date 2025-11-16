"""
API request models.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class PhraseDetectionRequest(BaseModel):
    """Request model for phrase detection."""
    
    search_phrases: List[str] = Field(..., description="List of phrases to search for")
    threshold: Optional[int] = Field(75, ge=50, le=100, description="Similarity threshold (50-100)")
    text_scale: Optional[int] = Field(100, ge=50, le=200, description="Text size scale (50-200)")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    max_image_width: int = Field(2560, description="Maximum image width")
    ocr_provider: Optional[str] = Field("google", description="OCR provider to use (google or deepseek)")
    
    @field_validator('search_phrases')
    @classmethod
    def validate_search_phrases(cls, v):
        if not v:
            raise ValueError("search_phrases cannot be empty")
        if not all(isinstance(phrase, str) and phrase.strip() for phrase in v):
            raise ValueError("All search phrases must be non-empty strings")
        return [phrase.strip() for phrase in v]
    
    class Config:
        json_schema_extra = {
            "example": {
                "search_phrases": ["Billy Joel", "U2", "Jewel"],
                "threshold": 75,
                "text_scale": 100,
                "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "max_image_width": 2560
            }
        }


class UpdateThresholdRequest(BaseModel):
    """Request model for updating detection threshold."""
    
    search_phrases: List[str]
    threshold: int = Field(..., ge=50, le=100)
    text_scale: int = Field(100, ge=50, le=200)
    image_hash: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "search_phrases": ["Billy Joel", "U2"],
                "threshold": 80,
                "text_scale": 120,
                "image_hash": "a1b2c3d4e5f6"
            }
        }
