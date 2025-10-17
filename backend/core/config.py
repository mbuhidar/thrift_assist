"""
Application configuration management.
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_TITLE: str = "ThriftAssist Text Detection API"
    API_DESCRIPTION: str = "REST API for detecting and annotating phrases in images using Google Cloud Vision API"
    API_VERSION: str = "1.0.0"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]
    
    # Cache Settings
    MAX_CACHE_SIZE: int = 100
    CACHE_EXPIRY_SECONDS: int = 3600
    
    # Google Cloud Settings
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS_JSON: Optional[str] = None
    GOOGLE_CREDENTIALS_BASE64: Optional[str] = None
    
    # Default Processing Settings
    DEFAULT_THRESHOLD: int = 75
    DEFAULT_TEXT_SCALE: int = 100
    
    # File Upload Settings
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/gif", "image/bmp"]
    
    # Memory optimization settings
    MAX_UPLOAD_SIZE_MB: int = 10  # Reduce from default if higher
    MAX_CACHE_SIZE_MB: int = 50  # Limit cache size
    CACHE_TTL_SECONDS: int = 300  # 5 minutes - shorter cache TTL
    ENABLE_MEMORY_OPTIMIZATION: bool = True
    GC_AFTER_REQUEST: bool = True
    
    # EasyOCR optimization
    EASYOCR_GPU: bool = False  # Disable GPU to save memory
    EASYOCR_QUANTIZE: bool = True  # Use quantized models
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()
