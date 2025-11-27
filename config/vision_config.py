"""Configuration for multi-provider OCR."""

import os
from dataclasses import dataclass, field
from typing import Set

@dataclass
class VisionConfig:
    """Configuration for OCR processing with multiple provider support."""
    
    # Provider Selection
    ocr_provider: str = "google"  # Options: "google", "gemini"
    
    # Google Cloud Vision Settings
    google_credentials_path: str = (
        "credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json"
    )
    
    # Gemini Vision Settings (via Google Cloud Vertex AI)
    google_cloud_project: str = "direct-bonsai-473201-t2"
    google_cloud_location: str = "us-central1"  # Vertex AI region for Gemini
    gemini_model: str = "gemini-1.5-flash-001"  # Gemini Vision model
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        # Allow environment variable to override if set
        env_project = os.getenv('GOOGLE_CLOUD_PROJECT')
        if env_project:
            self.google_cloud_project = env_project
    
    # Legacy compatibility
    @property
    def credentials_path(self) -> str:
        """Backward compatibility property."""
        return self.google_credentials_path
    
    # Matching Settings
    fuzz_threshold: int = 75
    angle_tolerance: int = 15
    line_proximity_tolerance: int = 20
    
    # Text Scale
    default_text_scale: int = 100
    
    # Common words to filter (only when appearing alone)
    common_words: Set[str] = field(default_factory=lambda: {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'a', 'an',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
    })
    
    def setup_credentials(self):
        """Set up credentials for the selected OCR provider."""
        # Get provider from environment variable if set, ensure it's a string
        provider = str(os.getenv('OCR_PROVIDER', self.ocr_provider)).lower()
        
        if provider == "google":
            # Setup Google Cloud credentials
            if os.path.exists(self.google_credentials_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = (
                    self.google_credentials_path
                )
        elif provider == "gemini":
            # Gemini uses same credentials as Google Cloud
            if os.path.exists(self.google_credentials_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = (
                    self.google_credentials_path
                )
