"""Configuration for Google Vision OCR."""

import os
from dataclasses import dataclass, field
from typing import Set

@dataclass
class VisionConfig:
    """Configuration for Vision API and OCR processing."""
    
    # API Settings
    credentials_path: str = "credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json"
    
    # Matching Settings
    fuzz_threshold: int = 75
    angle_tolerance: int = 15
    line_proximity_tolerance: int = 20
    
    # Text Scale
    default_text_scale: int = 100
    
    # Common words to filter (only when appearing alone)
    common_words: Set[str] = field(default_factory=lambda: {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'a', 'an', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
        'its', 'our', 'their'
    })
    
    def setup_credentials(self):
        """Set up Google Cloud credentials if available."""
        if os.path.exists(self.credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
    def setup_credentials(self):
        """Set up Google Cloud credentials if available."""
        if os.path.exists(self.credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
