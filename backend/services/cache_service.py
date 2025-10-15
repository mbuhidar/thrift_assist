"""
Cache management service for OCR results.
"""

import hashlib
import time
import sys
import os
from collections import OrderedDict
from typing import Optional, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core.config import settings


class CacheService:
    """Service for managing OCR result caching."""
    
    def __init__(self):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = settings.MAX_CACHE_SIZE
        self._expiry_seconds = settings.CACHE_EXPIRY_SECONDS
    
    def get_image_hash(self, image_data: bytes, text_scale: int = 100) -> str:
        """
        Generate a hash for image data combined with text_scale.
        
        Args:
            image_data: Raw image bytes
            text_scale: Text scale percentage
            
        Returns:
            MD5 hash string
        """
        combined_data = image_data + str(text_scale).encode('utf-8')
        return hashlib.md5(combined_data).hexdigest()
    
    def cache_result(self, image_hash: str, ocr_data: Dict[str, Any]) -> None:
        """
        Cache OCR result with LRU eviction.
        
        Args:
            image_hash: Hash key for the cached data
            ocr_data: OCR result data to cache
        """
        # Remove oldest entries if cache is full
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        
        # Cache the OCR data
        self._cache[image_hash] = {
            'timestamp': time.time(),
            'ocr_data': ocr_data
        }
        
        print(f"🔄 Cached OCR result for image hash: {image_hash[:8]}...")
    
    def get_cached_result(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached OCR result if available and not expired.
        
        Args:
            image_hash: Hash key for the cached data
            
        Returns:
            Cached OCR data if found and valid, None otherwise
        """
        if image_hash in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(image_hash)
            cached_data = self._cache[image_hash]
            
            # Check if cache is still valid
            if time.time() - cached_data['timestamp'] < self._expiry_seconds:
                print(f"✅ Using cached OCR result for image hash: {image_hash[:8]}...")
                return cached_data['ocr_data']
            else:
                # Remove expired cache entry
                del self._cache[image_hash]
                print(f"⏰ Cache expired for image hash: {image_hash[:8]}...")
        
        return None
    
    def clear_cache(self) -> int:
        """
        Clear all cached entries.
        
        Returns:
            Number of entries cleared
        """
        cache_size = len(self._cache)
        self._cache.clear()
        return cache_size
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get cache status information.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_info = []
        for image_hash, data in self._cache.items():
            cache_info.append({
                'hash': image_hash[:8] + '...',
                'timestamp': data['timestamp'],
                'age_seconds': time.time() - data['timestamp']
            })
        
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self._max_size,
            'entries': cache_info
        }


# Create global cache service instance
cache_service = CacheService()
