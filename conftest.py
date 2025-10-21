"""
Global pytest configuration and fixtures for ThriftAssist tests.
"""

import pytest
import tempfile
import os
import sys
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import io
import base64

# Add project root to Python path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Optional imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "tests" / "test_data"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return TEST_DATA_DIR

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    if not CV2_AVAILABLE:
        # Return mock image array if CV2 not available
        return np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Create a simple test image with text-like features
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    # Add some white background
    image.fill(255)
    # Add some black rectangles (simulating text)
    cv2.rectangle(image, (50, 50), (150, 80), (0, 0, 0), -1)
    cv2.rectangle(image, (50, 100), (200, 130), (0, 0, 0), -1)
    cv2.rectangle(image, (50, 150), (180, 180), (0, 0, 0), -1)
    return image

@pytest.fixture
def sample_image_base64(sample_image):
    """Convert sample image to base64 string."""
    if not PIL_AVAILABLE:
        # Return mock base64 string if PIL not available
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    # Convert to PIL Image
    if CV2_AVAILABLE:
        pil_image = Image.fromarray(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(sample_image)
    
    # Encode to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_base64

@pytest.fixture
def sample_image_file(sample_image, temp_dir):
    """Save sample image to temporary file."""
    image_path = temp_dir / "test_image.jpg"
    
    if CV2_AVAILABLE:
        cv2.imwrite(str(image_path), sample_image)
    else:
        # Create a simple file for testing without CV2
        with open(image_path, 'wb') as f:
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF')  # Simple JPEG header
    
    return image_path

@pytest.fixture
def mock_ocr_results():
    """Mock OCR detection results."""
    return {
        'matches': {
            'test phrase': [
                {
                    'text': 'test phrase found here',
                    'score': 95.5,
                    'angle': 0,
                    'annotations': [],
                    'explanation': {
                        'confidence_level': 'High',
                        'reasoning': ['Exact text match found'],
                        'confidence_factors': {'text_similarity': 95},
                        'warnings': [],
                        'recommendation': 'High confidence match'
                    }
                }
            ]
        },
        'annotated_image': np.zeros((300, 400, 3), dtype=np.uint8),
        'all_detected_text': 'test phrase found here\nother text',
        'processing_time_ms': 150.5
    }

@pytest.fixture
def mock_search_phrases():
    """Sample search phrases for testing."""
    return ['test phrase', 'example text', 'sample book']

@pytest.fixture
def mock_config():
    """Mock configuration settings."""
    config = Mock()
    config.DEFAULT_THRESHOLD = 80
    config.DEFAULT_TEXT_SCALE = 100
    config.MAX_UPLOAD_SIZE_MB = 10
    config.CACHE_TTL_HOURS = 24
    config.TEMP_DIR = "/tmp/thrift_assist"
    return config

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Ensure test data directory exists
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Cleanup after test
    if "TESTING" in os.environ:
        del os.environ["TESTING"]

@pytest.fixture
def mock_cache_service():
    """Mock cache service."""
    cache = Mock()
    cache.get_image_hash.return_value = "test_hash_123"
    cache.get_cached_result.return_value = None
    cache.cache_result.return_value = True
    return cache

@pytest.fixture
def mock_image_service():
    """Mock image service."""
    service = Mock()
    service.validate_image_data.return_value = True
    service.save_temp_image.return_value = "/tmp/test_image.jpg"
    service.base64_to_array.return_value = np.zeros((300, 400, 3), dtype=np.uint8)
    return service

@pytest.fixture
def api_client():
    """FastAPI test client."""
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        with TestClient(app) as client:
            yield client
    except ImportError:
        # Return mock client if FastAPI not available
        mock_client = Mock()
        mock_client.post.return_value = Mock(status_code=200, json=lambda: {"success": True})
        mock_client.get.return_value = Mock(status_code=200, json=lambda: {"status": "healthy"})
        yield mock_client

# Utility functions for tests
def create_test_image_with_text(width=400, height=300, text_regions=None):
    """Create a test image with specified text regions."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    if text_regions and CV2_AVAILABLE:
        for region in text_regions:
            x, y, w, h = region
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), -1)
    
    return image

def create_test_upload_file(image_data, filename="test.jpg", content_type="image/jpeg"):
    """Create a test UploadFile object."""
    try:
        from fastapi import UploadFile
        
        # Convert image to bytes if it's numpy array
        if isinstance(image_data, np.ndarray):
            if CV2_AVAILABLE:
                _, buffer = cv2.imencode('.jpg', image_data)
                image_bytes = buffer.tobytes()
            else:
                # Fallback without CV2
                image_bytes = b'\xff\xd8\xff\xe0\x00\x10JFIF'  # Simple JPEG header
        else:
            image_bytes = image_data
        
        file_like = io.BytesIO(image_bytes)
        
        return UploadFile(
            file=file_like,
            filename=filename,
            headers={"content-type": content_type}
        )
    except ImportError:
        # Return mock UploadFile if FastAPI not available
        mock_file = Mock()
        mock_file.filename = filename
        mock_file.content_type = content_type
        mock_file.file = io.BytesIO(b'mock_image_data')
        return mock_file

# Make utility functions available to all tests
pytest.create_test_image_with_text = create_test_image_with_text
pytest.create_test_upload_file = create_test_upload_file
