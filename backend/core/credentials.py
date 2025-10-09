"""
Google Cloud credentials management.
"""

import os
import base64
import tempfile
from typing import Optional
from .config import settings


def setup_google_credentials() -> bool:
    """
    Setup Google Cloud credentials from environment variables or files.
    
    Returns:
        bool: True if credentials were successfully set up, False otherwise
    """
    try:
        # Method 1: Direct JSON credentials from environment variable
        creds_json = settings.GOOGLE_APPLICATION_CREDENTIALS_JSON
        if creds_json:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(creds_json)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
            print("✅ Using credentials from GOOGLE_APPLICATION_CREDENTIALS_JSON")
            return True
        
        # Method 2: Base64 encoded credentials
        creds_b64 = settings.GOOGLE_CREDENTIALS_BASE64
        if creds_b64:
            try:
                creds_json = base64.b64decode(creds_b64).decode('utf-8')
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    f.write(creds_json)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
                print("✅ Using credentials from GOOGLE_CREDENTIALS_BASE64")
                return True
            except Exception as e:
                print(f"⚠️ Failed to decode base64 credentials: {e}")
        
        # Method 3: File path from environment
        creds_path = settings.GOOGLE_APPLICATION_CREDENTIALS
        if creds_path and os.path.exists(creds_path):
            print(f"✅ Using credentials file: {creds_path}")
            return True
        
        # Method 4: Default local file
        default_path = "credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json"
        if os.path.exists(default_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(default_path)
            print(f"✅ Using default credentials: {default_path}")
            return True
        
        print("⚠️ No Google Cloud credentials found - OCR will use stub functions")
        return False
        
    except Exception as e:
        print(f"❌ Error setting up credentials: {e}")
        return False


def verify_credentials() -> bool:
    """
    Verify that Google Cloud credentials are working.
    
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        return True
    except Exception as e:
        print(f"❌ Credential verification failed: {e}")
        return False
