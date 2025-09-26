#!/usr/bin/env python3
"""
Advanced warning suppression for Google Cloud ALTS warnings.
"""

import os
import sys
import warnings
from contextlib import redirect_stderr
from io import StringIO

# Set comprehensive environment variables
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GRPC_TRACE'] = ''
os.environ['GOOGLE_CLOUD_DISABLE_GRPC_FOR_GAE'] = 'true'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'

# Additional environment variables to try
os.environ['GLOG_minloglevel'] = '3'  # Only FATAL messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress all warnings
warnings.filterwarnings("ignore")

def suppress_stderr_warnings():
    """Context manager to suppress stderr warnings during client creation."""
    class FilteredStringIO(StringIO):
        def write(self, s):
            if 'ALTS creds ignored' not in s and 'absl::InitializeLog' not in s:
                return super().write(s)
            return len(s)  # Return length as if we wrote it
    
    return redirect_stderr(FilteredStringIO())

print("Testing advanced ALTS warning suppression...")

try:
    # Suppress stderr during the import and client creation
    with suppress_stderr_warnings():
        from google.cloud import vision
        print("✅ Google Cloud Vision library imported")
        
        # Create client with stderr suppressed
        client = vision.ImageAnnotatorClient()
        print("✅ Vision API client created successfully")
    
    print("🎉 ALTS warnings should be suppressed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    if "DefaultCredentialsError" in str(e):
        print("This is the expected authentication error.")
    
print("\nTest completed.")