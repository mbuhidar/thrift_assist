#!/usr/bin/env python3
"""
Test script to verify ALTS warning suppression works.
"""

# Set environment variables to suppress warnings BEFORE any imports
import os
import logging

# Suppress ALTS warnings and gRPC verbose logging
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['GOOGLE_CLOUD_DISABLE_GRPC_FOR_GAE'] = 'true'

# Set logging level to suppress warnings
logging.getLogger('google.auth').setLevel(logging.ERROR)
logging.getLogger('google.cloud').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)

print("Environment variables set to suppress warnings:")
print(f"GRPC_VERBOSITY = {os.environ.get('GRPC_VERBOSITY')}")
print(f"GRPC_TRACE = {os.environ.get('GRPC_TRACE')}")
gae_var = os.environ.get('GOOGLE_CLOUD_DISABLE_GRPC_FOR_GAE')
print(f"GOOGLE_CLOUD_DISABLE_GRPC_FOR_GAE = {gae_var}")

try:
    # Import Google Cloud libraries
    from google.cloud import vision
    
    print("✅ Google Cloud Vision library imported successfully")
    
    # Try to create a client (this is where ALTS warnings typically appear)
    print("Creating Vision API client...")
    client = vision.ImageAnnotatorClient()
    print("✅ Vision API client created successfully")
    print("   (warnings should be suppressed)")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("This is expected if authentication is not set up.")

print("\nIf you see this message without ALTS warnings above,")
print("the suppression worked!")