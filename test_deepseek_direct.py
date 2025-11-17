#!/usr/bin/env python3
"""
Direct test of DeepSeek via Google Cloud Vertex AI.
This script tests the DeepSeek OCR API service configuration.
"""

import os
import sys

# Test 1: Check if required packages are installed
print("=" * 60)
print("TEST 1: Checking required packages")
print("=" * 60)

try:
    from google.cloud import aiplatform
    print("✅ google-cloud-aiplatform is installed")
except ImportError as e:
    print(f"❌ google-cloud-aiplatform not installed: {e}")
    sys.exit(1)

try:
    from google.auth import default
    print("✅ google-auth is installed")
except ImportError as e:
    print(f"❌ google-auth not installed: {e}")
    sys.exit(1)

# Test 2: Check configuration
print("\n" + "=" * 60)
print("TEST 2: Checking configuration")
print("=" * 60)

# Load config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.vision_config import VisionConfig

config = VisionConfig()
project_id = config.google_cloud_project
location = config.google_cloud_location
endpoint = config.google_cloud_endpoint
credentials_path = config.google_credentials_path

print(f"Project ID: {project_id}")
print(f"Location: {location}")
print(f"Endpoint: {endpoint}")
print(f"Credentials path: {credentials_path}")

if not project_id:
    print("❌ No project ID configured")
    sys.exit(1)

# Check credentials file
if os.path.exists(credentials_path):
    print(f"✅ Credentials file exists: {credentials_path}")
    # Set environment variable for Google Cloud
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
else:
    print(f"❌ Credentials file not found: {credentials_path}")
    sys.exit(1)

# Test 3: Initialize Vertex AI
print("\n" + "=" * 60)
print("TEST 3: Initializing Vertex AI")
print("=" * 60)

try:
    aiplatform.init(
        project=project_id,
        location=location,
    )
    print(f"✅ Vertex AI initialized for project '{project_id}' in '{location}'")
except Exception as e:
    print(f"❌ Failed to initialize Vertex AI: {e}")
    sys.exit(1)

# Test 4: Check authentication
print("\n" + "=" * 60)
print("TEST 4: Checking authentication")
print("=" * 60)

try:
    credentials, project = default()
    print(f"✅ Authenticated with project: {project}")
    print(f"   Credentials type: {type(credentials).__name__}")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    sys.exit(1)

# Test 5: Try to access Model Registry
print("\n" + "=" * 60)
print("TEST 5: Checking Model Registry access")
print("=" * 60)

try:
    # List available models to verify API access
    from google.cloud.aiplatform_v1 import ModelServiceClient
    
    client = ModelServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )
    parent = f"projects/{project_id}/locations/{location}"
    
    print(f"Attempting to list models in: {parent}")
    
    # Try to list models (this will verify API access)
    request = {"parent": parent, "page_size": 1}
    page_result = client.list_models(request=request)
    
    print("✅ Successfully connected to Model Service API")
    
except Exception as e:
    print(f"⚠️  Model Service API access issue: {e}")
    print("   This might be normal if Model Garden isn't enabled")

# Test 6: Test DeepSeek provider directly
print("\n" + "=" * 60)
print("TEST 6: Testing DeepSeek provider initialization")
print("=" * 60)

try:
    from vision.providers.deepseek_provider import DeepSeekProvider
    
    provider = DeepSeekProvider(
        project_id=project_id,
        location=location,
        endpoint=endpoint
    )
    
    print(f"Provider name: {provider.name}")
    print(f"Provider available: {provider.is_available()}")
    
    if provider.is_available():
        print("✅ DeepSeek provider is configured and ready")
    else:
        print("❌ DeepSeek provider is not available")
        print(f"   project_id: {provider.project_id}")
        print(f"   location: {provider.location}")
        print(f"   endpoint: {provider.endpoint}")
        
except Exception as e:
    print(f"❌ Failed to initialize DeepSeek provider: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Check if DeepSeek R1 model is accessible
print("\n" + "=" * 60)
print("TEST 7: Checking DeepSeek R1 MaaS model access")
print("=" * 60)

try:
    from google.cloud.aiplatform_v1 import PredictionServiceClient
    
    model_name = getattr(config, 'deepseek_model', 'deepseek-r1-0528-maas')
    print(f"Attempting to access model: {model_name}")
    print(f"Location: {location}")
    
    # Create client
    client_options = {"api_endpoint": endpoint}
    client = PredictionServiceClient(client_options=client_options)
    
    # Prepare the endpoint name for MaaS model
    endpoint_path = (
        f"projects/{project_id}/locations/{location}/"
        f"publishers/google/models/{model_name}"
    )
    
    print(f"Model endpoint: {endpoint_path}")
    print("✅ Successfully configured DeepSeek R1 MaaS model endpoint")
    print("\n📝 Note: Actual model invocation will be tested when making OCR requests")
    
except Exception as e:
    print(f"❌ Failed to setup DeepSeek R1 model: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\n✅ Completed tests. Check output above for any issues.")
print("\nIf DeepSeek model access failed, you may need to:")
print("1. Enable Vertex AI API in your Google Cloud project")
print("2. Enable Model Garden and accept DeepSeek model terms")
print("3. Ensure your service account has the right permissions")
print("   Required roles: Vertex AI User, Service Account User")
