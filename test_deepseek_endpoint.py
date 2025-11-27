#!/usr/bin/env python3
"""
Test script to verify DeepSeek OCR endpoint accessibility.
This helps diagnose 403 Forbidden errors.
"""

import subprocess
import requests
import json

# Configuration
PROJECT_ID = "direct-bonsai-473201-t2"
LOCATION = "global"
SERVICE_ACCOUNT = (
    "first-project-service-acct@"
    "direct-bonsai-473201-t2.iam.gserviceaccount.com"
)
MODEL_NAME = "deepseek-ai/deepseek-ocr-maas"

print("=" * 70)
print("DeepSeek OCR Endpoint Diagnostic Tool")
print("=" * 70)
print()

# Step 1: Check gcloud authentication
print("Step 1: Checking gcloud authentication...")
try:
    result = subprocess.run(
        ['gcloud', 'auth', 'list'],
        capture_output=True,
        text=True,
        check=True
    )
    print("✅ Authenticated accounts:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"❌ Failed to check auth: {e}")
    exit(1)

# Step 2: Get access token
print("\nStep 2: Getting access token...")
try:
    result = subprocess.run(
        ['gcloud', 'auth', 'print-access-token',
         f'--account={SERVICE_ACCOUNT}'],
        capture_output=True,
        text=True,
        check=True
    )
    access_token = result.stdout.strip()
    print(f"✅ Access token obtained (length: {len(access_token)})")
    print(f"   First 20 chars: {access_token[:20]}...")
except subprocess.CalledProcessError as e:
    print(f"❌ Failed to get access token: {e}")
    print("   Trying without service account...")
    try:
        result = subprocess.run(
            ['gcloud', 'auth', 'print-access-token'],
            capture_output=True,
            text=True,
            check=True
        )
        access_token = result.stdout.strip()
        print(f"✅ Access token obtained (default account)")
    except Exception as e2:
        print(f"❌ Failed: {e2}")
        exit(1)

# Step 3: Check Vertex AI API status
print("\nStep 3: Checking if Vertex AI API is enabled...")
try:
    result = subprocess.run(
        ['gcloud', 'services', 'list', '--enabled',
         '--filter=aiplatform.googleapis.com',
         f'--project={PROJECT_ID}'],
        capture_output=True,
        text=True,
        check=True
    )
    if 'aiplatform.googleapis.com' in result.stdout:
        print("✅ Vertex AI API is enabled")
    else:
        print("❌ Vertex AI API is NOT enabled")
        print("   Enable it with:")
        print(f"   gcloud services enable aiplatform.googleapis.com "
              f"--project={PROJECT_ID}")
except subprocess.CalledProcessError as e:
    print(f"⚠️ Could not check API status: {e}")

# Step 4: Test different endpoint variations
print("\nStep 4: Testing endpoint variations...")

endpoints = [
    # OpenAPI endpoint (current attempt)
    (f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/"
     f"locations/{LOCATION}/endpoints/openapi/chat/completions",
     "OpenAPI endpoint"),
    
    # Model Garden endpoint
    (f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/"
     f"{PROJECT_ID}/locations/{LOCATION}/publishers/deepseek/models/"
     f"{MODEL_NAME}:predict",
     "Model Garden predict endpoint"),
    
    # Direct chat completions
    (f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/"
     f"{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi/"
     "chat/completions",
     "Regional OpenAPI endpoint"),
]

headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}

test_payload = {
    "model": MODEL_NAME,
    "messages": [{
        "role": "user",
        "content": "Hello"
    }],
    "max_tokens": 10
}

for url, description in endpoints:
    print(f"\n  Testing: {description}")
    print(f"  URL: {url}")
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=test_payload,
            timeout=10
        )
        
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            print("  ✅ SUCCESS! This endpoint works!")
            print(f"  Response: {response.text[:200]}")
            break
        elif response.status_code == 403:
            print("  ❌ 403 Forbidden")
            try:
                error_data = response.json()
                print(f"  Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"  Error text: {response.text[:200]}")
        elif response.status_code == 404:
            print("  ❌ 404 Not Found - endpoint doesn't exist")
        else:
            print(f"  ⚠️ Unexpected status: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("  ⏱️ Timeout - endpoint may not exist")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Step 5: Check available models
print("\n\nStep 5: Checking available Vertex AI models...")
try:
    result = subprocess.run(
        ['gcloud', 'ai', 'models', 'list',
         f'--region={LOCATION}',
         f'--project={PROJECT_ID}'],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode == 0:
        print("Available models:")
        print(result.stdout if result.stdout else "  (none)")
    else:
        print(f"Could not list models: {result.stderr}")
except subprocess.TimeoutExpired:
    print("⏱️ Timeout listing models")
except Exception as e:
    print(f"⚠️ Error listing models: {e}")

# Step 6: Recommendations
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print("""
1. If Vertex AI API is not enabled, enable it:
   gcloud services enable aiplatform.googleapis.com \\
     --project=direct-bonsai-473201-t2

2. DeepSeek OCR may require Model Garden access. Check:
   https://console.cloud.google.com/vertex-ai/model-garden

3. The model may need to be deployed to an endpoint first:
   - Go to Vertex AI > Model Garden
   - Search for "DeepSeek OCR"
   - Deploy the model to create an endpoint

4. Alternative: Use Google Cloud Vision API which is working correctly

5. Check IAM permissions for the service account:
   gcloud projects get-iam-policy direct-bonsai-473201-t2 \\
     --flatten="bindings[].members" \\
     --filter="bindings.members:first-project-service-acct@*"
""")
