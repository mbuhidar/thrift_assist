#!/usr/bin/env python3
"""Test Gemini using direct REST API to Vertex AI."""

import os
import requests
import json
import subprocess

project_id = "direct-bonsai-473201-t2"
location = "us-central1"

# Get access token
result = subprocess.run(
    ["gcloud", "auth", "print-access-token"],
    capture_output=True,
    text=True
)
token = result.stdout.strip()

print(f"Testing Gemini via Vertex AI REST API")
print(f"Project: {project_id}, Location: {location}")
print("=" * 60)

# List available models
url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print(f"\nListing models at: {url}")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    models = response.json()
    print(f"\nAvailable models:")
    if 'models' in models:
        for model in models.get('models', []):
            name = model.get('name', model.get('displayName', 'unknown'))
            print(f"  - {name}")
    else:
        print("  (No models listed in response)")
        print(f"  Response: {json.dumps(models, indent=2)[:500]}")
else:
    print(f"\n❌ Error {response.status_code}: {response.text[:500]}")

# Try to call gemini-1.5-flash directly
print(f"\n\nTrying to generate content with gemini-1.5-flash...")
gen_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/gemini-1.5-flash:generateContent"

payload = {
    "contents": [{
        "role": "user",
        "parts": [{"text": "Say hello in one word"}]
    }]
}

response = requests.post(gen_url, headers=headers, json=payload)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print(f"✅ SUCCESS! Response: {response.json()}")
else:
    print(f"❌ Error: {response.text[:500]}")
