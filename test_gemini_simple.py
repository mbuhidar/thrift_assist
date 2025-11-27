#!/usr/bin/env python3
"""Simple test to check Gemini API availability."""

import os
import vertexai
from vertexai.generative_models import GenerativeModel

# Set credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json'

project_id = "direct-bonsai-473201-t2"

print(f"Testing Gemini API access for project: {project_id}")
print("=" * 60)

# Try different regions
regions = ["us-central1", "us-east4", "europe-west1", "asia-southeast1"]
models = ["gemini-1.5-flash", "gemini-1.5-flash-001", "gemini-pro-vision", "gemini-1.0-pro-vision"]

for region in regions:
    print(f"\nTesting region: {region}")
    for model_name in models:
        try:
            vertexai.init(project=project_id, location=region)
            model = GenerativeModel(model_name)
            
            # Try a simple test
            response = model.generate_content("Say hello")
            print(f"  ✅ {model_name}: Available")
            break  # If one works, we're good for this region
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                print(f"  ❌ {model_name}: Not found")
            elif "403" in error_msg:
                print(f"  ⚠️  {model_name}: Permission denied")
            else:
                print(f"  ❌ {model_name}: {error_msg[:80]}")
