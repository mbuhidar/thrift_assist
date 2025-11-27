#!/usr/bin/env python3
"""Check what Vision AI and other Google Cloud services can do for OCR."""

import subprocess

project_id = "direct-bonsai-473201-t2"

print("Google Cloud Vision & AI Options for OCR")
print("=" * 70)

print("\n1. Cloud Vision API (currently using) ✅")
print("   - Text Detection (DOCUMENT_TEXT_DETECTION)")
print("   - Best for: Scene text, photos, book spines, signs")
print("   - Features: Fast, accurate, handles rotated/varied fonts")

print("\n2. Document AI")
print("   - Specialized for document processing")
print("   - Best for: PDFs, forms, invoices, structured documents")
print("   - NOT ideal for: Scene photos like bookshelves")

print("\n3. Vision AI (visionai.googleapis.com)")
print("   - Video intelligence and streaming")
print("   - Best for: Video analysis, not static images")

print("\n4. AutoML Vision")
print("   - Custom-trained models")
print("   - Best for: Specific use cases when you have training data")
print("   - Overkill for: General OCR tasks")

print("\n5. Vertex AI Models (if available)")
result = subprocess.run(
    f"gcloud ai models list --region=us-central1 --project={project_id} 2>&1 | head -20",
    shell=True,
    capture_output=True,
    text=True
)
if "Listed 0 items" in result.stdout or not result.stdout.strip():
    print("   ❌ No models deployed")
else:
    print(f"   Available models:\n{result.stdout}")

print("\n" + "=" * 70)
print("RECOMMENDATION: Stick with Cloud Vision API")
print("   ✅ Already working perfectly (6/6 matches)")
print("   ✅ Best for bookshelf/scene text")
print("   ✅ No additional setup needed")
print("   ✅ Fast and cost-effective")
