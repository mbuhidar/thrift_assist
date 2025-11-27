#!/bin/bash
# Quick diagnostic to check DeepSeek availability in Google Cloud

PROJECT_ID="direct-bonsai-473201-t2"

echo "=========================================="
echo "DeepSeek Google Cloud Availability Check"
echo "=========================================="
echo ""

# Switch to personal account
echo "Switching to personal account for admin commands..."
gcloud config set account mbuhidar@gmail.com
echo ""

echo "1. Checking deployed endpoints..."
echo "=================================="
gcloud ai endpoints list --region=global --project=$PROJECT_ID 2>&1 | head -20
echo ""

echo "2. Checking us-central1 region endpoints..."
echo "============================================"
gcloud ai endpoints list --region=us-central1 --project=$PROJECT_ID 2>&1 | head -20
echo ""

echo "3. Checking deployed models in us-central1..."
echo "=============================================="
gcloud ai models list --region=us-central1 --project=$PROJECT_ID 2>&1 | head -30
echo ""

echo "4. Searching Model Garden for DeepSeek..."
echo "=========================================="
gcloud ai model-garden models list --region=us-central1 2>&1 | grep -i deepseek || echo "DeepSeek not found in Model Garden"
echo ""

echo "5. Checking available publishers..."
echo "===================================="
gcloud ai model-garden models list --region=us-central1 2>&1 | grep -i "publisher" | head -10
echo ""

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "If you see endpoints above, note their IDs."
echo "If DeepSeek is in Model Garden, it should appear in search."
echo "If nothing found, DeepSeek is not available in Vertex AI yet."
echo ""
echo "Next steps:"
echo "1. Check Google Cloud Console Model Garden manually"
echo "2. Or use Google Cloud Vision (already working)"
echo "3. Or use DeepSeek's direct API (outside Google Cloud)"
