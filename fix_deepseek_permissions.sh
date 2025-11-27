#!/bin/bash
# Fix DeepSeek OCR permissions for service account
# Run this with your personal account (mbuhidar@gmail.com), not the service account

PROJECT_ID="direct-bonsai-473201-t2"
SERVICE_ACCOUNT="first-project-service-acct@direct-bonsai-473201-t2.iam.gserviceaccount.com"

echo "=================================="
echo "Fixing DeepSeek OCR Permissions"
echo "=================================="
echo ""

# Switch to personal account
echo "Switching to personal account..."
gcloud config set account mbuhidar@gmail.com
echo ""

# Enable Vertex AI API
echo "Step 1: Enabling Vertex AI API..."
gcloud services enable aiplatform.googleapis.com --project=$PROJECT_ID
echo ""

# Grant Vertex AI User role (includes endpoints.predict permission)
echo "Step 2: Granting Vertex AI User role to service account..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/aiplatform.user"
echo ""

# Alternative: Grant more specific permission
echo "Step 3: Granting Vertex AI Endpoint User role..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/aiplatform.endpointUser"
echo ""

# Switch back to service account for testing
echo "Switching back to service account..."
gcloud config set account $SERVICE_ACCOUNT
echo ""

echo "✅ Permissions updated!"
echo ""
echo "Wait a few seconds for permissions to propagate, then test again."
echo ""
echo "Run: python test_deepseek_endpoint.py"
