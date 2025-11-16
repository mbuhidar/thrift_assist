# DeepSeek-OCR Setup Guide for Google Cloud Vertex AI

This guide walks you through setting up DeepSeek-OCR via Google Cloud Vertex AI for the ThriftAssist application.

## Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI installed and configured
- Project with appropriate permissions (Owner or Editor role recommended)

## Step 1: Set Up Google Cloud Project

### 1.1 Create or Select a Project

```bash
# List existing projects
gcloud projects list

# Create a new project (optional)
gcloud projects create YOUR-PROJECT-ID --name="ThriftAssist OCR"

# Set your active project
gcloud config set project YOUR-PROJECT-ID
```

### 1.2 Enable Required APIs

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable other required services
gcloud services enable storage.googleapis.com
gcloud services enable compute.googleapis.com
```

## Step 2: Set Up Authentication

### 2.1 Create Service Account

```bash
# Create a service account
gcloud iam service-accounts create thriftassist-ocr \
    --display-name="ThriftAssist OCR Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
    --member="serviceAccount:thriftassist-ocr@YOUR-PROJECT-ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
    --member="serviceAccount:thriftassist-ocr@YOUR-PROJECT-ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
```

### 2.2 Download Service Account Key

```bash
# Create and download the key
gcloud iam service-accounts keys create credentials/deepseek-credentials.json \
    --iam-account=thriftassist-ocr@YOUR-PROJECT-ID.iam.gserviceaccount.com

# Verify the file was created
ls -lh credentials/deepseek-credentials.json
```

## Step 3: Enable DeepSeek Model in Vertex AI Model Garden

### 3.1 Via Google Cloud Console (Recommended)

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Vertex AI** → **Model Garden**
3. Search for **"deepseek-vl2"** or **"deepseek"**
4. Click on the **deepseek-ai/deepseek-vl2** model
5. Click **"Enable"** or **"Deploy"**
6. Follow the deployment wizard:
   - Select your project
   - Choose region: **us-central1** (recommended)
   - Configure resources as needed
   - Click **"Deploy"**

### 3.2 Via gcloud CLI (Alternative)

```bash
# Note: Model Garden deployment via CLI may vary based on model availability
# Check current model status
gcloud ai models list --region=us-central1 --filter="displayName:deepseek"

# If the model needs to be imported from Model Garden, you may need to:
# 1. Visit the Model Garden UI first
# 2. Accept any terms and conditions
# 3. Then use the CLI for subsequent operations
```

### 3.3 Verify Model Access

```bash
# List available models in Vertex AI
gcloud ai models list --region=us-central1

# Check endpoints (if model is deployed as an endpoint)
gcloud ai endpoints list --region=us-central1
```

## Step 4: Configure Environment Variables

### 4.1 Create `.env` File

Create a `.env` file in your project root:

```bash
# Google Cloud Project Configuration
GOOGLE_CLOUD_PROJECT=YOUR-PROJECT-ID
GOOGLE_CLOUD_LOCATION=us-central1

# Google Cloud Credentials (for both Vision API and Vertex AI)
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/credentials/deepseek-credentials.json

# OCR Provider Selection (optional, can be set via UI)
# OCR_PROVIDER=deepseek  # or 'google' for Google Vision API
```

### 4.2 Set Environment Variables (Linux/macOS)

```bash
# Add to your ~/.bashrc or ~/.zshrc
export GOOGLE_CLOUD_PROJECT="YOUR-PROJECT-ID"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/credentials/deepseek-credentials.json"

# Reload shell configuration
source ~/.bashrc  # or source ~/.zshrc
```

### 4.3 Set Environment Variables (Windows)

```powershell
# PowerShell
$env:GOOGLE_CLOUD_PROJECT="YOUR-PROJECT-ID"
$env:GOOGLE_CLOUD_LOCATION="us-central1"
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\credentials\deepseek-credentials.json"

# Or set system-wide in System Properties → Environment Variables
```

## Step 5: Verify Installation

### 5.1 Test Python Imports

```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(
    project="YOUR-PROJECT-ID",
    location="us-central1"
)

print("✅ Vertex AI initialized successfully!")
```

### 5.2 Test DeepSeek Provider

Run the test script:

```bash
# From project root
python test_provider_selection.py
```

### 5.3 Test via Web Interface

1. Start the application:
   ```bash
   make run
   # or
   python run_api.py
   ```

2. Open browser to `http://localhost:8000`

3. Select **"DeepSeek-OCR via Google Cloud"** from the OCR provider dropdown

4. Upload an image and test phrase detection

## Step 6: Troubleshooting

### Common Issues

#### "Project not found" or "Permission denied"

```bash
# Verify your credentials are correct
gcloud auth application-default login

# Check active project
gcloud config get-value project

# Verify service account has correct permissions
gcloud projects get-iam-policy YOUR-PROJECT-ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:thriftassist-ocr@*"
```

#### "Model not found" or "Endpoint not available"

- Verify the model is deployed in Model Garden
- Check that you're using the correct region (`us-central1`)
- Some models may require approval or have regional restrictions
- Visit the Vertex AI console to check model status

#### "Quota exceeded"

- Check your project quotas in Cloud Console → IAM & Admin → Quotas
- Request quota increase if needed
- Consider using a different region with available quota

#### Credential issues

```bash
# Verify credentials file exists and is valid
cat $GOOGLE_APPLICATION_CREDENTIALS | python -m json.tool

# Test authentication
gcloud auth application-default print-access-token

# Verify the service account
gcloud iam service-accounts list
```

### Enable Detailed Logging

Add to your Python code or `.env`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:
```bash
export GOOGLE_CLOUD_LOGGING_LEVEL=DEBUG
```

## Cost Considerations

### Vertex AI Pricing

- **Model Garden models**: Charges may vary based on:
  - Model deployment type (dedicated vs. shared resources)
  - Number of predictions/requests
  - Compute resources allocated
  - Data transfer costs

- **Typical costs**:
  - Online prediction: ~$0.02 - $0.10 per 1000 predictions
  - Always check [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator)

### Cost Optimization Tips

1. **Use shared endpoints** instead of dedicated deployments when possible
2. **Set up budget alerts** in Google Cloud Console
3. **Monitor usage** via Cloud Console → Billing
4. **Consider batch predictions** for large volumes
5. **Use Google Vision API** for simpler use cases (may be cheaper)

## Additional Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Model Garden Overview](https://cloud.google.com/vertex-ai/docs/model-garden/explore-models)
- [Vertex AI Python SDK](https://googleapis.dev/python/aiplatform/latest/)
- [DeepSeek Model Documentation](https://github.com/deepseek-ai/DeepSeek-VL2)
- [Google Cloud IAM Guide](https://cloud.google.com/iam/docs)

## Quick Reference

### Essential Commands

```bash
# Check project configuration
gcloud config list

# Test API access
gcloud ai models list --region=us-central1

# View service account keys
gcloud iam service-accounts keys list \
    --iam-account=thriftassist-ocr@YOUR-PROJECT-ID.iam.gserviceaccount.com

# Monitor API usage
gcloud logging read "resource.type=aiplatform.googleapis.com" --limit=50

# Check quotas
gcloud compute project-info describe --project=YOUR-PROJECT-ID
```

### Environment Variables Summary

| Variable | Purpose | Example |
|----------|---------|---------|
| `GOOGLE_CLOUD_PROJECT` | Your GCP project ID | `my-project-123` |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI region | `us-central1` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to credentials JSON | `/path/to/creds.json` |

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review application logs in `logs/` directory
3. Check Google Cloud Console for error messages
4. Review Vertex AI documentation for model-specific requirements

---

**Last Updated**: November 16, 2025
