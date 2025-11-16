# Quick Start: Using DeepSeek-OCR with ThriftAssist

This guide shows you how to switch from Google Cloud Vision to DeepSeek-OCR in ThriftAssist using Google Cloud's Vertex AI Model Garden.

## Why Use DeepSeek-OCR?

- **AI-Powered**: Uses DeepSeek's advanced vision model for text understanding
- **Google Cloud Integration**: Runs on Google Cloud Vertex AI infrastructure
- **Good for Complex Text**: Better at understanding context and unusual layouts
- **Alternative Option**: Good to have a backup when standard Vision API struggles

## Setup Steps

### 1. Enable Google Cloud Vertex AI

1. Go to https://console.cloud.google.com/
2. Select or create a Google Cloud project
3. Enable the Vertex AI API:
   - Navigate to "Vertex AI" in the console
   - Click "Enable API" if not already enabled
4. Enable Model Garden access:
   - Go to Vertex AI > Model Garden
   - Search for "deepseek-ai/deepseek-vl2"
   - Click "Enable" if required

### 2. Set Up Credentials

You'll use the same Google Cloud credentials as for Vision API:

**Option A: Using service account (recommended):**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

**Option B: Using gcloud CLI:**
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### 3. Set Environment Variables

**For a single session:**
```bash
export OCR_PROVIDER=deepseek
export GOOGLE_CLOUD_PROJECT="your-project-id"
# Optional: specify region (default is us-central1)
export GOOGLE_CLOUD_LOCATION="us-central1"
```

**For permanent setup (add to ~/.bashrc or ~/.zshrc):**
```bash
echo 'export OCR_PROVIDER=deepseek' >> ~/.bashrc
echo 'export GOOGLE_CLOUD_PROJECT="your-project-id"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Run ThriftAssist

```bash
# Test provider is working
python test_providers.py deepseek

# Use with the web app
python run_api.py

# Use with command line
python main.py --image photo.jpg --phrases "brand name,model number"
```

## Switching Between Providers

You can easily switch between providers:

**Use Google Cloud Vision:**
```bash
export OCR_PROVIDER=google
python run_api.py
```

**Use DeepSeek-OCR (via Google Cloud):**
```bash
export OCR_PROVIDER=deepseek
export GOOGLE_CLOUD_PROJECT="your-project-id"
python run_api.py
```

**Let the app decide (defaults to Google):**
```bash
unset OCR_PROVIDER
python run_api.py
```

## Testing Your Setup

### Test DeepSeek Configuration

```bash
python test_providers.py deepseek
```

Expected output:
```
============================================================
Testing OCR Provider: deepseek
============================================================

✓ Created VisionConfig
  - Provider: deepseek
✅ Using OCR provider: DeepSeek-OCR

✓ Created VisionPhraseDetector
  - Provider name: DeepSeek-OCR
  - Provider available: True

✅ Provider initialized successfully!
```

### Test with an Image

```bash
python test_providers.py deepseek image/your_test_image.jpg
```

## Troubleshooting

### "Provider not available" Error

**Problem**: Google Cloud project not set or Vertex AI not enabled

**Solution**:
```bash
# Check if project is set
echo $GOOGLE_CLOUD_PROJECT

# Set the project
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Verify credentials
gcloud auth application-default print-access-token

# Verify again
python test_providers.py deepseek
```

### "google-cloud-aiplatform library not available" Error

**Problem**: Missing Python dependency

**Solution**:
```bash
pip install google-cloud-aiplatform
```

### "Falling back to Google Cloud Vision" Warning

**Problem**: DeepSeek provider couldn't be initialized

**Solution**: This is normal behavior. The app will automatically use Google Cloud Vision instead. Check:
1. GOOGLE_CLOUD_PROJECT is set correctly
2. Vertex AI API is enabled in your project
3. DeepSeek model is available in Model Garden
4. Credentials have proper permissions

### "No results returned" When Testing

**Problem**: Could be API rate limiting or invalid image

**Solution**:
1. Check your DeepSeek account for API usage/limits
2. Try a different image file
3. Check image file is valid: `file your_image.jpg`

## Performance Considerations

### Speed
- **Google Cloud Vision**: Faster (optimized binary protocol)
- **DeepSeek-OCR**: Moderate (runs on Vertex AI infrastructure)

### Accuracy
- **Google Cloud Vision**: Excellent for printed text, structured documents
- **DeepSeek-OCR**: Better for complex layouts, understanding context, AI-powered

### Cost
- **Google Cloud Vision**: Free tier then pay-per-use
- **DeepSeek-OCR**: Vertex AI pricing (model inference costs)

## Best Practices

1. **Use Google Vision by default** - It's faster and very accurate
2. **Try DeepSeek for complex cases** - When Google Vision struggles with unusual text
3. **Same credentials work for both** - Uses your existing Google Cloud setup
4. **Test before production** - Run test_providers.py with your images

## Environment Variable Reference

```bash
# Required for both providers
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# Required for DeepSeek
export OCR_PROVIDER=deepseek
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Optional for DeepSeek
export GOOGLE_CLOUD_LOCATION="us-central1"  # Default region

# Required for Google Cloud Vision
export OCR_PROVIDER=google
```

## Web App Usage

When using the web app (public/web_app.html), the OCR provider is selected on the backend:

1. Start the API server:
   ```bash
   export OCR_PROVIDER=deepseek
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   python run_api.py
   ```

2. Open web browser to http://localhost:8000

3. Upload images and search for phrases as normal

The OCR provider selection is transparent to the web interface. You'll see "DeepSeek-OCR (Google Cloud)" in the results if using DeepSeek.

## Next Steps

- Read `OCR_PROVIDERS.md` for detailed provider documentation
- Read `IMPLEMENTATION_SUMMARY.md` for technical details
- Try both providers with your images to compare results
- Both providers use Google Cloud infrastructure

## Getting Help

If you encounter issues:
1. Run `python test_providers.py deepseek` to diagnose
2. Check the console output for specific error messages
3. Verify environment variables: `env | grep -E 'OCR|GOOGLE'`
4. Check Vertex AI is enabled: `gcloud services list --enabled | grep aiplatform`
5. Review logs for detailed error information
