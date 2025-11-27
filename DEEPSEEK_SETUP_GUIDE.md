# DeepSeek OCR Setup Guide for Google Cloud

This guide will walk you through setting up DeepSeek OCR via Google Cloud Vertex AI.

## Current Status

Based on diagnostic tests:
- ✅ Service account has proper IAM permissions (`aiplatform.user`, `aiplatform.endpointUser`)
- ❌ Endpoint `/endpoints/openapi` does not exist in your project
- ❌ 403 error suggests the endpoint needs to be created/deployed first

## Step-by-Step Setup

### Step 1: Access Vertex AI Model Garden

1. Open Google Cloud Console: https://console.cloud.google.com
2. Select project: `direct-bonsai-473201-t2`
3. Navigate to **Vertex AI** → **Model Garden**
   - Direct link: https://console.cloud.google.com/vertex-ai/model-garden?project=direct-bonsai-473201-t2

### Step 2: Search for DeepSeek Models

In Model Garden, search for:
- "DeepSeek OCR"
- "DeepSeek VL"
- "DeepSeek Vision"

**What to look for:**
- Model name containing "deepseek"
- Vision/OCR capabilities
- Model ID that matches: `deepseek-ai/deepseek-ocr-maas`

### Step 3: Check Model Availability

DeepSeek models might be available through:

#### Option A: Model Garden (Recommended)
If you find DeepSeek in Model Garden:
1. Click on the model
2. Look for "Deploy" or "Enable" button
3. Choose deployment options:
   - Endpoint name: `deepseek-ocr-endpoint` (or custom name)
   - Region: `global` or `us-central1`
   - Machine type: Default (or smallest available)

#### Option B: Partner Integrations
DeepSeek might be available through:
- Hugging Face integration
- Third-party model providers
- Direct API access (not through Vertex AI)

#### Option C: Not Available Yet
If DeepSeek is not in Model Garden:
- It may be in private preview
- Might require signup/approval
- Could be region-restricted

### Step 4: Verify Endpoint After Deployment

After deploying, the endpoint should be accessible at:
```
https://aiplatform.googleapis.com/v1/projects/direct-bonsai-473201-t2/locations/global/endpoints/YOUR_ENDPOINT_ID/chat/completions
```

**Important:** The endpoint ID won't be "openapi" - it will be a generated ID like:
- `1234567890123456789`
- `deepseek-ocr-endpoint-20231125`

### Step 5: Update Configuration

Once you have the correct endpoint ID, update the code:

1. Find the endpoint ID from Cloud Console
2. Update `vision/providers/deepseek_provider.py`:
   ```python
   # Current (wrong):
   url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"
   
   # Should be:
   url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
   ```

## Alternative: Check if DeepSeek Uses Different API Pattern

DeepSeek might not follow the OpenAI chat completions pattern. Let's check alternatives:

### Check 1: List Existing Endpoints

Run this command to see all deployed endpoints:
```bash
gcloud ai endpoints list \
  --region=global \
  --project=direct-bonsai-473201-t2
```

### Check 2: Search Available Models

```bash
gcloud ai models list \
  --region=us-central1 \
  --project=direct-bonsai-473201-t2
```

### Check 3: Check Model Garden via gcloud

```bash
# List available publisher models
gcloud ai model-garden models list --region=us-central1

# Or search for deepseek
gcloud ai model-garden models list --region=us-central1 | grep -i deepseek
```

## What to Do Next

### If DeepSeek is Available in Model Garden:
1. ✅ Deploy it following the UI wizard
2. ✅ Note the endpoint ID
3. ✅ Update the code with correct endpoint
4. ✅ Test with `python test_deepseek_endpoint.py`

### If DeepSeek is NOT Available:
**Option 1: Use Direct DeepSeek API**
- Sign up at: https://platform.deepseek.com
- Get API key
- Use their direct endpoint (not through Google Cloud)
- Modify code to use DeepSeek's native API

**Option 2: Use Alternative OCR Models**
Available in Vertex AI Model Garden:
- Google Cloud Vision API (already working ✅)
- PaLM 2 Vision (if available)
- Gemini Vision (Google's latest)
- Claude 3 Vision (via Anthropic on Vertex AI)

**Option 3: Wait for DeepSeek**
- Check Model Garden periodically
- Contact Google Cloud support for access
- Request early access if in preview

## Testing Commands

After any configuration change, run:

```bash
# Test endpoint connectivity
python test_deepseek_endpoint.py

# If endpoint works, test with actual image
python test_deepseek_image.py /path/to/test/image.jpg

# Test in the main app
python run_api.py
# Then select DeepSeek in the UI
```

## Current Recommendation

Given that:
1. Google Cloud Vision API is working perfectly ✅
2. DeepSeek endpoint doesn't exist in your project
3. Setup requires additional deployment steps

**I recommend:**
- Keep using Google Cloud Vision (it's working great!)
- Mark DeepSeek as "experimental" or "requires setup"
- Optionally hide it from the UI until deployed

Would you like me to:
1. Help you search for DeepSeek in Model Garden?
2. Update the code to use DeepSeek's direct API instead?
3. Remove DeepSeek option until it's properly deployed?
4. Add Gemini Vision as an alternative (Google's latest AI vision model)?

## Next Steps - Choose One:

### A. Manual Setup via Cloud Console
1. Go to: https://console.cloud.google.com/vertex-ai/model-garden?project=direct-bonsai-473201-t2
2. Search for "deepseek" or vision models
3. Deploy if available
4. Report back what you find

### B. Use gcloud CLI to Search
Run these commands and share output:
```bash
# Switch to personal account (has permissions)
gcloud config set account mbuhidar@gmail.com

# List endpoints
gcloud ai endpoints list --region=global --project=direct-bonsai-473201-t2

# List models  
gcloud ai models list --region=us-central1 --project=direct-bonsai-473201-t2

# Search model garden
gcloud ai model-garden models list --region=us-central1 2>/dev/null | grep -i deepseek || echo "DeepSeek not found"
```

### C. Stick with Google Cloud Vision
It's working perfectly, well-supported, and reliable. DeepSeek can be added later when available.

---

**Let me know which path you'd like to take and I'll guide you through it!**
