# How to Enable Gemini Vision on Google Cloud

## Prerequisites
- Google Cloud project: `direct-bonsai-473201-t2`
- Billing must be enabled on the project
- You need Owner or Editor permissions

## Step 1: Enable Required APIs

```bash
# Enable Vertex AI API (already done ✅)
gcloud services enable aiplatform.googleapis.com --project=direct-bonsai-473201-t2

# Enable Generative AI services
gcloud services enable generativelanguage.googleapis.com --project=direct-bonsai-473201-t2
```

## Step 2: Verify Billing is Enabled

Gemini models require active billing. Check in Google Cloud Console:

1. Go to: https://console.cloud.google.com/billing
2. Select your project `direct-bonsai-473201-t2`
3. Ensure a billing account is linked
4. Vertex AI Generative AI may have additional billing setup

Or check via command line:
```bash
gcloud beta billing projects describe direct-bonsai-473201-t2
```

## Step 3: Grant IAM Permissions

Ensure your service account has the right permissions:

```bash
# Grant Vertex AI User role
gcloud projects add-iam-policy-binding direct-bonsai-473201-t2 \
    --member="serviceAccount:first-project-service-acct@direct-bonsai-473201-t2.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Grant additional permissions for generative AI
gcloud projects add-iam-policy-binding direct-bonsai-473201-t2 \
    --member="serviceAccount:first-project-service-acct@direct-bonsai-473201-t2.iam.gserviceaccount.com" \
    --role="roles/ml.admin"
```

## Step 4: Test Access to Gemini Models

Create and run this test script:

```python
#!/usr/bin/env python3
import os
import vertexai
from vertexai.generative_models import GenerativeModel

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json'

project_id = "direct-bonsai-473201-t2"
location = "us-central1"

try:
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Say hello")
    print(f"✅ Gemini working! Response: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")
```

## Step 5: Common Issues and Solutions

### Issue: "404 Publisher Model not found"
**Cause:** Gemini not available in your region or project doesn't have access

**Solutions:**
1. Try different regions (us-central1, us-east4, europe-west1)
2. Wait 5-10 minutes after enabling APIs
3. Check if you need to accept terms of service in Cloud Console

**Manual check in Console:**
- Go to: https://console.cloud.google.com/vertex-ai/generative
- Select your project
- Click "Enable API" if prompted
- Accept Terms of Service

### Issue: "403 Permission denied"
**Cause:** Service account lacks permissions

**Solution:**
```bash
# List current IAM bindings
gcloud projects get-iam-policy direct-bonsai-473201-t2

# Add missing roles
gcloud projects add-iam-policy-binding direct-bonsai-473201-t2 \
    --member="serviceAccount:first-project-service-acct@direct-bonsai-473201-t2.iam.gserviceaccount.com" \
    --role="roles/aiplatform.admin"
```

### Issue: "Billing not enabled"
**Cause:** Project doesn't have billing enabled for Vertex AI

**Solution:**
1. Go to: https://console.cloud.google.com/billing
2. Link billing account to project
3. May need to enable Vertex AI billing separately

## Step 6: Verify Available Models

Run this to see what's actually available:

```bash
# List available Vertex AI models
gcloud ai models list --region=us-central1 --project=direct-bonsai-473201-t2

# Try REST API directly
curl -X GET \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/direct-bonsai-473201-t2/locations/us-central1/publishers/google/models"
```

## Alternative: Use Google AI Studio API Instead

If Vertex AI Gemini isn't available, you can use Google AI Studio (simpler):

1. Get API key: https://makersuite.google.com/app/apikey
2. Use the `google-generativeai` package (different from Vertex AI)
3. Simpler authentication but less enterprise features

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Extract text from this image", image)
```

## Summary: What's Required for Gemini Vision

✅ **Must Have:**
- Billing enabled on Google Cloud project
- Vertex AI API enabled
- Proper IAM permissions (aiplatform.user minimum)
- Project allowlisted for Gemini (may require manual enablement in Console)

⚠️ **Current Status for Your Project:**
- Vertex AI API: ✅ Enabled
- Billing: ❓ Unknown (check Console)
- Gemini Access: ❌ Getting 404 errors
- Terms of Service: ❓ May need to accept in Console

## Recommended Next Steps

1. **Check billing:** Visit Console and verify active billing
2. **Visit Vertex AI Generative AI page:** https://console.cloud.google.com/vertex-ai/generative
3. **Accept Terms of Service** if prompted
4. **Wait 5-10 minutes** after accepting for propagation
5. **Re-test** the Gemini API

## Decision Point

**If Gemini still doesn't work after these steps:**
- Google Cloud Vision is already working perfectly (6/6 matches)
- No compelling reason to switch from working solution
- Gemini adds complexity, cost, and potential unreliability

**Recommendation:** Stick with Google Cloud Vision unless you have a specific need that it can't meet.
