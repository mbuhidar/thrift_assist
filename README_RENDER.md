# ThriftAssist Deployment on Render.com

## Quick Deploy

1. **Fork this repository** to your GitHub account

2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Connect your GitHub account
   - Create new Web Service from this repository

3. **Configuration:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Environment: `Python 3.12`

## Environment Variables

Set these in Render dashboard:

```
GOOGLE_APPLICATION_CREDENTIALS_JSON=<your-google-cloud-credentials-json>
PORT=10000
```

## Accessing Your App

- **API**: `https://your-app-name.onrender.com`
- **Docs**: `https://your-app-name.onrender.com/docs`
- **Web App**: Upload `web_app.html` to a static site service

## Google Cloud Setup

1. Create Google Cloud Project
2. Enable Vision API
3. Create Service Account
4. Download JSON key
5. Copy JSON content to `GOOGLE_APPLICATION_CREDENTIALS_JSON` environment variable

## Free Tier Limitations

- Service sleeps after 15 minutes of inactivity
- First request after sleep takes ~30 seconds
- 750 hours/month free compute time

## Troubleshooting

- Check logs in Render dashboard
- Verify environment variables are set
- Ensure Google Cloud credentials are valid
