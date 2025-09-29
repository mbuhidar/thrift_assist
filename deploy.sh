#!/bin/bash

echo "🚀 Deploying ThriftAssist OCR Web Service"

# Pull the latest code from the repository
echo "🔄 Pulling latest code..."
git pull origin main

# Set up Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Set environment variables for FastAPI
export FASTAPI_APP=thriftassist_web_service.py
export FASTAPI_ENV=production

# Run database migrations (if any)
# echo "🔄 Running database migrations..."
# alembic upgrade head

# Collect static files (if any)
# echo "📂 Collecting static files..."
# python -m fastapi collectstatic

# Production deployment with Uvicorn
echo "🌐 Starting production server with FastAPI..."
uvicorn thriftassist_web_service:app --host 0.0.0.0 --port 8000 --workers 4

# Alternative: Production deployment with Gunicorn + Uvicorn workers
# gunicorn thriftassist_web_service:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120

# Deactivate virtual environment
deactivate

echo "✅ Deployment completed successfully!"