#!/bin/bash

echo "🚀 Starting ThriftAssist Web Service"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -r requirements_web.txt

# Set Google Cloud credentials if available
if [ -f "credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json"
    echo "✅ Google Cloud credentials configured"
else
    echo "⚠️  Warning: Google Cloud credentials not found"
fi

# Start the FastAPI service
echo "🌐 Starting ThriftAssist Web Service..."
echo "📖 API Documentation will be available at: http://localhost:8000/docs"
echo "🔍 Interactive API at: http://localhost:8000/redoc"

python thriftassist_web_service.py
