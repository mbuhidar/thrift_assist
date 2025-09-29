#!/bin/bash
# ThriftAssist Text Detection Web Service Startup Script

echo "🚀 Starting ThriftAssist Text Detection Web Service"
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Create a virtual environment...exiting..."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing web service dependencies..."
pip install -r requirements_web.txt

# Check for Google Cloud credentials
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "⚠️  Warning: GOOGLE_APPLICATION_CREDENTIALS not set"
    echo "   Make sure your Google Cloud credentials are configured"
fi

# Display access information
echo ""
echo "🌐 ThriftAssist Application URLs:"
echo "   API Service: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   Web App: file://$(pwd)/web_app.html"
echo ""
echo "📖 To use the web app:"
echo "   1. Open web_app.html in your browser"
echo "   2. Upload an image and enter search phrases"
echo "   3. View results with annotated images"
echo ""

# Start the web service
echo "🌐 Starting FastAPI server..."
python3 ocr_web_service.py

