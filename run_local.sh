#!/bin/bash
# Local Development and Testing Script

echo "🧪 ThriftAssist Local Development Setup"
echo "======================================"

# Configuration for local testing
export HOST="127.0.0.1"
export PORT="8000"

# Use local credentials if available
if [ -f "credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json"
    echo "✅ Using local Google Cloud credentials"
else
    echo "⚠️  No local credentials found - OCR features will use stub functions"
fi

# Quick setup
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

# Kill any existing process on port
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true

echo ""
echo "🌐 Local Development URLs:"
echo "   API: http://localhost:$PORT"
echo "   Docs: http://localhost:$PORT/docs"
echo "   Web App: file://$(pwd)/web_app.html"
echo ""
echo "📋 Test Commands:"
echo "   curl http://localhost:$PORT/health"
echo "   open web_app.html (in browser)"
echo ""

# Start with reload for development
uvicorn main:app --host $HOST --port $PORT --reload --log-level debug
