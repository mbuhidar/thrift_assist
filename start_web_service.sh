#!/bin/bash
# ThriftAssist Web Service Startup Script

set -e  # Exit on any error

echo "  Starting ThriftAssist Web Service"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configure for access via http://localhost:8000 or http://<server-ip>:8000
HOST="0.0.0.0"
PORT="8000"

# Clean up function to stop background jobs on exit 
cleanup() {
    echo "🛑 Stopping service..."
    jobs -p | xargs -r kill
    exit 0
}

# Invoke cleanup on script exit via Ctrl+C or termination
trap cleanup SIGINT SIGTERM

# Set Google Cloud credentials if available
if [ -f "credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$SCRIPT_DIR/credentials/direct-bonsai-473201-t2-f19c1eb1cb53.json"
    echo "✅ Google Cloud credentials configured"
fi

# Find main.py
if [ -f "main.py" ]; then
    MAIN_FILE="main.py"
    MODULE="main:app"
elif [ -f "src/main.py" ]; then
    MAIN_FILE="src/main.py"
    MODULE="src.main:app"
else
    echo "❌ main.py not found"
    exit 1
fi

echo "📂 Using: $MAIN_FILE"

# Kill any process on port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "💀 Killing processes on port $PORT..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Start server
echo ""
echo "🌐 Starting server on http://$HOST:$PORT"
echo "📖 API Documentation: http://localhost:$PORT/docs"
echo "🌍 Web App (integrated): http://localhost:$PORT/"
echo "🌍 Web App (standalone): file://$SCRIPT_DIR/public/web_app.html"
echo "⏹️  Press Ctrl+C to stop"
echo ""

uvicorn $MODULE --host $HOST --port $PORT --reload
