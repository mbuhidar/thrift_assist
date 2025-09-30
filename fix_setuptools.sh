#!/bin/bash

echo "🔧 Fixing setuptools and pip issues..."

# Remove broken virtual environment
if [ -d "venv" ]; then
    echo "🗑️ Removing existing virtual environment..."
    rm -rf venv
fi

# Create fresh virtual environment
echo "📦 Creating fresh virtual environment..."
python3 -m venv venv --clear

# Activate virtual environment
source venv/bin/activate

# Upgrade core packages first
echo "⬆️ Upgrading core packages..."
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📥 Installing dependencies..."
pip install fastapi uvicorn[standard] opencv-python numpy pydantic python-multipart google-cloud-vision rapidfuzz Pillow matplotlib

echo "✅ Setup complete! Now run ./start_web_service.sh"
