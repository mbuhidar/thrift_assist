#!/bin/bash
# Build script for Render deployment

set -e

echo "🔧 Starting build process for Render..."

# Upgrade core build tools first
echo "⬆️ Upgrading build tools..."
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Build complete!"
