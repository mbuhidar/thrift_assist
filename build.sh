#!/bin/bash
# Build script for Render deployment

set -e

echo "🔧 Starting build process for Render..."

# Force reinstall of build tools to fix setuptools.build_meta issue
echo "⬆️ Installing compatible build tools..."
python -m pip install --force-reinstall pip==23.2.1
python -m pip install --force-reinstall setuptools==67.8.0 wheel==0.40.0

# Clear pip cache
pip cache purge

# Install dependencies
echo "📥 Installing dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

echo "✅ Build complete!"
