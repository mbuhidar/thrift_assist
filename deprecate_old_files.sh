#!/bin/bash

echo "🗂️  Deprecating old files..."
echo "================================"

# Function to safely rename a file
deprecate_file() {
    local file=$1
    if [ -f "$file" ]; then
        echo "📝 Renaming: $file → ${file}.deprecated"
        mv "$file" "${file}.deprecated"
    else
        echo "⏭️  Skipping (not found): $file"
    fi
}

# Deprecate main.py (replaced by backend/api/main.py)
deprecate_file "main.py"

# Deprecate any old test files
deprecate_file "test_ocr.py"
deprecate_file "test_example.py"
deprecate_file "example_usage.py"

# Deprecate thriftassist_googlevision.py (will be moved to backend/services/)
# Actually, we should keep this for now since it's still being imported
# deprecate_file "thriftassist_googlevision.py"

echo ""
echo "✅ Deprecation complete!"
echo ""
echo "Files renamed:"
echo "  - main.py → main.py.deprecated"
echo ""
echo "Files to keep (still in use):"
echo "  - thriftassist_googlevision.py (imported by backend services)"
echo "  - requirements.txt"
echo "  - run_api.py"
echo "  - ARCHITECTURE.md"
echo "  - MIGRATION.md"
echo ""
echo "To use the new architecture:"
echo "  python run_api.py"
