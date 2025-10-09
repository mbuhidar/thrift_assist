# Migration Guide: Monolithic to Modular Architecture

## Quick Start: Deprecate Old Files

Run the deprecation script to rename old files:

```bash
# Make the script executable
chmod +x deprecate_old_files.sh

# Run the deprecation
./deprecate_old_files.sh
```

This will rename:
- `main.py` → `main.py.deprecated`
- Any old test files → `*.deprecated`

## Files Status

### ❌ Deprecated Files (Renamed with .deprecated)

1. **main.py.deprecated** (formerly main.py)
   - **Replaced by:** `backend/api/main.py`
   - **Status:** Deprecated, keep for reference
   - **Timeline:** Remove completely in version 2.0.0

2. **test_*.py.deprecated** (old test files)
   - **Replaced by:** Proper test suite in `tests/`
   - **Status:** Can be deleted

### ✅ Core Functionality (Keep These)

**Still Active:**
- `thriftassist_googlevision.py` - Core OCR module (imported by backend)
- `requirements.txt` - Python dependencies
- `credentials/` - API credentials directory
- `.env` - Environment variables (if exists)

**New Architecture:**
- `backend/` - All new modular code
  - `backend/api/main.py` - New API entry point
  - `backend/api/routes/` - Route handlers
  - `backend/services/` - Business logic
  - `backend/core/` - Configuration
- `frontend/public/web_app.html` - Web interface
- `run_api.py` - Development server launcher
- `ARCHITECTURE.md` - Architecture documentation
- `MIGRATION.md` - This file

### 🔄 Files to Move (Future Refactoring)

These files work but should be refactored:

1. **thriftassist_googlevision.py**
   - Current: Root directory
   - Future: Move to `backend/core/ocr_engine.py`
   - Status: Working, low priority to move

## Migration Steps

### ✅ Phase 1: Deprecate Old Files (COMPLETE)

```bash
./deprecate_old_files.sh
```

### Phase 2: Test New Architecture

```bash
# Start the new API
python run_api.py

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs

# Test web interface
open http://localhost:8000
```

### Phase 3: Verify Functionality

- [ ] Upload image through web interface
- [ ] Run OCR detection
- [ ] Verify results display correctly
- [ ] Test cache functionality
- [ ] Test threshold updates

### Phase 4: Update Deployment

Update your deployment configuration:

**Before:**
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**After:**
```bash
python run_api.py
# OR
uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT
```

### Phase 5: Clean Up (After Full Testing)

Only after confirming everything works:

```bash
# Remove deprecated files
rm -f *.deprecated

# Remove old test files
rm -f test_*.py.deprecated
```

## Deployment Changes

### Render.com Configuration

Update your `render.yaml` or web service settings:

**Old Start Command:**
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**New Start Command:**
```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT
```

Or use the run script:
```bash
python run_api.py
```

### Docker Configuration (if using)

**Old Dockerfile:**
```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**New Dockerfile:**
```dockerfile
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Environment Variables (No Change)

All existing environment variables still work:
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_APPLICATION_CREDENTIALS_JSON`
- `GOOGLE_CREDENTIALS_BASE64`
- `PORT`
- `HOST`

## Rollback Plan

If you need to rollback to the old architecture:

```bash
# Restore old main.py
mv main.py.deprecated main.py

# Use old startup command
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'backend'`:

```bash
# Make sure you're in the project root
cd /home/mbuhidar/Code/mbuhidar/thrift_assist

# Verify directory structure
ls -la backend/

# Run from project root
python run_api.py
```

### Google Cloud Credentials

If OCR fails with authentication errors:

```bash
# Verify credentials file exists
ls -la credentials/

# Check environment variable
echo $GOOGLE_APPLICATION_CREDENTIALS

# Test credentials
python -c "from google.cloud import vision; client = vision.ImageAnnotatorClient()"
```

### Port Already in Use

```bash
# Find process using port 8000
lsof -ti:8000

# Kill the process
kill -9 $(lsof -ti:8000)

# Or use a different port
PORT=8001 python run_api.py
```

## Post-Migration Checklist

- [ ] Old files renamed with `.deprecated` extension
- [ ] New API starts without errors
- [ ] Web interface loads correctly
- [ ] OCR detection works
- [ ] Image upload works
- [ ] Cache functionality verified
- [ ] API documentation accessible at `/docs`
- [ ] Health check responds at `/health`
- [ ] Deployment updated (if applicable)
- [ ] Team notified of changes

## Support

For questions or issues:
1. Check `ARCHITECTURE.md` for system design
2. Review API docs at `http://localhost:8000/docs`
3. Check logs for error messages
4. Verify environment variables are set

---

**Migration Status:** Phase 1 Complete  
**Last Updated:** 2024-01-15  
**Next Phase:** Testing & Verification
