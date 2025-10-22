# Testing commands

.PHONY: test test-unit test-integration test-api test-coverage test-fast test-slow

# Run all tests
test:
	pytest

# Run only unit tests (fast)
test-unit:
	pytest -m unit

# Run integration tests
test-integration:
	pytest -m integration

# Run API tests
test-api:
	pytest -m api

# Run tests with coverage report
test-coverage:
	pytest --cov=backend --cov=vision --cov=config --cov=utils --cov=main --cov-report=html --cov-report=term --cov-report=term-missing

# Run fast tests only
test-fast:
	pytest -m "not slow"

# Run slow tests only
test-slow:
	pytest -m slow

# Run specific service tests
test-ocr:
	pytest -m ocr

test-cache:
	pytest -m cache

test-matcher:
	pytest -m matcher

test-image:
	pytest -m image

test-main:
	pytest tests/test_main.py

test-backend:
	pytest tests/test_backend/

test-services:
	pytest tests/test_services/

test-utils:
	pytest tests/test_utils/

# Run smoke tests (basic functionality)
test-smoke:
	pytest -m smoke

# Test specific files
test-image-service:
	pytest tests/test_services/test_image_service.py -v

test-cache-service:
	pytest tests/test_services/test_cache_service.py -v

test-ocr-service:
	pytest tests/test_services/test_ocr_service.py -v

test-ocr-routes:
	pytest tests/test_api/test_ocr_routes.py -v

# Clean test artifacts
test-clean:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf tests/test_data/temp_*
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Setup test environment
test-setup:
	@echo "Setting up test environment..."
	@if [ ! -f requirements-test.txt ]; then \
		echo "Creating requirements-test.txt..."; \
		echo "pytest>=7.0.0" > requirements-test.txt; \
		echo "pytest-cov>=4.0.0" >> requirements-test.txt; \
		echo "pytest-mock>=3.10.0" >> requirements-test.txt; \
		echo "numpy>=1.24.0" >> requirements-test.txt; \
	fi
	pip install -r requirements-test.txt
	mkdir -p tests/test_data
	mkdir -p logs/test
	@echo "✅ Basic test environment setup complete"

# Setup test environment with full dependencies
test-setup-full:
	@echo "Setting up full test environment..."
	@if [ ! -f requirements-test.txt ]; then \
		echo "Creating requirements-test.txt..."; \
		echo "pytest>=7.0.0" > requirements-test.txt; \
		echo "pytest-cov>=4.0.0" >> requirements-test.txt; \
		echo "pytest-mock>=3.10.0" >> requirements-test.txt; \
		echo "numpy>=1.24.0" >> requirements-test.txt; \
	fi
	pip install -r requirements-test.txt
	pip install opencv-python Pillow fastapi httpx
	mkdir -p tests/test_data
	mkdir -p logs/test
	@echo "✅ Full test environment setup complete"

# Setup minimal test environment (just pytest)
test-setup-minimal:
	@echo "Setting up minimal test environment..."
	pip install pytest pytest-cov
	mkdir -p tests/test_data
	mkdir -p logs/test
	@echo "✅ Minimal test environment setup complete"

# Run tests with dependency checks
test-check:
	@echo "Checking test dependencies..."
	@python -c "import pytest; print('✅ pytest available')" || echo "❌ pytest not available - run 'make test-setup'"
	@python -c "import cv2; print('✅ opencv available')" || echo "⚠️  opencv not available (some tests will be skipped)"
	@python -c "from PIL import Image; print('✅ PIL available')" || echo "⚠️  PIL not available (some tests will be skipped)"
	@python -c "import fastapi; print('✅ fastapi available')" || echo "⚠️  fastapi not available (API tests will be skipped)"
	@python -c "import numpy; print('✅ numpy available')" || echo "⚠️  numpy not available (some tests will be skipped)"

# Run tests by component
test-all-services:
	pytest tests/test_services/ -v

test-all-api:
	pytest tests/test_api/ -v

test-all-backend:
	pytest tests/test_backend/ -v

test-all-utils:
	pytest tests/test_utils/ -v

# Run tests in parallel (if pytest-xdist is installed)
test-parallel:
	pytest -n auto

# Comprehensive test suite with coverage
test-full:
	pytest --cov=backend --cov=vision --cov=config --cov=utils --cov=main --cov-report=html --cov-report=term-missing --durations=10 --cov-fail-under=50

# Quick development test
test-dev:
	pytest -m "unit and not slow" --tb=short

# Install dependencies based on what's available
install-test-deps:
	@echo "Installing available test dependencies..."
	@pip install pytest pytest-cov || echo "Failed to install pytest"
	@pip install numpy || echo "Failed to install numpy"
	@pip install opencv-python || echo "Failed to install opencv-python"
	@pip install Pillow || echo "Failed to install Pillow"
	@pip install fastapi httpx || echo "Failed to install fastapi/httpx"
	@echo "✅ Installed available dependencies"

# Coverage report only (no tests)
coverage-report:
	coverage report -m
	coverage html
	@echo "📊 Coverage report generated in htmlcov/index.html"

# Find untested files
coverage-missing:
	pytest --cov=backend --cov=vision --cov=config --cov=utils --cov=main --cov-report=term-missing | grep "TOTAL"
