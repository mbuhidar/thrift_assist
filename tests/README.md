# Testing Guide for ThriftAssist

## Test Organization

### Test Categories

Tests are organized by type using pytest markers:

- **`@pytest.mark.unit`** - Pure unit tests, no external dependencies
- **`@pytest.mark.integration`** - Integration tests requiring external services
- **`@pytest.mark.api`** - API endpoint tests
- **`@pytest.mark.smoke`** - Basic smoke tests for quick validation
- **`@pytest.mark.slow`** - Tests that take longer to execute

### Running Tests

```bash
# Run all tests
make test

# Run only unit tests (fast, no dependencies)
make test-unit

# Run with coverage report
make test-coverage

# Run specific test categories
make test-api        # API tests only
make test-matcher    # Phrase matching tests
make test-utils      # Utility function tests
```

## Test Structure

```
tests/
├── conftest.py              # Global fixtures and configuration
├── test_main.py             # Main application tests
├── test_api/                # API endpoint tests
│   ├── test_health_routes.py
│   ├── test_ocr_routes.py
│   └── test_cache_routes.py
├── test_backend/            # Backend module tests
│   ├── test_main.py
│   ├── test_credentials.py
│   └── test_utils.py
├── test_config/             # Configuration tests
│   ├── test_settings.py
│   └── test_vision_config.py
├── test_services/           # Service layer tests
│   ├── test_cache_service.py
│   ├── test_image_service.py
│   └── test_ocr_service.py
├── test_utils/              # Utility function tests
│   ├── test_geometry_utils.py
│   ├── test_image_utils.py
│   ├── test_image_utils_explicit.py
│   └── test_text_utils.py
└── test_vision/             # Vision/OCR module tests
    ├── test_detector.py
    ├── test_grouper.py
    └── test_matcher.py
```

## Dependency Requirements

### Tests That Require NO Dependencies (Pure Unit Tests)

These tests can run without installing production dependencies:

- `test_config/test_vision_config.py` - Configuration validation
- `test_utils/test_text_utils.py` - Text normalization and validation
- `test_utils/test_geometry_utils.py` - Geometry calculations
- `test_utils/test_image_utils_explicit.py` - Image utility logic (uses PIL only)

### Tests That Require Production Dependencies

These tests need specific packages installed:

#### Requires: `fastapi`, `pydantic`, `pydantic-settings`
- `test_main.py`
- `test_api/test_ocr_routes.py`
- `test_backend/test_main.py`
- `test_config/test_settings.py`
- `test_services/test_cache_service.py`
- `test_services/test_ocr_service.py`

#### Requires: `opencv-python` (cv2)
- `test_services/test_image_service.py`
- `test_vision/test_detector.py`
- `test_vision/test_grouper.py`
- `test_vision/test_matcher.py`

#### Requires: Google Cloud Vision API credentials
- Integration tests marked with `@pytest.mark.integration`
- Tests in `test_services/test_ocr_service.py` (integration tests only)

## Test Coverage Goals

### Current Coverage Status

Run `make test-coverage` to generate detailed coverage report in `htmlcov/index.html`

### Target Coverage by Module

- **Utils**: 95%+ (pure functions, easy to test)
- **Services**: 80%+ (business logic with some external deps)
- **API Routes**: 75%+ (integration with FastAPI)
- **Vision**: 70%+ (depends on external OCR service)

## Writing New Tests

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<what_is_being_tested>`

Examples:
```python
def test_normalize_text_for_search_lowercase():
    """Test text normalization converts to lowercase."""
    
def test_find_matches_exact_match():
    """Test finding exact phrase matches."""
```

### Explicit Test Patterns

**Good - Explicit test names:**
```python
def test_normalize_text_for_search_lowercase():
def test_normalize_text_for_search_whitespace():
def test_normalize_text_for_search_empty_and_edge_cases():
```

**Avoid - Vague test names:**
```python
def test_normalize():
def test_basic():
def test_edge_cases():
```

### Using Fixtures

Common fixtures available in `conftest.py`:

```python
def test_with_sample_image(sample_image):
    """Use pre-generated test image."""
    assert sample_image.shape == (300, 400, 3)

def test_with_temp_directory(temp_dir):
    """Use temporary directory for file operations."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test data")
```

### Parametrized Tests

Use `@pytest.mark.parametrize` for testing multiple inputs:

```python
@pytest.mark.parametrize("width,expected_quality", [
    (100, 88),
    (1000, 85),
    (2000, 82),
])
def test_quality_for_width(width, expected_quality):
    actual = calculate_optimal_jpeg_quality(width)
    assert actual == expected_quality
```

## Mocking External Dependencies

### Mocking Google Vision API

```python
@patch('vision.detector.vision.ImageAnnotatorClient')
def test_with_mock_vision_api(mock_client):
    mock_response = Mock()
    mock_response.text_annotations = []
    mock_client.return_value.text_detection.return_value = mock_response
    
    # Test code here
```

### Mocking File System

```python
@patch('builtins.open', mock_open(read_data='test data'))
def test_file_reading():
    # Test code that reads files
```

## Continuous Integration

### Pre-commit Checks

Before committing, run:
```bash
make test-fast     # Quick unit tests
make test-unit     # All unit tests
```

### CI Pipeline

The CI pipeline runs:
1. Unit tests (no external dependencies)
2. Integration tests (with mocked services)
3. Coverage report generation
4. Linting and type checking

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'fastapi'"**
- Solution: Install production dependencies: `pip install -r requirements.txt`
- Or run only unit tests: `make test-unit`

**"ModuleNotFoundError: No module named 'cv2'"**
- Solution: Install OpenCV: `pip install opencv-python`
- Or skip vision tests: `pytest -m "not vision"`

**"Fixture not found"**
- Check that fixture is defined in `conftest.py`
- Ensure pytest can find `conftest.py` (run from project root)

### Debug Mode

Run tests with verbose output and debugging:
```bash
pytest -vv --tb=long tests/test_utils/
pytest -vv -s tests/test_specific.py::test_function
```

## Best Practices

1. **Write explicit, focused tests** - One test per behavior
2. **Use descriptive test names** - Name should explain what's being tested
3. **Test edge cases** - Empty strings, None values, boundary conditions
4. **Use parametrize for multiple inputs** - Reduces code duplication
5. **Mock external dependencies** - Tests should be fast and reliable
6. **Keep tests independent** - Each test should run in isolation
7. **Assert specific values** - Avoid vague assertions like `assert result`
8. **Document test intent** - Use docstrings to explain complex tests

## Performance Guidelines

- Unit tests should complete in < 100ms each
- Integration tests should complete in < 5s each
- Total test suite should complete in < 30s for fast feedback

## Coverage Analysis

View detailed coverage by running:
```bash
make test-coverage
# Open htmlcov/index.html in browser
```

Focus coverage efforts on:
1. Core business logic (services, vision modules)
2. Utility functions (high reuse, easy to test)
3. API endpoints (user-facing functionality)

Lower priority for coverage:
- Configuration files (static data)
- Simple getters/setters
- Logging statements
