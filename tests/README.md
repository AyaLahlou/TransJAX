# CLM-JAX Tests

This directory contains comprehensive test suites for all CLM-JAX modules.

## Directory Structure

The test structure mirrors the source code structure in `src/`:

```
tests/
├── conftest.py                     # Shared pytest configuration and fixtures
├── __init__.py                     # Main tests package
├── cime_src_share_util/           # Tests for CIME shared utilities
├── clm_src_biogeophys/            # Tests for biogeophysics modules
│   ├── test_CanopyStateType.py    # Canopy state type tests
│   └── test_SoilTemperatureMod.py # Soil temperature model tests
├── clm_src_cpl/                   # Tests for coupler interfaces
├── clm_src_main/                  # Tests for main CLM modules
│   └── test_decompMod.py          # Decomposition module tests
├── clm_src_utils/                 # Tests for utility modules
├── multilayer_canopy/             # Tests for multi-layer canopy model
└── offline_driver/                # Tests for offline driver
```

## Running Tests

### Run all tests:
```bash
pytest
```

### Run tests for a specific module:
```bash
pytest tests/clm_src_biogeophys/
pytest tests/clm_src_main/test_decompMod.py
```

### Run tests with coverage:
```bash
pytest --cov=src --cov-report=html
```

### Run only fast tests (exclude slow tests):
```bash
pytest -m "not slow"
```

## Test Configuration

- **pytest.ini**: Main pytest configuration in the project root
- **conftest.py**: Shared fixtures and test configuration
- Tests use JAX CPU backend by default for consistency
- Coverage reports are generated in `htmlcov/`

## Writing New Tests

1. Create test files following the pattern `test_<module_name>.py`
2. Place tests in the appropriate subdirectory matching the source structure
3. Use the shared fixtures from `conftest.py` for common test data
4. Mark slow tests with `@pytest.mark.slow` decorator
5. Use descriptive test function names starting with `test_`

## Test Types

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test interaction between modules
- **Property tests**: Test mathematical properties and invariants
- **Performance tests**: Test computational efficiency (marked as slow)