# Test Documentation: clm_varctl Module

## Test Suite Overview

This test suite provides comprehensive coverage for the `clm_varctl` module, which manages runtime control variables for the Community Land Model (CLM). The module provides an immutable configuration system using named tuples.

### Functions Tested

1. **`create_clm_varctl()`** - Factory function for creating configuration objects
2. **`update_clm_varctl()`** - Immutable update function for configuration objects
3. **`get_log_unit()`** - Accessor function for log unit number
4. **`validate_clm_varctl()`** - Validation function for configuration objects

### Test Statistics

- **Total Test Cases**: 10 parametrized cases
- **Test Types**: 
  - Nominal cases (typical usage patterns)
  - Edge cases (boundary conditions)
  - Special cases (error conditions and validation)
- **Coverage Areas**:
  - Configuration creation with default and custom parameters
  - Configuration updates and immutability
  - Parameter validation and constraints
  - Accessor functions
  - Error handling

## Running the Tests

### Basic Execution

```bash
# Run all tests in the module
pytest test_clm_varctl.py

# Run with verbose output
pytest test_clm_varctl.py -v

# Run specific test function
pytest test_clm_varctl.py::test_create_clm_varctl

# Run with specific markers (if defined)
pytest test_clm_varctl.py -m "not slow"
```

### Coverage Analysis

```bash
# Run with coverage report
pytest test_clm_varctl.py --cov=clm_varctl

# Generate HTML coverage report
pytest test_clm_varctl.py --cov=clm_varctl --cov-report=html

# Show missing lines
pytest test_clm_varctl.py --cov=clm_varctl --cov-report=term-missing
```

### Advanced Options

```bash
# Run with detailed output and stop on first failure
pytest test_clm_varctl.py -vv -x

# Run only failed tests from last run
pytest test_clm_varctl.py --lf

# Run tests in parallel (requires pytest-xdist)
pytest test_clm_varctl.py -n auto

# Generate JUnit XML report for CI/CD
pytest test_clm_varctl.py --junitxml=test-results.xml
```

## Test Cases

### Nominal Cases

**Purpose**: Verify correct behavior under typical usage scenarios.

- **Default Configuration**: Tests `create_clm_varctl()` with no arguments, expecting `iulog=6`
- **Custom Log Unit**: Tests creation with valid custom log unit numbers (e.g., `iulog=10`)
- **Configuration Update**: Tests `update_clm_varctl()` with valid parameter changes
- **Accessor Functions**: Tests `get_log_unit()` returns correct values

**Example**:
```python
# Default configuration
ctl = create_clm_varctl()
assert ctl.iulog == 6

# Custom configuration
ctl = create_clm_varctl(iulog=10)
assert ctl.iulog == 10
```

### Edge Cases

**Purpose**: Test boundary conditions and limits of valid inputs.

- **Minimum Valid Value**: Tests `iulog=1` (minimum valid unit number)
- **Large Valid Value**: Tests `iulog=999` (large but valid unit number)
- **Boundary Validation**: Ensures validation correctly identifies boundary values

**Example**:
```python
# Minimum valid unit
ctl = create_clm_varctl(iulog=1)
assert validate_clm_varctl(ctl) == True

# Large valid unit
ctl = create_clm_varctl(iulog=999)
assert validate_clm_varctl(ctl) == True
```

### Special Cases

**Purpose**: Verify error handling and validation logic.

- **Invalid Unit Numbers**: Tests rejection of `iulog <= 0` (zero, negative)
- **Invalid Types**: Tests rejection of non-integer types (float, string, None)
- **Immutability**: Verifies that configuration objects cannot be modified in-place
- **Update with Invalid Fields**: Tests error handling when updating with invalid parameters
- **Validation Function**: Tests `validate_clm_varctl()` correctly identifies invalid configurations

**Example**:
```python
# Invalid unit number (should raise ValueError)
with pytest.raises(ValueError):
    create_clm_varctl(iulog=0)

# Invalid type (should raise TypeError)
with pytest.raises(TypeError):
    create_clm_varctl(iulog=6.5)

# Immutability (should raise AttributeError)
ctl = create_clm_varctl()
with pytest.raises(AttributeError):
    ctl.iulog = 10
```

## Test Data

### Generation Strategy

Test data is generated using a systematic approach covering:

1. **Representative Values**: Common use cases (default=6, typical alternatives=7,8,10)
2. **Boundary Values**: Minimum (1) and practical maximum (999)
3. **Invalid Values**: Zero, negative numbers, non-integers
4. **Type Variations**: Integers, floats, strings, None

### Data Coverage Matrix

| Category | Values | Purpose |
|----------|--------|---------|
| Valid Integers | 1, 6, 7, 10, 999 | Nominal and edge cases |
| Invalid Integers | 0, -1, -10 | Constraint validation |
| Invalid Types | 6.5, "6", None | Type checking |
| Update Operations | Valid and invalid kwargs | Update function testing |

### Fixtures

The test suite uses pytest fixtures for:

- **`default_ctl`**: Provides a default configuration object for tests
- **`valid_iulog_values`**: Parametrized fixture with valid log unit numbers
- **`invalid_iulog_values`**: Parametrized fixture with invalid values for error testing

## Expected Behavior

### Should Pass ✓

1. **Creation with valid parameters**:
   - Default creation: `create_clm_varctl()`
   - Custom valid units: `create_clm_varctl(iulog=10)`
   - Minimum valid: `create_clm_varctl(iulog=1)`

2. **Updates with valid parameters**:
   - `update_clm_varctl(ctl, iulog=7)`
   - Returns new object, original unchanged

3. **Validation of valid configurations**:
   - `validate_clm_varctl(valid_ctl)` returns `True`

4. **Accessor functions**:
   - `get_log_unit(ctl)` returns correct integer

### Should Fail ✗

1. **Creation with invalid parameters**:
   - Zero or negative: `create_clm_varctl(iulog=0)` → `ValueError`
   - Non-integer: `create_clm_varctl(iulog=6.5)` → `TypeError`
   - None: `create_clm_varctl(iulog=None)` → `TypeError`

2. **Invalid updates**:
   - Update with invalid value → `ValueError`
   - Update with invalid field name → `AttributeError` or `TypeError`

3. **Validation of invalid configurations**:
   - `validate_clm_varctl(invalid_ctl)` returns `False`

4. **Immutability violations**:
   - Direct attribute assignment → `AttributeError`

## Extending Tests

### Adding New Test Cases

1. **Add to parametrized fixtures**:
```python
@pytest.fixture(params=[1, 6, 10, 999, 1000])  # Add new value
def valid_iulog_values(request):
    return request.param
```

2. **Create new test function**:
```python
def test_new_functionality():
    """Test description."""
    ctl = create_clm_varctl(iulog=15)
    # Add assertions
    assert some_condition
```

3. **Add property-based tests** (using hypothesis):
```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=1000))
def test_valid_range_property(iulog):
    """Property: all positive integers should create valid configs."""
    ctl = create_clm_varctl(iulog=iulog)
    assert validate_clm_varctl(ctl)
```

### Adding New Functions to Test

When new functions are added to `clm_varctl`:

1. **Update test data generation** to include relevant cases
2. **Create dedicated test function**:
```python
def test_new_function():
    """Test new_function behavior."""
    # Setup
    ctl = create_clm_varctl()
    
    # Execute
    result = new_function(ctl)
    
    # Assert
    assert result == expected_value
```

3. **Add integration tests** if function interacts with existing functions
4. **Update this documentation** with new test descriptions

## Common Issues

### Issue 1: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'clm_varctl'`

**Solution**:
```bash
# Ensure module is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/module"

# Or install in development mode
pip install -e .
```

### Issue 2: JAX/NumPy Version Conflicts

**Problem**: Tests fail due to JAX version incompatibility

**Solution**:
```bash
# Check versions
pip list | grep jax

# Update to compatible versions
pip install --upgrade jax jaxlib
```

### Issue 3: Floating Point Comparison Failures

**Problem**: Tests fail due to floating point precision issues

**Solution**:
```python
# Use pytest.approx for floating point comparisons
assert result == pytest.approx(expected, rel=1e-9)

# Or use numpy testing utilities
import numpy.testing as npt
npt.assert_allclose(result, expected, rtol=1e-9)
```

### Issue 4: Parametrized Test Identification

**Problem**: Difficult to identify which parametrized case failed

**Solution**:
```bash
# Run with verbose output showing parameters
pytest test_clm_varctl.py -v

# Use custom test IDs in parametrize decorator
@pytest.mark.parametrize("iulog", [1, 6, 10], ids=["min", "default", "custom"])
```

### Issue 5: Slow Test Execution

**Problem**: Test suite takes too long to run

**Solution**:
```bash
# Run tests in parallel
pytest test_clm_varctl.py -n auto

# Run only fast tests (if marked)
pytest test_clm_varctl.py -m "not slow"

# Profile test execution
pytest test_clm_varctl.py --durations=10
```

### Issue 6: Coverage Not Reaching 100%

**Problem**: Some code paths not covered by tests

**Solution**:
```bash
# Generate detailed coverage report
pytest test_clm_varctl.py --cov=clm_varctl --cov-report=html

# Open htmlcov/index.html to see uncovered lines
# Add tests for missing branches/conditions
```

## Best Practices

### Test Organization

- **One test function per behavior**: Each test should verify one specific behavior
- **Clear test names**: Use descriptive names like `test_create_with_invalid_negative_iulog`
- **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification phases
- **Independent tests**: Each test should be runnable in isolation

### Assertions

```python
# Good: Specific assertion with message
assert ctl.iulog == 6, f"Expected iulog=6, got {ctl.iulog}"

# Good: Use pytest.raises for exceptions
with pytest.raises(ValueError, match="must be positive"):
    create_clm_varctl(iulog=-1)

# Good: Multiple related assertions
assert isinstance(ctl, ClmVarCtl)
assert ctl.iulog > 0
assert validate_clm_varctl(ctl)
```

### Documentation

- **Docstrings**: Every test function should have a docstring explaining what it tests
- **Comments**: Add comments for non-obvious test logic
- **Type hints**: Use type hints in test functions for clarity

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install pytest pytest-cov jax jaxlib
      - name: Run tests
        run: |
          pytest test_clm_varctl.py --cov=clm_varctl --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Maintenance

### Regular Updates

- **Review tests quarterly** to ensure they remain relevant
- **Update test data** when new edge cases are discovered
- **Refactor tests** to reduce duplication and improve clarity
- **Monitor coverage** and add tests for uncovered code paths

### Version Compatibility

- Test against multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Test against multiple JAX versions
- Document minimum required versions

---

**Last Updated**: 2024
**Test Suite Version**: 1.0
**Module Version**: Compatible with clm_varctl v1.0+