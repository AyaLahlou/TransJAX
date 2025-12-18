# Test Documentation: abortutils Module

## Test Suite Overview

The `abortutils` module provides error handling and program termination utilities for the CLM (Community Land Model) Python/JAX implementation. This test suite validates the behavior of abort and error handling functions.

### Functions Tested

1. **`endrun(msg: Optional[str])`** - Program termination with optional message
2. **`handle_err(status: int, errmsg: str)`** - NetCDF error handler
3. **`check_netcdf_status(status: int, operation: str)`** - NetCDF status checker
4. **`assert_condition(condition: bool, msg: str)`** - Conditional assertion with abort
5. **`warn_and_continue(msg: str)`** - Warning without termination

### Exception Classes Tested

1. **`CLMError`** - Base exception class
2. **`CLMNetCDFError`** - NetCDF-specific errors
3. **`CLMInitializationError`** - Initialization errors
4. **`CLMComputationError`** - Computation errors

### Test Coverage

- **Total Test Cases**: 10 nominal cases
- **Test Types**: 
  - Nominal/typical behavior
  - Error condition handling
  - Exception hierarchy validation
  - Side effect verification (sys.exit, logging)

### Coverage Areas

- ✅ Program termination behavior
- ✅ Error message formatting and output
- ✅ NetCDF status code handling
- ✅ Conditional assertions
- ✅ Warning messages without termination
- ✅ Exception class hierarchy
- ✅ Logging integration
- ✅ Exit code verification

## Running the Tests

### Basic Execution

```bash
# Run all tests in the module
pytest test_abortutils.py

# Run with verbose output
pytest test_abortutils.py -v

# Run specific test function
pytest test_abortutils.py::test_endrun_with_message

# Run tests matching pattern
pytest test_abortutils.py -k "netcdf"
```

### Coverage Analysis

```bash
# Run with coverage report
pytest test_abortutils.py --cov=abortutils

# Generate HTML coverage report
pytest test_abortutils.py --cov=abortutils --cov-report=html

# Show missing lines
pytest test_abortutils.py --cov=abortutils --cov-report=term-missing
```

### Advanced Options

```bash
# Run with output capture disabled (see print statements)
pytest test_abortutils.py -s

# Run with detailed traceback
pytest test_abortutils.py --tb=long

# Run in parallel (requires pytest-xdist)
pytest test_abortutils.py -n auto

# Stop on first failure
pytest test_abortutils.py -x
```

## Test Cases

### 1. Nominal Cases

#### `test_endrun_with_message`
- **Purpose**: Verify `endrun()` terminates with custom message
- **Behavior**: Should call `sys.exit(1)` and print message
- **Validation**: Uses `pytest.raises(SystemExit)` to catch termination

#### `test_endrun_without_message`
- **Purpose**: Verify `endrun()` works with no message
- **Behavior**: Should terminate gracefully with default behavior
- **Validation**: Confirms exit code is 1

#### `test_handle_err_with_error_status`
- **Purpose**: Test NetCDF error handling with non-zero status
- **Behavior**: Should terminate when status != NF_NOERR (0)
- **Validation**: Verifies error message includes status code and custom message

#### `test_handle_err_with_success_status`
- **Purpose**: Test NetCDF error handling with success status
- **Behavior**: Should NOT terminate when status == NF_NOERR (0)
- **Validation**: Function returns normally without exception

#### `test_check_netcdf_status_error`
- **Purpose**: Verify NetCDF status checking with error code
- **Behavior**: Should call `handle_err` and terminate
- **Validation**: Confirms termination with appropriate error message

#### `test_check_netcdf_status_success`
- **Purpose**: Verify NetCDF status checking with success code
- **Behavior**: Should return normally without error
- **Validation**: No exception raised

#### `test_assert_condition_true`
- **Purpose**: Test assertion with true condition
- **Behavior**: Should continue execution without termination
- **Validation**: Function completes normally

#### `test_assert_condition_false`
- **Purpose**: Test assertion with false condition
- **Behavior**: Should call `endrun()` and terminate
- **Validation**: Verifies SystemExit with error message

#### `test_warn_and_continue`
- **Purpose**: Verify warning function doesn't terminate
- **Behavior**: Should print warning and continue execution
- **Validation**: No exception raised, function returns normally

#### `test_exception_hierarchy`
- **Purpose**: Validate exception class relationships
- **Behavior**: Confirms inheritance chain is correct
- **Validation**: 
  - `CLMNetCDFError` is subclass of `CLMError`
  - `CLMInitializationError` is subclass of `CLMError`
  - `CLMComputationError` is subclass of `CLMError`
  - All inherit from base `Exception`

### 2. Edge Cases

**Note**: This module primarily deals with error handling and termination. Edge cases are limited because:
- Functions either terminate (sys.exit) or continue
- No numerical computations or array operations
- String messages are arbitrary

Potential edge cases to consider in future iterations:
- Very long error messages (>1000 characters)
- Unicode/special characters in messages
- Concurrent calls to abort functions
- Memory constraints with large error messages

### 3. Special Cases

#### NetCDF Status Codes
- **NF_NOERR (0)**: Success - no error
- **Non-zero values**: Various NetCDF error conditions
- Tests validate both success and failure paths

#### Logging Integration
- All functions should log to `iulog` (CLM log file)
- Tests verify logging calls are made (via mocking in implementation)

#### Exit Codes
- All termination functions use exit code `1`
- Consistent with Unix convention for error termination

## Test Data

### Data Generation Strategy

Since `abortutils` is a utility module for error handling, test data is minimal and straightforward:

1. **String Messages**: 
   - Simple descriptive strings
   - Empty strings
   - None values (for optional parameters)

2. **Status Codes**:
   - `0` (NF_NOERR) - success
   - `-1`, `1`, `100` - various error codes

3. **Boolean Conditions**:
   - `True` - assertion passes
   - `False` - assertion fails

4. **Operation Descriptions**:
   - Descriptive strings like "opening file", "reading variable"

### Coverage Rationale

The test data covers:
- ✅ All function parameters (required and optional)
- ✅ Both success and failure paths
- ✅ Default parameter values
- ✅ Exception class instantiation
- ✅ Inheritance relationships

## Expected Behavior

### Should Pass ✅

1. **Termination Functions**: All calls to `endrun()`, `handle_err()` (with error status), `check_netcdf_status()` (with error status), and `assert_condition()` (with False) should raise `SystemExit`

2. **Non-Termination Functions**: 
   - `warn_and_continue()` should complete normally
   - `handle_err()` with status=0 should complete normally
   - `check_netcdf_status()` with status=0 should complete normally
   - `assert_condition()` with True should complete normally

3. **Exception Hierarchy**: All exception classes should properly inherit from their parent classes

### Should Fail ❌

Tests will fail if:
- Termination functions don't call `sys.exit(1)`
- Non-termination functions unexpectedly exit
- Exception classes don't inherit correctly
- Error messages aren't formatted properly
- Exit codes are incorrect (not 1)

### Common Assertions

```python
# Termination expected
with pytest.raises(SystemExit) as exc_info:
    endrun("Error occurred")
assert exc_info.value.code == 1

# No termination expected
warn_and_continue("Warning message")  # Should not raise

# Exception hierarchy
assert issubclass(CLMNetCDFError, CLMError)
assert issubclass(CLMError, Exception)
```

## Extending Tests

### Adding New Test Cases

1. **New Function Tests**:

```python
def test_new_abort_function():
    """Test description."""
    # Arrange
    test_input = "test value"
    
    # Act & Assert
    with pytest.raises(SystemExit) as exc_info:
        new_abort_function(test_input)
    
    assert exc_info.value.code == 1
    # Add additional assertions for logging, message format, etc.
```

2. **Parametrized Tests** for multiple inputs:

```python
@pytest.mark.parametrize("status,should_exit", [
    (0, False),      # Success
    (-1, True),      # Error
    (1, True),       # Error
    (100, True),     # Error
])
def test_handle_err_parametrized(status, should_exit):
    """Test handle_err with various status codes."""
    if should_exit:
        with pytest.raises(SystemExit):
            handle_err(status, "Test error")
    else:
        handle_err(status, "Test error")  # Should not raise
```

3. **Testing Logging Output**:

```python
def test_endrun_logging(caplog):
    """Verify error message is logged."""
    with pytest.raises(SystemExit):
        endrun("Test error message")
    
    assert "Test error message" in caplog.text
```

4. **Testing with Mock Objects**:

```python
from unittest.mock import patch, MagicMock

def test_endrun_with_mock_logger():
    """Test endrun with mocked logger."""
    with patch('abortutils.iulog') as mock_log:
        with pytest.raises(SystemExit):
            endrun("Test message")
        
        mock_log.write.assert_called()
```

### Adding New Exception Tests

```python
def test_new_exception_class():
    """Test new CLM exception class."""
    # Test instantiation
    exc = CLMNewError("Test error")
    assert str(exc) == "Test error"
    
    # Test inheritance
    assert isinstance(exc, CLMError)
    assert isinstance(exc, Exception)
    
    # Test raising
    with pytest.raises(CLMNewError):
        raise CLMNewError("Test error")
```

### Testing Integration with Other Modules

```python
def test_abort_in_computation_context():
    """Test abort behavior when called from computation."""
    from some_module import some_function_that_may_abort
    
    with pytest.raises(SystemExit):
        some_function_that_may_abort(invalid_input)
```

## Common Issues

### Issue 1: Tests Hang or Don't Complete

**Problem**: Test appears to hang when testing termination functions.

**Cause**: Not properly catching `SystemExit` exception.

**Solution**:
```python
# ❌ Wrong - will terminate test runner
endrun("Error")

# ✅ Correct - catches exit
with pytest.raises(SystemExit):
    endrun("Error")
```

### Issue 2: Captured Output Not Visible

**Problem**: Can't see print statements or error messages during test.

**Cause**: pytest captures stdout/stderr by default.

**Solution**:
```bash
# Run with capture disabled
pytest test_abortutils.py -s

# Or use capfd fixture
def test_with_output(capfd):
    endrun("Test")
    captured = capfd.readouterr()
    assert "Test" in captured.out
```

### Issue 3: Import Errors

**Problem**: `ModuleNotFoundError` when running tests.

**Cause**: Python path not set correctly or missing dependencies.

**Solution**:
```bash
# Run from project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest test_abortutils.py

# Or install package in development mode
pip install -e .
```

### Issue 4: Logging Tests Fail

**Problem**: Logging assertions don't work as expected.

**Cause**: Logger not configured or wrong logger captured.

**Solution**:
```python
# Use caplog fixture
def test_logging(caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            endrun("Error message")
    
    assert "Error message" in caplog.text
```

### Issue 5: Exception Hierarchy Tests Fail

**Problem**: `issubclass()` assertions fail unexpectedly.

**Cause**: Exception classes not imported correctly or circular imports.

**Solution**:
```python
# Ensure proper imports
from abortutils import CLMError, CLMNetCDFError

# Verify import worked
assert CLMError is not None
assert CLMNetCDFError is not None

# Then test hierarchy
assert issubclass(CLMNetCDFError, CLMError)
```

### Issue 6: Mock Objects Not Working

**Problem**: Mocked functions still execute original code.

**Cause**: Incorrect patch path or patch applied after import.

**Solution**:
```python
# ❌ Wrong - patches wrong location
with patch('sys.exit') as mock_exit:
    endrun("Test")

# ✅ Correct - patches where it's used
with patch('abortutils.sys.exit') as mock_exit:
    endrun("Test")
    mock_exit.assert_called_once_with(1)
```

### Issue 7: Tests Pass Locally but Fail in CI

**Problem**: Tests work on local machine but fail in continuous integration.

**Cause**: Environment differences (Python version, dependencies, file paths).

**Solution**:
- Pin dependency versions in `requirements.txt`
- Use `tox` to test multiple Python versions
- Check for hardcoded paths or environment-specific assumptions
- Review CI logs for specific error messages

```bash
# Test with tox for multiple environments
tox -e py38,py39,py310
```

## Best Practices

1. **Always catch SystemExit**: Use `pytest.raises(SystemExit)` for termination tests
2. **Verify exit codes**: Check that `exc_info.value.code == 1`
3. **Test both paths**: Success and failure conditions
4. **Use fixtures**: For common test setup (though minimal for this module)
5. **Clear test names**: Use descriptive names that explain what's being tested
6. **Document expected behavior**: Add docstrings to test functions
7. **Isolate tests**: Each test should be independent
8. **Mock external dependencies**: Use mocks for logging, file I/O, etc.

## Maintenance Notes

- **Module Stability**: This is a utility module with stable API
- **Update Frequency**: Low - only when adding new error types
- **Breaking Changes**: Unlikely - error handling patterns are well-established
- **Dependencies**: Minimal - only standard library (sys, logging)

## Future Enhancements

Potential areas for test expansion:

1. **Performance Tests**: Verify error handling doesn't introduce significant overhead
2. **Concurrency Tests**: Test behavior with multiple threads/processes
3. **Integration Tests**: Test with actual NetCDF operations
4. **Stress Tests**: Very long error messages, rapid repeated calls
5. **Localization Tests**: Unicode and international character support
6. **Memory Tests**: Verify no memory leaks in error paths

---

**Last Updated**: 2024
**Test Framework**: pytest 7.x+
**Python Version**: 3.8+
**Maintainer**: CLM-JAX Development Team