# Test Documentation: ncdio_pio Module

## Test Suite Overview

This test suite provides comprehensive coverage for the `ncdio_pio` module, which handles NetCDF file I/O operations using a PIO (Parallel I/O) interface abstraction. The module provides both low-level file operations and high-level data I/O functions.

### Functions Tested

The test suite covers **11 functions** across three categories:

#### File Management Functions
- `ncd_pio_openfile` - Opens NetCDF files with various access modes
- `ncd_pio_closefile` - Closes open NetCDF files and cleans up resources
- `create_simple_netcdf_file` - Creates test NetCDF files with variables
- `print_netcdf_summary` - Displays file metadata and structure

#### Metadata Query Functions
- `ncd_inqdid` - Queries dimension IDs by name
- `ncd_inqdlen` - Retrieves dimension lengths by ID or name

#### Data I/O Functions
- `ncd_io` - Generic I/O dispatcher (1D or 2D)
- `ncd_io_1d` - Specialized 1D array I/O
- `ncd_io_2d` - Specialized 2D array I/O

#### Compatibility Functions
- `ncd_defvar` - Dummy routine for API compatibility
- `ncd_inqvdlen` - Dummy routine for API compatibility

### Test Statistics

- **Total Test Cases**: 10 parametrized test scenarios
- **Test Types**:
  - **Nominal Cases** (40%): Standard usage patterns with typical data
  - **Edge Cases** (40%): Boundary conditions, empty arrays, extreme values
  - **Special Cases** (20%): Error conditions, invalid inputs, state transitions

### Coverage Areas

1. **File Operations** (25%)
   - Opening files in different modes (read, write, append, read-write)
   - Closing files and resource cleanup
   - File creation with various dimension configurations
   - Error handling for missing files and invalid modes

2. **Dimension Queries** (15%)
   - Querying dimension IDs by name
   - Retrieving dimension lengths by ID or name
   - Handling unlimited dimensions
   - Error cases for non-existent dimensions

3. **Data I/O Operations** (45%)
   - Reading/writing 1D arrays
   - Reading/writing 2D arrays
   - Time-indexed data access
   - Variable existence checking
   - Data type conversions (float32, float64, int32, int64)
   - Array shape validation

4. **Data Classes and State Management** (10%)
   - `file_desc_t` lifecycle and validation
   - `NetCDFIOManager` file tracking
   - File mode enumeration
   - Data type enumeration

5. **Error Handling and Edge Cases** (5%)
   - Invalid file descriptors
   - Closed file operations
   - Non-existent variables
   - Dimension mismatches
   - Invalid flag values

## Running the Tests

### Basic Execution

```bash
# Run all tests
pytest test_ncdio_pio.py

# Run with verbose output
pytest test_ncdio_pio.py -v

# Run specific test function
pytest test_ncdio_pio.py::test_ncd_pio_openfile -v

# Run tests matching a pattern
pytest test_ncdio_pio.py -k "io_1d" -v
```

### Coverage Analysis

```bash
# Generate coverage report
pytest test_ncdio_pio.py --cov=ncdio_pio --cov-report=html

# Coverage with missing lines
pytest test_ncdio_pio.py --cov=ncdio_pio --cov-report=term-missing

# Coverage threshold enforcement
pytest test_ncdio_pio.py --cov=ncdio_pio --cov-fail-under=80
```

### Advanced Options

```bash
# Run with detailed output and stop on first failure
pytest test_ncdio_pio.py -vv -x

# Run only failed tests from last run
pytest test_ncdio_pio.py --lf

# Run tests in parallel (requires pytest-xdist)
pytest test_ncdio_pio.py -n auto

# Generate JUnit XML report for CI/CD
pytest test_ncdio_pio.py --junitxml=test-results.xml

# Run with warnings displayed
pytest test_ncdio_pio.py -v -W default
```

### Filtering Tests

```bash
# Run only nominal cases
pytest test_ncdio_pio.py -m "not edge and not special"

# Run only edge cases
pytest test_ncdio_pio.py -k "edge"

# Skip slow tests
pytest test_ncdio_pio.py -m "not slow"
```

## Test Cases

### Nominal Cases (Standard Usage)

**Purpose**: Verify correct behavior under typical operating conditions with realistic data.

1. **File Open/Close Cycle**
   - Opens existing NetCDF file in read mode
   - Verifies file descriptor state transitions
   - Closes file and confirms cleanup
   - **Expected**: Successful open/close with proper state management

2. **1D Array Read/Write**
   - Creates test file with 1D temperature data (273.15-373.15 K)
   - Writes array to file
   - Reads array back and verifies values
   - **Expected**: Data integrity preserved, values match within tolerance

3. **2D Array Read/Write**
   - Creates test file with 2D spatial data (100x50 grid)
   - Writes multi-dimensional array
   - Reads back and verifies shape and values
   - **Expected**: Correct shape preservation, value accuracy

4. **Dimension Queries**
   - Queries standard dimensions (time, lat, lon, lev)
   - Retrieves dimension lengths
   - **Expected**: Correct dimension IDs and lengths returned

5. **Time-Indexed Access**
   - Reads specific time slices from multi-temporal dataset
   - Verifies correct temporal indexing (nt parameter)
   - **Expected**: Correct time slice extracted

### Edge Cases (Boundary Conditions)

**Purpose**: Test behavior at limits and boundaries of valid input ranges.

1. **Empty Arrays**
   - Attempts I/O with zero-length arrays
   - Tests dimension queries on empty dimensions
   - **Expected**: Graceful handling or appropriate error messages

2. **Single-Element Arrays**
   - I/O operations with arrays of length 1
   - Minimum valid dimension sizes
   - **Expected**: Correct handling of minimal valid inputs

3. **Large Arrays**
   - Tests with arrays approaching memory limits (10^6+ elements)
   - Verifies chunking and buffering strategies
   - **Expected**: Successful I/O without memory errors

4. **Extreme Values**
   - Near-zero temperatures (0.01 K)
   - Very large values (1e10)
   - Negative values where physically invalid
   - **Expected**: Values preserved, validation warnings where appropriate

5. **Boundary Dimensions**
   - 1D arrays at minimum/maximum supported sizes
   - 2D arrays with extreme aspect ratios (1x10000, 10000x1)
   - **Expected**: Correct handling of unusual but valid shapes

6. **Time Index Boundaries**
   - nt=0 (first time step)
   - nt=max_time-1 (last time step)
   - **Expected**: Correct boundary indexing

### Special Cases (Error Conditions & Edge Scenarios)

**Purpose**: Verify robust error handling and recovery from invalid states.

1. **Invalid File Operations**
   - Opening non-existent files
   - Invalid file modes ("x", "invalid")
   - Operations on closed files
   - **Expected**: Clear error messages, no crashes

2. **Non-Existent Variables**
   - Reading variables not in file
   - Querying undefined dimensions
   - **Expected**: readvar=False, appropriate error handling

3. **Dimension Mismatches**
   - Writing data with wrong shape for variable
   - Reading into incorrectly sized arrays
   - **Expected**: Shape validation errors

4. **Invalid Flags**
   - ncd_io with flag="invalid" (not "read" or "write")
   - Mixed case flags ("Read", "WRITE")
   - **Expected**: Clear error messages about valid flag values

5. **State Transition Errors**
   - Double-closing files
   - Opening already-open files
   - Reading from write-only files
   - **Expected**: State validation prevents invalid operations

6. **Special Float Values**
   - Arrays containing NaN
   - Arrays containing Inf/-Inf
   - **Expected**: Proper handling or warnings about non-physical values

7. **Concurrent Access**
   - Multiple file descriptors to same file
   - Read/write conflicts
   - **Expected**: Proper file locking or clear error messages

## Test Data

### Generation Strategy

Test data is generated using a combination of approaches:

1. **Synthetic Data Generation**
   ```python
   # Temperature profiles (physically realistic)
   temps = jnp.linspace(273.15, 373.15, 100)  # 0°C to 100°C
   
   # Spatial grids
   lat = jnp.linspace(-90, 90, 180)
   lon = jnp.linspace(-180, 180, 360)
   
   # Random but seeded for reproducibility
   rng = jax.random.PRNGKey(42)
   data = jax.random.normal(rng, shape=(100, 50))
   ```

2. **Fixture-Based Test Files**
   - Temporary NetCDF files created per test
   - Cleaned up automatically after test completion
   - Isolated test environments prevent interference

3. **Parametrized Test Data**
   ```python
   @pytest.mark.parametrize("shape,dtype", [
       ((100,), jnp.float32),
       ((50, 30), jnp.float64),
       ((1000,), jnp.int32),
   ])
   ```

### Coverage Matrix

| Data Type | 1D Arrays | 2D Arrays | Edge Cases | Special Values |
|-----------|-----------|-----------|------------|----------------|
| float32   | ✓         | ✓         | ✓          | ✓ (NaN/Inf)    |
| float64   | ✓         | ✓         | ✓          | ✓ (NaN/Inf)    |
| int32     | ✓         | ✓         | ✓          | ✓ (min/max)    |
| int64     | ✓         | ✓         | ✓          | ✓ (min/max)    |

### Physical Realism Constraints

- **Temperature**: Always > 0 K (absolute zero constraint)
- **Fractions**: Values in [0, 1] range
- **Pressure**: Positive values only
- **Coordinates**: Lat ∈ [-90, 90], Lon ∈ [-180, 180]

### Dimension Configurations Tested

- **1D**: Sizes from 1 to 10,000 elements
- **2D**: Shapes including (10,10), (100,50), (1,1000), (1000,1)
- **Time Series**: 1 to 365 time steps
- **Vertical Levels**: 1 to 100 levels

## Expected Behavior

### Successful Operations

1. **File Opening**
   - Returns valid file descriptor
   - `is_open` flag set to True
   - File path correctly stored
   - Mode correctly set

2. **Data Reading**
   - Returns tuple (data, True) on success
   - Data shape matches variable dimensions
   - Data type preserved or correctly converted
   - Values within expected tolerance (1e-6 for floats)

3. **Data Writing**
   - Returns tuple (data, True) on success
   - Data persisted to file
   - Readable by subsequent operations
   - Metadata correctly stored

4. **Dimension Queries**
   - Returns valid integer dimension IDs (≥ 0)
   - Returns correct dimension lengths
   - Handles both ID and name lookups

### Expected Failures

1. **File Not Found**
   - Raises `FileNotFoundError` or returns error status
   - Clear error message with filename

2. **Invalid Mode**
   - Raises `ValueError` with valid mode options
   - Does not corrupt file descriptor

3. **Closed File Operations**
   - Raises `RuntimeError` or returns (None, False)
   - Error message indicates file is closed

4. **Variable Not Found**
   - Returns (data, False) with readvar=False
   - Does not raise exception (graceful degradation)
   - Original data array unchanged

5. **Shape Mismatch**
   - Raises `ValueError` with shape information
   - Indicates expected vs. actual shapes

### Tolerance Specifications

- **Float32**: Absolute tolerance = 1e-6, Relative tolerance = 1e-5
- **Float64**: Absolute tolerance = 1e-12, Relative tolerance = 1e-10
- **Integer**: Exact match required
- **Dimensions**: Exact match required

## Extending Tests

### Adding New Test Cases

1. **Create Test Function**
   ```python
   def test_new_feature(tmp_path):
       """Test description.
       
       This test verifies that [specific behavior] works correctly
       when [specific conditions].
       """
       # Arrange
       test_file = tmp_path / "test.nc"
       ncid = create_simple_netcdf_file(
           str(test_file),
           {"var": jnp.array([1, 2, 3])}
       )
       
       # Act
       result = function_under_test(ncid, ...)
       
       # Assert
       assert result == expected_value
       
       # Cleanup
       ncd_pio_closefile(ncid)
   ```

2. **Add Parametrized Cases**
   ```python
   @pytest.mark.parametrize("input_data,expected", [
       (jnp.array([1, 2, 3]), True),
       (jnp.array([]), False),
       (jnp.array([jnp.nan]), False),
   ])
   def test_with_params(input_data, expected):
       result = function(input_data)
       assert result == expected
   ```

3. **Add Fixtures for Complex Setup**
   ```python
   @pytest.fixture
   def complex_netcdf_file(tmp_path):
       """Create NetCDF file with multiple variables and dimensions."""
       filepath = tmp_path / "complex.nc"
       # ... setup code ...
       yield filepath
       # ... cleanup code ...
   ```

### Test Organization Guidelines

- **One concept per test**: Each test should verify one specific behavior
- **Clear naming**: `test_<function>_<scenario>_<expected_result>`
- **AAA Pattern**: Arrange, Act, Assert structure
- **Docstrings**: Explain what and why, not how
- **Cleanup**: Use fixtures or try/finally for resource cleanup

### Adding Test Categories

```python
# Mark tests with custom markers
@pytest.mark.slow
def test_large_file_io():
    """Test with very large files (may take >10s)."""
    pass

@pytest.mark.integration
def test_full_workflow():
    """Test complete read-modify-write cycle."""
    pass

# Register markers in pytest.ini or conftest.py
# markers =
#     slow: marks tests as slow (deselect with '-m "not slow"')
#     integration: marks tests as integration tests
```

### Property-Based Testing

For more comprehensive coverage, consider adding property-based tests:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=0, max_value=1000), min_size=1, max_size=100))
def test_io_preserves_data(data_list):
    """Property: Reading written data returns original values."""
    data = jnp.array(data_list)
    # ... test that write then read returns same data ...
```

## Common Issues

### Issue 1: File Locking Errors

**Symptom**: `PermissionError` or "file is locked" messages

**Cause**: File not properly closed in previous test

**Solution**:
```python
# Always use try/finally or context managers
try:
    ncid = ncd_pio_openfile(...)
    # ... operations ...
finally:
    ncd_pio_closefile(ncid)

# Or use fixtures with proper cleanup
@pytest.fixture
def open_file(tmp_path):
    ncid = ncd_pio_openfile(...)
    yield ncid
    ncd_pio_closefile(ncid)
```

### Issue 2: Floating Point Comparison Failures

**Symptom**: Tests fail with small differences (e.g., 1e-7)

**Cause**: Floating point arithmetic is not exact

**Solution**:
```python
# Use numpy testing utilities
import numpy.testing as npt

npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

# Or pytest.approx
assert actual == pytest.approx(expected, rel=1e-5, abs=1e-8)
```

### Issue 3: Temporary File Cleanup

**Symptom**: Disk fills up with test files, or tests interfere with each other

**Cause**: Test files not cleaned up properly

**Solution**:
```python
# Use pytest's tmp_path fixture (automatic cleanup)
def test_with_temp_file(tmp_path):
    test_file = tmp_path / "test.nc"
    # ... test code ...
    # File automatically deleted after test

# Or use tempfile module
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    # ... test code ...
```

### Issue 4: JAX Array Mutability

**Symptom**: `TypeError: JAX arrays are immutable`

**Cause**: Attempting to modify JAX array in-place

**Solution**:
```python
# Don't do this:
data[0] = 5  # Error!

# Do this instead:
data = data.at[0].set(5)  # Returns new array

# Or convert to numpy for modification
data_np = np.array(data)
data_np[0] = 5
data = jnp.array(data_np)
```

### Issue 5: Dimension Order Confusion

**Symptom**: Shape mismatches or transposed data

**Cause**: NetCDF uses C-order (row-major), confusion with Fortran-order

**Solution**:
```python
# Be explicit about dimension order
# NetCDF: (time, lev, lat, lon)
# JAX: same order by default

# Document expected shapes clearly
def test_2d_array():
    """Test with shape (nlat=180, nlon=360)."""
    data = jnp.zeros((180, 360))  # lat first, lon second
```

### Issue 6: Test Data Randomness

**Symptom**: Tests pass sometimes, fail other times

**Cause**: Unseeded random number generation

**Solution**:
```python
# Always seed random number generators
rng = jax.random.PRNGKey(42)  # Fixed seed
data = jax.random.normal(rng, shape=(100,))

# Or use hypothesis for controlled randomness
from hypothesis import given, strategies as st, seed
@seed(42)
@given(st.lists(st.floats()))
def test_with_random_data(data):
    pass
```

### Issue 7: NetCDF Library Version Differences

**Symptom**: Tests pass locally but fail in CI/CD

**Cause**: Different NetCDF library versions have different behaviors

**Solution**:
```python
# Pin dependencies in requirements.txt
# netCDF4==1.6.0
# xarray==2023.1.0

# Or skip tests based on version
import netCDF4
@pytest.mark.skipif(
    netCDF4.__version__ < "1.6.0",
    reason="Requires netCDF4 >= 1.6.0"
)
def test_new_feature():
    pass
```

### Issue 8: Memory Leaks in Long Test Runs

**Symptom**: Tests slow down or fail after many iterations

**Cause**: File handles or datasets not released

**Solution**:
```python
# Explicitly close all resources
def test_many_files():
    for i in range(1000):
        ncid = ncd_pio_openfile(...)
        # ... operations ...
        ncd_pio_closefile(ncid)  # Don't forget!
        
# Use context managers when available
with xr.open_dataset(file) as ds:
    # ... operations ...
    pass  # Automatically closed
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest test_ncdio_pio.py --cov=ncdio_pio --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Performance Benchmarks

For performance-critical operations, consider adding benchmarks:

```python
import pytest

@pytest.mark.benchmark
def test_large_array_io_performance(benchmark, tmp_path):
    """Benchmark I/O performance with large arrays."""
    data = jnp.zeros((1000, 1000))
    
    def io_operation():
        # ... write and read operation ...
        pass
    
    result = benchmark(io_operation)
    assert result is not None
```

Run with: `pytest test_ncdio_pio.py --benchmark-only`

---

**Last Updated**: 2024
**Test Suite Version**: 1.0
**Minimum Python Version**: 3.8
**Required Dependencies**: pytest, jax, numpy, xarray, netCDF4