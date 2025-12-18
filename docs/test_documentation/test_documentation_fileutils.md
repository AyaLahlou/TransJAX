# Test Documentation: fileutils Module

## Test Suite Overview

This test suite provides comprehensive coverage for the `fileutils` module, which contains Python file I/O utility functions originally translated from Fortran. The module provides file path manipulation, file retrieval, file opening, and file closing operations.

### Functions Tested

1. **`get_filename(fulpath: str) -> str`**
   - Extracts filename from full path
   - Handles various path formats and edge cases

2. **`getfil(fulpath: str, iflag: int = 0) -> Tuple[str, bool]`**
   - Retrieves files from archival/permanent storage
   - Returns local filename and success status
   - Supports abort/no-abort modes

3. **`opnfil(locfn: str, iun: int, form: Literal['u', 'U', 'f', 'F']) -> Optional[object]`**
   - Opens files in binary or text mode
   - Returns file handle or None on failure

4. **`relavu(iunit: Optional[TextIO]) -> None`**
   - Safely closes file handles
   - Handles None inputs gracefully

### Test Statistics

- **Total Test Cases**: 10 comprehensive test scenarios
- **Test Types**:
  - **Nominal Cases** (40%): Standard usage patterns with typical inputs
  - **Edge Cases** (40%): Boundary conditions, empty strings, special characters
  - **Special Cases** (20%): Error conditions, None values, invalid inputs

### Coverage Areas

- ✅ Path parsing and filename extraction
- ✅ File format handling (binary/text)
- ✅ Error handling and validation
- ✅ File handle lifecycle management
- ✅ Cross-platform path compatibility
- ✅ Empty and malformed input handling
- ✅ Special characters in filenames
- ✅ File existence checking
- ✅ Resource cleanup

---

## Running the Tests

### Basic Execution

```bash
# Run all tests in the module
pytest test_fileutils.py

# Run with verbose output
pytest test_fileutils.py -v

# Run specific test function
pytest test_fileutils.py::test_get_filename_nominal

# Run tests matching a pattern
pytest test_fileutils.py -k "edge"
```

### Coverage Analysis

```bash
# Run with coverage report
pytest test_fileutils.py --cov=fileutils

# Generate HTML coverage report
pytest test_fileutils.py --cov=fileutils --cov-report=html

# Show missing lines
pytest test_fileutils.py --cov=fileutils --cov-report=term-missing
```

### Advanced Options

```bash
# Run with detailed output and stop on first failure
pytest test_fileutils.py -vx

# Run in parallel (requires pytest-xdist)
pytest test_fileutils.py -n auto

# Generate JUnit XML report
pytest test_fileutils.py --junitxml=test-results.xml

# Run with warnings displayed
pytest test_fileutils.py -v -W all
```

---

## Test Cases

### Nominal Cases (Standard Usage)

**Purpose**: Verify correct behavior under typical operating conditions

1. **Standard Unix Path**
   - Input: `/home/user/data/file.txt`
   - Expected: Extracts `file.txt` correctly
   - Tests: Basic path parsing

2. **Windows Path**
   - Input: `C:\Users\data\file.dat`
   - Expected: Handles backslashes appropriately
   - Tests: Cross-platform compatibility

3. **File Opening (Text Mode)**
   - Input: Valid filename, format='f'
   - Expected: Returns valid file handle in text mode
   - Tests: Formatted file operations

4. **File Opening (Binary Mode)**
   - Input: Valid filename, format='u'
   - Expected: Returns valid file handle in binary mode
   - Tests: Unformatted file operations

### Edge Cases (Boundary Conditions)

**Purpose**: Test behavior at limits and unusual but valid inputs

1. **Empty Path String**
   - Input: `""`
   - Expected: Returns empty string or handles gracefully
   - Tests: Minimal input handling

2. **Path Without Directory**
   - Input: `file.txt` (no slashes)
   - Expected: Returns entire string as filename
   - Tests: Simple filename extraction

3. **Path Ending with Separator**
   - Input: `/home/user/data/`
   - Expected: Returns empty string or last directory name
   - Tests: Trailing separator handling

4. **Multiple Consecutive Separators**
   - Input: `/home//user///file.txt`
   - Expected: Handles redundant separators correctly
   - Tests: Malformed path resilience

5. **Special Characters in Filename**
   - Input: `/data/file-name_v2.0[test].dat`
   - Expected: Preserves special characters
   - Tests: Character encoding and preservation

### Special Cases (Error Conditions)

**Purpose**: Verify proper error handling and edge case management

1. **None Input to relavu**
   - Input: `None`
   - Expected: No error, graceful handling
   - Tests: Null safety

2. **Invalid File Format Specifier**
   - Input: format='x' (invalid)
   - Expected: Raises ValueError or returns None
   - Tests: Input validation

3. **Non-existent File with iflag=0**
   - Input: Path to missing file, iflag=0
   - Expected: Abort behavior or exception
   - Tests: Error mode handling

4. **Non-existent File with iflag=1**
   - Input: Path to missing file, iflag=1
   - Expected: Returns (filename, False)
   - Tests: Non-abort mode handling

---

## Test Data

### Generation Strategy

Test data was systematically generated to cover:

1. **Path Variations**
   - Unix-style paths (`/path/to/file`)
   - Windows-style paths (`C:\path\to\file`)
   - Relative paths (`../data/file.txt`)
   - Network paths (`//server/share/file`)

2. **Filename Patterns**
   - Simple names (`file.txt`)
   - Complex names (`data_v2.0-final[backup].dat`)
   - Names with spaces (`my file.txt`)
   - Hidden files (`.config`)

3. **Format Specifiers**
   - All valid values: `'u'`, `'U'`, `'f'`, `'F'`
   - Invalid values for error testing

4. **File States**
   - Existing files (created in temp directory)
   - Non-existent files
   - Files with various permissions

### Coverage Matrix

| Function | Nominal | Edge | Error | Total |
|----------|---------|------|-------|-------|
| get_filename | 3 | 4 | 1 | 8 |
| getfil | 2 | 2 | 2 | 6 |
| opnfil | 4 | 2 | 2 | 8 |
| relavu | 2 | 1 | 1 | 4 |

---

## Expected Behavior

### Passing Tests Should Demonstrate

✅ **get_filename**
- Correctly extracts filename from any valid path format
- Returns empty string for paths ending in separator
- Handles paths without separators by returning entire string

✅ **getfil**
- Returns (local_filename, True) when file exists
- Returns (filename, False) when file missing and iflag=1
- Aborts or raises exception when file missing and iflag=0

✅ **opnfil**
- Returns valid file handle for existing files
- Opens in correct mode (binary for 'u'/'U', text for 'f'/'F')
- Returns None or raises exception for invalid inputs

✅ **relavu**
- Closes open file handles without error
- Handles None input gracefully
- Handles already-closed handles safely

### Failing Tests Indicate

❌ **Path Parsing Issues**
- Incorrect separator handling
- Platform-specific path problems
- Unicode/encoding errors

❌ **File Operation Failures**
- Permission errors
- File system access issues
- Resource leaks (unclosed handles)

❌ **Validation Problems**
- Accepting invalid format specifiers
- Not handling None/empty inputs
- Missing error checks

---

## Extending Tests

### Adding New Test Cases

1. **Create Test Data Fixture**

```python
@pytest.fixture
def new_test_case():
    """Description of what this tests."""
    return {
        'input': 'test_value',
        'expected': 'expected_result',
        'description': 'What this case validates'
    }
```

2. **Add Parametrized Test**

```python
@pytest.mark.parametrize("path,expected", [
    ("/new/path/file.txt", "file.txt"),
    ("//network/share/data.dat", "data.dat"),
])
def test_get_filename_new_cases(path, expected):
    """Test new path formats."""
    result = get_filename(path)
    assert result == expected
```

3. **Add Integration Test**

```python
def test_full_workflow(tmp_path):
    """Test complete file operation workflow."""
    # Create test file
    test_file = tmp_path / "test.dat"
    test_file.write_text("test data")
    
    # Test getfil -> opnfil -> relavu chain
    locfn, success = getfil(str(test_file), iflag=1)
    assert success
    
    fh = opnfil(locfn, 10, 'f')
    assert fh is not None
    
    relavu(fh)
    # Verify file is closed
```

### Test Organization Best Practices

- **Group related tests** using classes
- **Use descriptive names** following pattern `test_<function>_<scenario>`
- **Add docstrings** explaining test purpose
- **Use fixtures** for common setup/teardown
- **Parametrize** similar tests to reduce duplication

### Example Test Class Structure

```python
class TestGetFilename:
    """Tests for get_filename function."""
    
    def test_unix_paths(self):
        """Test Unix-style path handling."""
        pass
    
    def test_windows_paths(self):
        """Test Windows-style path handling."""
        pass
    
    def test_edge_cases(self):
        """Test boundary conditions."""
        pass
```

---

## Common Issues

### Issue 1: File Permission Errors

**Symptom**: Tests fail with `PermissionError`

**Cause**: Insufficient permissions to create/access test files

**Solution**:
```python
# Use tmp_path fixture for test files
def test_with_temp_file(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("data")
    # Test operations...
```

### Issue 2: Platform-Specific Path Failures

**Symptom**: Tests pass on Unix but fail on Windows (or vice versa)

**Cause**: Hardcoded path separators

**Solution**:
```python
import os
from pathlib import Path

# Use pathlib for cross-platform paths
test_path = Path("/home/user") / "file.txt"

# Or use os.path.join
test_path = os.path.join("home", "user", "file.txt")
```

### Issue 3: Resource Leaks (Unclosed Files)

**Symptom**: "Too many open files" error after many tests

**Cause**: File handles not properly closed

**Solution**:
```python
@pytest.fixture
def open_file(tmp_path):
    """Fixture that ensures file cleanup."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("data")
    fh = open(test_file, 'r')
    yield fh
    if not fh.closed:
        fh.close()
```

### Issue 4: Test Isolation Problems

**Symptom**: Tests pass individually but fail when run together

**Cause**: Shared state or file system side effects

**Solution**:
```python
# Use unique filenames per test
def test_operation(tmp_path):
    unique_file = tmp_path / f"test_{uuid.uuid4()}.txt"
    # Test operations...

# Or use autouse fixtures for cleanup
@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Cleanup code here
```

### Issue 5: Encoding Issues

**Symptom**: Tests fail with `UnicodeDecodeError`

**Cause**: File opened in wrong mode or encoding mismatch

**Solution**:
```python
# Explicitly specify encoding for text files
fh = open(filename, 'r', encoding='utf-8')

# Use binary mode for non-text data
fh = open(filename, 'rb')
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test fileutils

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        pytest test_fileutils.py --cov=fileutils --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Notes

- **Not JAX/JIT Compatible**: These functions perform I/O operations and cannot be JIT-compiled
- **Side Effects**: Functions modify file system state; tests should use temporary directories
- **Legacy Interface**: Some parameters (like `iun`) maintained for Fortran compatibility but unused
- **Error Handling**: Different functions use different error strategies (abort vs. return status)

---

## Maintenance

### Regular Updates Needed

- Add tests for new Python versions
- Update for new file system features
- Add tests for discovered edge cases
- Review and update platform-specific tests

### Test Health Metrics

Monitor these indicators:
- **Coverage**: Should maintain >90%
- **Execution Time**: Should complete in <5 seconds
- **Flakiness**: Zero flaky tests tolerated
- **Platform Parity**: All tests pass on all supported platforms

---

**Last Updated**: 2024
**Test Framework**: pytest 7.x+
**Python Versions**: 3.8+