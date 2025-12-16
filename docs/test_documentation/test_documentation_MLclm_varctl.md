# Test Documentation: MLclm_varctl Module

## Test Suite Overview

This test suite provides comprehensive coverage for the `MLclm_varctl` module, which manages configuration parameters for the Multi-Layer Canopy (MLCanopy) model used in CLM (Community Land Model) simulations.

### Functions Tested

The test suite covers all 9 functions in the module:

1. **Configuration Factories** (3 functions):
   - `create_default_config()` - Creates default configuration
   - `create_clm5_config()` - Creates CLM5-specific configuration
   - `create_clm45_config()` - Creates CLM4.5-specific configuration

2. **Validation & Queries** (4 functions):
   - `validate_config()` - Validates configuration parameters
   - `is_clm5_physics()` - Checks if using CLM5 physics
   - `uses_auto_layers()` - Checks if using automatic layer determination
   - `uses_rsl_turbulence()` - Checks if using RSL turbulence parameterization

3. **Utility Functions** (2 functions):
   - `get_canopy_dz()` - Returns appropriate height increment based on canopy height
   - `config_summary()` - Generates human-readable configuration summary

### Test Statistics

- **Total Test Cases**: 10 parametrized test scenarios
- **Test Types**: 
  - Nominal cases (typical usage)
  - Edge cases (boundary conditions)
  - Special cases (physics-specific configurations)
- **Configuration Variants**: 23 unique field combinations tested
- **Total Assertions**: 100+ individual checks

### Coverage Areas

The test suite ensures coverage of:

- ✅ Default configuration creation and validation
- ✅ CLM version-specific configurations (CLM4.5 vs CLM5.0)
- ✅ All enumerated parameter options (gs_type, turb_type, etc.)
- ✅ Boundary conditions (minimum/maximum values)
- ✅ Physical constraints (positive heights, valid fractions)
- ✅ Auto-determination vs manual layer specification
- ✅ Canopy height thresholds and layer spacing
- ✅ Configuration consistency checks
- ✅ String formatting and summary generation

---

## Running the Tests

### Basic Execution

```bash
# Run all tests
pytest test_MLclm_varctl.py

# Run with verbose output
pytest test_MLclm_varctl.py -v

# Run specific test function
pytest test_MLclm_varctl.py::test_create_default_config

# Run tests matching pattern
pytest test_MLclm_varctl.py -k "clm5"
```

### Coverage Analysis

```bash
# Generate coverage report
pytest test_MLclm_varctl.py --cov=MLclm_varctl

# Generate HTML coverage report
pytest test_MLclm_varctl.py --cov=MLclm_varctl --cov-report=html

# Show missing lines
pytest test_MLclm_varctl.py --cov=MLclm_varctl --cov-report=term-missing
```

### Advanced Options

```bash
# Run with detailed output and stop on first failure
pytest test_MLclm_varctl.py -vv -x

# Run only failed tests from last run
pytest test_MLclm_varctl.py --lf

# Run tests in parallel (requires pytest-xdist)
pytest test_MLclm_varctl.py -n auto

# Generate JUnit XML report for CI/CD
pytest test_MLclm_varctl.py --junitxml=test-results.xml
```

---

## Test Cases

### Nominal Cases

**Purpose**: Verify correct behavior under typical usage scenarios

1. **Default Configuration**
   - Tests `create_default_config()` returns valid configuration
   - Verifies all default values match specification
   - Checks configuration passes validation

2. **CLM5 Configuration**
   - Tests `create_clm5_config()` sets CLM5-specific parameters
   - Verifies `clm_phys='CLM5_0'`, `fpi_type=2`, `root_type=2`
   - Confirms physics detection works correctly

3. **CLM4.5 Configuration**
   - Tests `create_clm45_config()` sets CLM4.5-specific parameters
   - Verifies `clm_phys='CLM4_5'`, `fpi_type=1`, `root_type=1`
   - Confirms backward compatibility

4. **Canopy Height Calculations**
   - Tests `get_canopy_dz()` with various canopy heights
   - Verifies correct dz selection (tall vs short)
   - Checks threshold behavior at `dz_param=2.0m`

### Edge Cases

**Purpose**: Test boundary conditions and extreme values

5. **Minimum Values**
   - Tests smallest valid positive values for height parameters
   - Verifies `dz_tall=0.01m`, `dz_short=0.01m`, `dpai_min=0.001`
   - Checks zero layer counts (`nlayer_above=0`, `nlayer_within=0`)

6. **Maximum Values**
   - Tests large but realistic canopy heights (100m)
   - Verifies large layer counts (100 layers)
   - Checks large timesteps (3600s)

7. **Boundary Thresholds**
   - Tests canopy height exactly at `dz_param` threshold
   - Verifies correct dz selection at boundary
   - Tests heights just above/below threshold

8. **Special Numeric Values**
   - Tests sentinel values (`kn_val=-999.0`, `fracdir=-999.0`)
   - Verifies `ml_vert_init=-9999` (uninitialized state)
   - Checks auto-determination flags

### Special Cases

**Purpose**: Test physics-specific and configuration-dependent behavior

9. **RSL Turbulence Configuration**
   - Tests `turb_type=1` (Harman & Finnigan RSL)
   - Verifies `uses_rsl_turbulence()` detection
   - Checks RSL file path handling

10. **Auto Layer Determination**
    - Tests `nlayer_above=0` and `nlayer_within=0`
    - Verifies `uses_auto_layers()` returns True
    - Checks manual layer specification (>0) returns False

### Invalid Cases (Validation Tests)

**Purpose**: Ensure proper error handling for invalid configurations

- Invalid `clm_phys` values (not 'CLM4_5' or 'CLM5_0')
- Out-of-range enumerated parameters (e.g., `gs_type=5`)
- Negative or zero values for positive-only parameters
- Inconsistent physics combinations

---

## Test Data

### Generation Strategy

Test data is generated using pytest fixtures and parametrization to ensure:

1. **Systematic Coverage**: All valid parameter combinations are tested
2. **Reproducibility**: Fixed test data ensures consistent results
3. **Maintainability**: Centralized fixtures make updates easy

### Data Categories

#### Configuration Fixtures

```python
@pytest.fixture
def default_config():
    """Provides default configuration for testing"""
    return create_default_config()

@pytest.fixture
def clm5_config():
    """Provides CLM5-specific configuration"""
    return create_clm5_config()
```

#### Parametrized Test Data

```python
@pytest.mark.parametrize("canopy_height,expected_dz", [
    (0.5, 0.1),    # Short canopy
    (2.0, 0.1),    # At threshold
    (5.0, 0.5),    # Tall canopy
    (100.0, 0.5),  # Very tall canopy
])
```

### Physical Realism

Test data respects physical constraints:

- **Heights**: Always positive (>0m)
- **Fractions**: In range [0, 1] or sentinel values (<0)
- **Temperatures**: Above absolute zero (when applicable)
- **Layer counts**: Non-negative integers
- **Timesteps**: Positive values (>0s)

### Coverage Matrix

| Parameter | Tested Values | Coverage |
|-----------|---------------|----------|
| `clm_phys` | 'CLM4_5', 'CLM5_0' | 100% |
| `gs_type` | 0, 1, 2 | 100% |
| `turb_type` | -1, 0, 1 | 100% |
| `gb_type` | 0, 1, 2, 3 | 100% |
| `canopy_height` | 0.1-100m | Representative range |
| `nlayer_*` | 0, 1, 10, 100 | Auto + manual modes |

---

## Expected Behavior

### Passing Tests

Tests should **PASS** when:

✅ Default configurations are created with correct values  
✅ CLM version-specific configurations set appropriate parameters  
✅ Validation accepts all valid parameter combinations  
✅ Query functions correctly identify configuration states  
✅ `get_canopy_dz()` returns correct dz based on height threshold  
✅ Configuration summaries generate valid strings  
✅ All constraints are satisfied (positive values, valid enums)

### Failing Tests

Tests should **FAIL** when:

❌ Invalid `clm_phys` values are used  
❌ Enumerated parameters have out-of-range values  
❌ Required positive parameters are zero or negative  
❌ Configuration validation detects inconsistencies  
❌ Type mismatches occur (e.g., string instead of int)

### Example Validation Failures

```python
# Should raise ValueError
config = MLCanopyConfig(
    clm_phys='INVALID',  # Not in ['CLM4_5', 'CLM5_0']
    gs_type=5,           # Not in [0, 1, 2]
    dz_tall=-1.0,        # Must be positive
)
validate_config(config)  # Raises ValueError
```

---

## Extending Tests

### Adding New Test Cases

1. **Add to Parametrization**:

```python
@pytest.mark.parametrize("config_field,test_value,should_pass", [
    ("gs_type", 0, True),
    ("gs_type", 1, True),
    ("gs_type", 2, True),
    ("gs_type", 3, False),  # New invalid case
])
def test_validation(config_field, test_value, should_pass):
    config = create_default_config()._replace(**{config_field: test_value})
    if should_pass:
        assert validate_config(config)
    else:
        with pytest.raises(ValueError):
            validate_config(config)
```

2. **Add New Fixture**:

```python
@pytest.fixture
def custom_config():
    """Configuration for specific test scenario"""
    return MLCanopyConfig(
        clm_phys='CLM5_0',
        gs_type=0,  # Medlyn
        turb_type=1,  # RSL
        # ... other parameters
    )
```

3. **Add Integration Test**:

```python
def test_clm5_with_rsl_turbulence():
    """Test CLM5 physics with RSL turbulence parameterization"""
    config = create_clm5_config()._replace(turb_type=1)
    
    assert is_clm5_physics(config)
    assert uses_rsl_turbulence(config)
    assert validate_config(config)
```

### Testing New Functions

When adding new functions to the module:

1. Create test function with descriptive name
2. Add parametrized test cases covering:
   - Nominal behavior
   - Edge cases
   - Error conditions
3. Document expected behavior in docstring
4. Update this documentation

### Property-Based Testing

For complex validation logic, consider hypothesis:

```python
from hypothesis import given, strategies as st

@given(
    height=st.floats(min_value=0.01, max_value=100.0),
    dz_param=st.floats(min_value=0.1, max_value=10.0)
)
def test_canopy_dz_properties(height, dz_param):
    """Property: dz selection is consistent with threshold"""
    config = create_default_config()._replace(dz_param=dz_param)
    dz = get_canopy_dz(config, height)
    
    if height > dz_param:
        assert dz == config.dz_tall
    else:
        assert dz == config.dz_short
```

---

## Common Issues

### Issue 1: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'MLclm_varctl'`

**Solution**:
```bash
# Ensure module is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/module"

# Or install in development mode
pip install -e .
```

### Issue 2: Validation Failures

**Problem**: Tests fail with `ValueError` during validation

**Solution**:
- Check that test data respects all constraints
- Verify enumerated values are in allowed sets
- Ensure positive-only parameters are >0
- Review recent changes to validation logic

### Issue 3: Floating Point Comparisons

**Problem**: Tests fail due to floating point precision

**Solution**:
```python
# Use pytest.approx for float comparisons
assert result == pytest.approx(expected, rel=1e-6)

# Or numpy testing utilities
import numpy.testing as npt
npt.assert_allclose(result, expected, rtol=1e-6)
```

### Issue 4: Fixture Scope Issues

**Problem**: Fixtures are modified between tests causing failures

**Solution**:
```python
# Use function scope (default) for mutable fixtures
@pytest.fixture(scope="function")
def config():
    return create_default_config()

# Or create new instance each time
@pytest.fixture
def config():
    return create_default_config()  # Returns new instance
```

### Issue 5: Parametrization ID Conflicts

**Problem**: Unclear test names in output

**Solution**:
```python
@pytest.mark.parametrize("height,expected", [
    pytest.param(0.5, 0.1, id="short_canopy"),
    pytest.param(5.0, 0.5, id="tall_canopy"),
])
```

### Issue 6: Missing Test Dependencies

**Problem**: Tests fail due to missing optional dependencies

**Solution**:
```python
# Skip tests if dependency unavailable
pytest.importorskip("hypothesis")

# Or mark as optional
@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
```

---

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
      - run: pip install pytest pytest-cov
      - run: pytest test_MLclm_varctl.py --cov=MLclm_varctl --cov-report=xml
      - uses: codecov/codecov-action@v2
```

---

## Best Practices

1. **Run tests before committing**: `pytest test_MLclm_varctl.py`
2. **Maintain >90% coverage**: Check with `--cov-report=term-missing`
3. **Keep tests fast**: Each test should complete in <1s
4. **Use descriptive names**: Test names should explain what they test
5. **Document edge cases**: Add comments explaining non-obvious test cases
6. **Update tests with code**: When changing module, update tests first (TDD)

---

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest parametrize](https://docs.pytest.org/en/stable/parametrize.html)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Python unittest best practices](https://docs.python.org/3/library/unittest.html)

---

**Last Updated**: 2024  
**Test Suite Version**: 1.0  
**Module Version**: MLclm_varctl (CLM Multi-Layer Canopy)