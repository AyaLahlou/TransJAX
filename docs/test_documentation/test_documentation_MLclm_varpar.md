# Test Documentation: MLclm_varpar Module

## Test Suite Overview

This test suite validates the `MLclm_varpar` module, which provides configuration parameters for multilayer canopy models used in land surface simulations. The module defines critical structural parameters that determine how canopy radiation, turbulence, and biogeochemical processes are computed across vertical layers and leaf types (sunlit vs. shaded).

### Functions Tested

1. **`get_mlcanopy_params()`**
   - Returns default MLCanopyParams configuration
   - No input parameters
   - Tests verify correct default values and namedtuple structure

2. **`validate_mlcanopy_params(params)`**
   - Validates MLCanopyParams instances for physical consistency
   - Tests cover valid configurations and various invalid states

### Test Statistics

- **Total Test Cases**: 10
- **Test Types**: 
  - Nominal cases: 10 (100%)
  - Edge cases: 0 (0%)
  - Error cases: 0 (0%)

### Coverage Areas

- ✅ Default parameter retrieval
- ✅ Parameter validation logic
- ✅ Namedtuple structure and field access
- ✅ Physical constraints enforcement
- ✅ Index convention verification (1-based Fortran style)
- ✅ Module-level constants

## Running the Tests

### Basic Execution

```bash
# Run all tests in the module
pytest test_MLclm_varpar.py

# Run with verbose output
pytest test_MLclm_varpar.py -v

# Run specific test function
pytest test_MLclm_varpar.py::test_get_mlcanopy_params_returns_correct_defaults
```

### Coverage Analysis

```bash
# Generate coverage report
pytest test_MLclm_varpar.py --cov=MLclm_varpar

# Generate HTML coverage report
pytest test_MLclm_varpar.py --cov=MLclm_varpar --cov-report=html

# Coverage with missing lines
pytest test_MLclm_varpar.py --cov=MLclm_varpar --cov-report=term-missing
```

### Continuous Integration

```bash
# Run with strict warnings
pytest test_MLclm_varpar.py -W error

# Run with markers (if defined)
pytest test_MLclm_varpar.py -m "not slow"

# Parallel execution (requires pytest-xdist)
pytest test_MLclm_varpar.py -n auto
```

## Test Cases

### Nominal Cases (10 tests)

These tests verify correct behavior under standard operating conditions:

#### 1. **Default Parameter Retrieval**
- **Test**: `test_get_mlcanopy_params_returns_correct_defaults`
- **Purpose**: Verifies `get_mlcanopy_params()` returns expected default values
- **Validates**:
  - `nlevmlcan = 100` (standard vertical resolution)
  - `nleaf = 2` (sunlit and shaded)
  - `isun = 1` (sunlit index)
  - `isha = 2` (shaded index)

#### 2. **Namedtuple Structure**
- **Test**: `test_mlcanopy_params_is_namedtuple`
- **Purpose**: Confirms MLCanopyParams is a proper namedtuple
- **Validates**: Can access fields by name and position

#### 3. **Field Immutability**
- **Test**: `test_mlcanopy_params_immutable`
- **Purpose**: Ensures parameters cannot be modified after creation
- **Validates**: Attempting field assignment raises AttributeError

#### 4. **Valid Configuration Acceptance**
- **Test**: `test_validate_mlcanopy_params_accepts_valid_config`
- **Purpose**: Confirms validation passes for correct parameters
- **Validates**: Standard configuration returns True

#### 5. **Invalid Layer Count Detection**
- **Test**: `test_validate_mlcanopy_params_rejects_invalid_nlevmlcan`
- **Purpose**: Catches non-positive layer counts
- **Validates**: `nlevmlcan ≤ 0` returns False
- **Physical Rationale**: Need at least one layer for canopy calculations

#### 6. **Invalid Leaf Type Count Detection**
- **Test**: `test_validate_mlcanopy_params_rejects_invalid_nleaf`
- **Purpose**: Enforces exactly 2 leaf types
- **Validates**: `nleaf ≠ 2` returns False
- **Physical Rationale**: Model requires sunlit/shaded distinction

#### 7. **Invalid Sunlit Index Detection**
- **Test**: `test_validate_mlcanopy_params_rejects_invalid_isun`
- **Purpose**: Enforces standard sunlit index
- **Validates**: `isun ≠ 1` returns False
- **Convention**: Fortran-style 1-based indexing

#### 8. **Invalid Shaded Index Detection**
- **Test**: `test_validate_mlcanopy_params_rejects_invalid_isha`
- **Purpose**: Enforces standard shaded index
- **Validates**: `isha ≠ 2` returns False
- **Convention**: Fortran-style 1-based indexing

#### 9. **Module Constants Verification**
- **Test**: `test_module_constants_match_defaults`
- **Purpose**: Ensures convenience constants match default values
- **Validates**: `NLEVMLCAN`, `NLEAF`, `ISUN`, `ISHA` consistency

#### 10. **Array Indexing Convention**
- **Test**: `test_index_convention_for_jax_arrays`
- **Purpose**: Documents 1-based to 0-based conversion
- **Validates**: Correct JAX array indexing (subtract 1)
- **Example**: `array[:, params.isun - 1]` for sunlit data

### Edge Cases (Not Currently Implemented)

Potential edge cases for future expansion:

- **Extreme Layer Counts**: Very large `nlevmlcan` (e.g., 1000+)
- **Boundary Values**: `nlevmlcan = 1` (minimum valid)
- **Type Validation**: Non-integer inputs
- **Negative Values**: Negative layer counts or indices

### Error Cases (Not Currently Implemented)

Potential error scenarios for future testing:

- **Type Errors**: Passing wrong types to validation
- **None Values**: Handling None/null parameters
- **Missing Fields**: Incomplete namedtuple construction

## Test Data

### Generation Strategy

Test data for this module is **deterministic and constant-based** rather than randomly generated, because:

1. **Configuration Parameters**: The module defines structural constants, not numerical computations
2. **Discrete Values**: All parameters are small integers with specific meanings
3. **Physical Constraints**: Values are tightly constrained by model design
4. **Reproducibility**: Tests must be perfectly reproducible across runs

### Data Coverage

The test suite covers:

- ✅ **Default Configuration**: Standard 100-layer, 2-leaf-type setup
- ✅ **Valid Alternatives**: Different layer counts (50, 200)
- ✅ **Invalid States**: Zero/negative layers, wrong leaf counts, incorrect indices
- ✅ **Boundary Conditions**: Minimum valid layer count (1)

### Physical Realism

All test cases respect physical constraints:

- **Layer Count**: Must be positive (need at least one layer)
- **Leaf Types**: Must be exactly 2 (sunlit and shaded)
- **Index Convention**: Must follow 1-based Fortran convention
- **Consistency**: Indices must match leaf type count

## Expected Behavior

### Should Pass ✅

1. **Default retrieval** with standard values (100, 2, 1, 2)
2. **Validation** of correctly configured parameters
3. **Alternative layer counts** (e.g., 50, 200) with standard leaf setup
4. **Immutability** enforcement (cannot modify fields)
5. **Namedtuple access** by name and position

### Should Fail ❌

1. **Zero or negative layers** (`nlevmlcan ≤ 0`)
2. **Wrong leaf count** (`nleaf ≠ 2`)
3. **Non-standard indices** (`isun ≠ 1` or `isha ≠ 2`)
4. **Field modification** attempts (AttributeError)
5. **Type mismatches** (if type checking implemented)

### Rationale

The strict validation ensures:

- **Model Consistency**: Canopy arrays are properly dimensioned
- **Physical Correctness**: Sunlit/shaded distinction is maintained
- **Code Compatibility**: Fortran-to-Python index conversion is correct
- **Numerical Stability**: Positive layer counts prevent division by zero

## Extending Tests

### Adding New Test Cases

#### 1. Add Test Function

```python
def test_new_validation_scenario():
    """Test description."""
    # Arrange
    params = MLCanopyParams(
        nlevmlcan=50,
        nleaf=2,
        isun=1,
        isha=2
    )
    
    # Act
    result = validate_mlcanopy_params(params)
    
    # Assert
    assert result is True, "Should accept 50-layer configuration"
```

#### 2. Add Parametrized Test

```python
@pytest.mark.parametrize("nlevmlcan,expected", [
    (1, True),      # Minimum valid
    (50, True),     # Alternative valid
    (100, True),    # Default
    (1000, True),   # Large valid
    (0, False),     # Invalid
    (-1, False),    # Invalid
])
def test_layer_count_validation(nlevmlcan, expected):
    """Test various layer counts."""
    params = MLCanopyParams(nlevmlcan, 2, 1, 2)
    assert validate_mlcanopy_params(params) == expected
```

#### 3. Add Fixture for Complex Setup

```python
@pytest.fixture
def custom_canopy_config():
    """Fixture for non-default configuration."""
    return MLCanopyParams(
        nlevmlcan=50,
        nleaf=2,
        isun=1,
        isha=2
    )

def test_with_custom_config(custom_canopy_config):
    """Test using custom configuration."""
    assert custom_canopy_config.nlevmlcan == 50
```

### Adding Edge Cases

```python
def test_maximum_practical_layers():
    """Test with very large layer count."""
    params = MLCanopyParams(10000, 2, 1, 2)
    assert validate_mlcanopy_params(params) is True
    # Could add performance checks here

def test_minimum_valid_layers():
    """Test with single layer."""
    params = MLCanopyParams(1, 2, 1, 2)
    assert validate_mlcanopy_params(params) is True
```

### Adding Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=1000))
def test_any_positive_layer_count_valid(nlevmlcan):
    """Property: any positive layer count should be valid."""
    params = MLCanopyParams(nlevmlcan, 2, 1, 2)
    assert validate_mlcanopy_params(params) is True
```

## Common Issues

### Issue 1: Index Convention Confusion

**Problem**: Using 1-based indices directly with JAX arrays

```python
# ❌ WRONG - will access wrong element
sunlit_data = array[:, params.isun]  # Gets index 1, not sunlit!

# ✅ CORRECT - subtract 1 for 0-based indexing
sunlit_data = array[:, params.isun - 1]  # Gets index 0 (sunlit)
```

**Solution**: Always subtract 1 when using `isun` or `isha` for array indexing

### Issue 2: Modifying Parameters

**Problem**: Attempting to change namedtuple fields

```python
# ❌ WRONG - namedtuples are immutable
params = get_mlcanopy_params()
params.nlevmlcan = 50  # Raises AttributeError
```

**Solution**: Create new instance with `_replace()` or construct new namedtuple

```python
# ✅ CORRECT
params = get_mlcanopy_params()
new_params = params._replace(nlevmlcan=50)
```

### Issue 3: Validation Not Called

**Problem**: Using invalid parameters without validation

```python
# ❌ RISKY - no validation
params = MLCanopyParams(0, 3, 1, 2)  # Invalid but not caught
# Later code may fail with cryptic errors
```

**Solution**: Always validate after construction

```python
# ✅ CORRECT
params = MLCanopyParams(0, 3, 1, 2)
if not validate_mlcanopy_params(params):
    raise ValueError("Invalid canopy parameters")
```

### Issue 4: Assuming Default Values

**Problem**: Not checking if defaults have changed

```python
# ❌ FRAGILE - hardcoded assumption
array = jnp.zeros((100, 2))  # Assumes nlevmlcan=100
```

**Solution**: Use parameter values dynamically

```python
# ✅ ROBUST
params = get_mlcanopy_params()
array = jnp.zeros((params.nlevmlcan, params.nleaf))
```

### Issue 5: Test Data Staleness

**Problem**: Tests pass but don't reflect current defaults

**Solution**: 
- Use `get_mlcanopy_params()` in tests rather than hardcoding
- Add test to verify module constants match function output
- Review tests when defaults change

### Issue 6: Missing Type Hints

**Problem**: No runtime type checking for parameters

**Solution**: Consider adding type validation in `validate_mlcanopy_params`:

```python
def validate_mlcanopy_params(params):
    """Validate with type checking."""
    if not isinstance(params, MLCanopyParams):
        return False
    if not all(isinstance(getattr(params, f), int) 
               for f in params._fields):
        return False
    # ... rest of validation
```

## Performance Considerations

### Test Execution Speed

- **Expected Runtime**: < 0.1 seconds for full suite
- **Bottlenecks**: None (simple parameter validation)
- **Optimization**: Not needed for this module

### Memory Usage

- **Per Test**: Negligible (< 1 KB)
- **Total Suite**: < 10 KB
- **Concerns**: None

## Integration Testing

### Downstream Dependencies

This module is used by:
- Radiation transfer calculations (array dimensioning)
- Turbulence models (layer iteration)
- Photosynthesis routines (sunlit/shaded separation)

### Integration Test Recommendations

```python
def test_params_compatible_with_radiation_arrays():
    """Ensure params work with radiation module."""
    params = get_mlcanopy_params()
    # Create radiation arrays with these dimensions
    rad_array = jnp.zeros((params.nlevmlcan, params.nleaf))
    assert rad_array.shape == (100, 2)

def test_index_convention_with_real_data():
    """Verify indexing works with actual leaf data."""
    params = get_mlcanopy_params()
    leaf_data = jnp.array([[1.0, 2.0]])  # [sunlit, shaded]
    
    sunlit = leaf_data[0, params.isun - 1]
    shaded = leaf_data[0, params.isha - 1]
    
    assert sunlit == 1.0
    assert shaded == 2.0
```

## Maintenance Notes

### When to Update Tests

- **Default values change**: Update expected values in tests
- **New validation rules**: Add corresponding test cases
- **New parameters added**: Extend test coverage
- **Bug fixes**: Add regression test

### Test Review Checklist

- [ ] All validation rules have corresponding tests
- [ ] Edge cases are covered
- [ ] Error messages are clear
- [ ] Tests are independent (no shared state)
- [ ] Documentation is up-to-date
- [ ] Performance is acceptable

---

**Last Updated**: 2024
**Test Framework**: pytest
**Python Version**: 3.8+
**Dependencies**: JAX, NumPy, pytest