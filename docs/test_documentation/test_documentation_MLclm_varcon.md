# Test Documentation: MLclm_varcon Module

## Test Suite Overview

This test suite validates the `create_empty_rsl_lookup_tables` function from the MLclm_varcon module, which initializes Roughness Sublayer (RSL) lookup tables for multilayer canopy turbulence calculations.

### Functions Tested
- `create_empty_rsl_lookup_tables`: Creates empty RSL Psihat lookup tables with correct shapes

### Test Coverage
- **Number of test cases**: 10 comprehensive scenarios
- **Test types**: 
  - Nominal cases (4): Standard configurations with typical dimensions
  - Edge cases (4): Boundary conditions and extreme values
  - Special cases (2): Default behavior and physical constraints

### Coverage Areas
1. **Shape validation**: Ensures all output arrays have correct dimensions
2. **Initialization**: Verifies arrays are properly initialized to zeros
3. **Data types**: Confirms JAX array types for GPU compatibility
4. **Named tuple structure**: Validates RSLPsihatLookupTables structure
5. **Dimension variations**: Tests small, medium, and large grid sizes
6. **Default behavior**: Tests function with default constants
7. **Physical constraints**: Validates dimension constraints (n_z, n_l ≥ 1)

---

## Running the Tests

### Basic Execution
```bash
# Run all tests in the module
pytest test_MLclm_varcon.py

# Run with verbose output
pytest test_MLclm_varcon.py -v

# Run specific test
pytest test_MLclm_varcon.py::test_create_empty_rsl_lookup_tables_nominal
```

### With Coverage Analysis
```bash
# Generate coverage report
pytest test_MLclm_varcon.py --cov=MLclm_varcon --cov-report=html

# View coverage in terminal
pytest test_MLclm_varcon.py --cov=MLclm_varcon --cov-report=term-missing
```

### Advanced Options
```bash
# Run only edge case tests
pytest test_MLclm_varcon.py -k "edge"

# Stop on first failure
pytest test_MLclm_varcon.py -x

# Show local variables on failure
pytest test_MLclm_varcon.py -l

# Parallel execution (requires pytest-xdist)
pytest test_MLclm_varcon.py -n auto
```

---

## Test Cases

### Nominal Cases (4 tests)
**Purpose**: Validate standard operational scenarios with typical grid dimensions

1. **Small grid (5×3)**: Tests minimal practical grid size
   - n_z=5, n_l=3
   - Validates basic functionality with small arrays
   - Fast execution for quick validation

2. **Medium grid (20×10)**: Tests typical simulation grid
   - n_z=20, n_l=10
   - Represents common canopy model configurations
   - Balances resolution and computational cost

3. **Large grid (50×30)**: Tests high-resolution scenarios
   - n_z=50, n_l=30
   - Validates performance with larger arrays
   - Tests memory allocation for detailed simulations

4. **Asymmetric grid (10×25)**: Tests non-square grids
   - n_z=10, n_l=25
   - Ensures no assumptions about grid aspect ratio
   - Tests stability parameter resolution vs. height resolution

### Edge Cases (4 tests)
**Purpose**: Test boundary conditions and extreme values

5. **Minimum dimensions (1×1)**: Tests smallest valid grid
   - n_z=1, n_l=1
   - Validates lower bounds on array dimensions
   - Tests degenerate case handling

6. **Very large grid (100×100)**: Tests computational limits
   - n_z=100, n_l=100
   - Validates memory allocation for large-scale simulations
   - Tests array creation performance

7. **Extreme asymmetry (100×1)**: Tests tall, narrow grid
   - n_z=100, n_l=1
   - High vertical resolution, single stability class
   - Tests edge case of stability parameter discretization

8. **Extreme asymmetry (1×100)**: Tests short, wide grid
   - n_z=1, n_l=100
   - Single height, many stability classes
   - Tests edge case of height discretization

### Special Cases (2 tests)
**Purpose**: Test default behavior and constraint validation

9. **Default constants**: Tests function with no arguments
   - Uses global ML_CANOPY_CONSTANTS
   - Validates default configuration
   - Ensures backward compatibility

10. **Invalid dimensions**: Tests constraint enforcement
    - n_z=0 or n_l=0 (invalid)
    - Should raise ValueError or handle gracefully
    - Validates input validation logic

---

## Test Data

### Generation Strategy
Test data is generated using pytest fixtures and parametrization to ensure:
- **Reproducibility**: Fixed dimension values for consistent results
- **Comprehensiveness**: Coverage of typical, boundary, and extreme cases
- **Maintainability**: Centralized test data in fixtures

### Test Data Fixtures

```python
@pytest.fixture
def sample_constants():
    """Provides MLCanopyConstants instances with various dimensions"""
    # Returns constants with different n_z and n_l values
```

### Coverage Matrix

| Dimension Type | n_z Range | n_l Range | Test Cases |
|---------------|-----------|-----------|------------|
| Minimal       | 1         | 1         | 1          |
| Small         | 5-10      | 3-10      | 2          |
| Medium        | 20        | 10-25     | 2          |
| Large         | 50-100    | 30-100    | 3          |

### Physical Realism
- All dimensions are positive integers (n_z, n_l ≥ 1)
- Grid sizes reflect realistic canopy model requirements
- Asymmetric grids test different resolution trade-offs

---

## Expected Behavior

### Successful Test Outcomes

#### Shape Validation
```python
# All arrays should have correct shapes
assert result.zdtgrid_m.shape == (n_z, 1)
assert result.dtlgrid_m.shape == (1, n_l)
assert result.psigrid_m.shape == (n_z, n_l)
# Same for _h variants
```

#### Initialization
```python
# All arrays should be initialized to zeros
assert jnp.all(result.zdtgrid_m == 0.0)
assert jnp.all(result.psigrid_m == 0.0)
```

#### Data Types
```python
# All arrays should be JAX arrays
assert isinstance(result.zdtgrid_m, jnp.ndarray)
```

### Expected Failures

1. **Invalid dimensions** (n_z=0 or n_l=0):
   - Should raise `ValueError` with descriptive message
   - Or return None/empty structure (depending on implementation)

2. **Negative dimensions**:
   - Should raise `ValueError` before array creation

3. **Non-integer dimensions**:
   - Should raise `TypeError` or convert to integer

### Performance Expectations
- Small grids (≤10×10): < 1ms
- Medium grids (≤50×50): < 10ms
- Large grids (≤100×100): < 100ms

---

## Extending Tests

### Adding New Test Cases

#### 1. Add to Parametrized Tests
```python
@pytest.mark.parametrize("n_z,n_l,description", [
    (5, 3, "small_grid"),
    (20, 10, "medium_grid"),
    # Add your new case here:
    (75, 50, "custom_grid"),
])
def test_create_empty_rsl_lookup_tables_nominal(n_z, n_l, description):
    # Test implementation
```

#### 2. Create New Test Function
```python
def test_create_empty_rsl_lookup_tables_custom_scenario():
    """Test specific scenario not covered by parametrized tests"""
    constants = MLCanopyConstants(
        # ... set all required fields
        n_z=15,
        n_l=20
    )
    result = create_empty_rsl_lookup_tables(constants)
    
    # Custom assertions
    assert result.zdtgrid_m.shape == (15, 1)
    # ... additional checks
```

#### 3. Add Property-Based Tests
```python
from hypothesis import given, strategies as st

@given(
    n_z=st.integers(min_value=1, max_value=100),
    n_l=st.integers(min_value=1, max_value=100)
)
def test_create_empty_rsl_lookup_tables_property(n_z, n_l):
    """Property-based test for arbitrary valid dimensions"""
    # Test implementation
```

### Testing New Features

When adding new fields to `RSLPsihatLookupTables`:

1. **Update shape assertions**:
   ```python
   assert result.new_field.shape == expected_shape
   ```

2. **Add initialization checks**:
   ```python
   assert jnp.all(result.new_field == expected_initial_value)
   ```

3. **Document in test docstrings**:
   ```python
   """
   Tests new_field initialization:
   - Shape: (n_z, n_l, n_new_dim)
   - Initial value: zeros
   - Type: jnp.ndarray
   """
   ```

---

## Common Issues

### Issue 1: JAX Array Comparison Failures
**Symptom**: Tests fail with "JAX arrays cannot be compared with =="

**Solution**:
```python
# ❌ Wrong
assert result.zdtgrid_m == expected

# ✅ Correct
assert jnp.allclose(result.zdtgrid_m, expected)
# or
assert jnp.array_equal(result.zdtgrid_m, expected)
```

### Issue 2: Shape Mismatch Errors
**Symptom**: AssertionError on shape checks

**Diagnosis**:
```python
print(f"Expected: {expected_shape}, Got: {result.zdtgrid_m.shape}")
```

**Common causes**:
- Incorrect dimension order (n_z vs n_l)
- Missing singleton dimensions
- Broadcasting issues

### Issue 3: Import Errors
**Symptom**: `ModuleNotFoundError: No module named 'MLclm_varcon'`

**Solution**:
```bash
# Ensure module is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/module"

# Or install in development mode
pip install -e .
```

### Issue 4: Slow Test Execution
**Symptom**: Tests take too long with large grids

**Solutions**:
1. **Use pytest markers** to skip slow tests:
   ```python
   @pytest.mark.slow
   def test_very_large_grid():
       # ...
   
   # Run without slow tests
   pytest -m "not slow"
   ```

2. **Reduce grid sizes** in parametrized tests
3. **Use hypothesis** with smaller example counts

### Issue 5: Floating Point Precision
**Symptom**: Tests fail due to small numerical differences

**Solution**:
```python
# Use appropriate tolerances
assert jnp.allclose(result, expected, rtol=1e-7, atol=1e-9)
```

### Issue 6: GPU/CPU Compatibility
**Symptom**: Tests pass on CPU but fail on GPU

**Solution**:
```python
# Ensure device-agnostic code
import jax
jax.config.update('jax_platform_name', 'cpu')  # Force CPU for testing

# Or test on both
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_on_device(device):
    with jax.default_device(jax.devices(device)[0]):
        # Test implementation
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
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest test_MLclm_varcon.py --cov=MLclm_varcon --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## Best Practices

1. **Keep tests independent**: Each test should run in isolation
2. **Use descriptive names**: Test names should explain what they test
3. **Document edge cases**: Explain why edge cases are important
4. **Test one thing**: Each test should verify one specific behavior
5. **Use fixtures**: Share common setup code via fixtures
6. **Parametrize when possible**: Reduce code duplication
7. **Assert with messages**: Provide helpful error messages
8. **Test error conditions**: Don't just test happy paths

---

## References

- [pytest documentation](https://docs.pytest.org/)
- [JAX testing guide](https://jax.readthedocs.io/en/latest/notebooks/testing.html)
- [Hypothesis for property-based testing](https://hypothesis.readthedocs.io/)