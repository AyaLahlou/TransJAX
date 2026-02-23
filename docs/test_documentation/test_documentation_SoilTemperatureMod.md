# SoilTemperatureMod Test Suite Documentation

## Test Suite Overview

### Functions Tested
- `compute_soil_temperature`: Main function for computing soil and snow temperature evolution using thermal diffusion equations

### Test Statistics
- **Total Test Cases**: 10
- **Test Types**:
  - **Nominal Cases** (4): Standard operating conditions with realistic physical parameters
  - **Edge Cases** (4): Boundary conditions, extreme values, and numerical stability tests
  - **Special Cases** (2): Physical constraints and conservation law verification

### Coverage Areas

The test suite comprehensively covers:

1. **Physical Realism**
   - Temperature ranges (233K - 323K)
   - Thermal conductivity variations (soil minerals, water, ice, air)
   - Heat capacity calculations
   - Phase transitions (ice/water)

2. **Numerical Stability**
   - Zero and near-zero values
   - Large timesteps
   - Thin layers
   - Extreme temperature gradients

3. **Geometric Configurations**
   - Snow-free conditions
   - Multi-layer snow packs
   - Variable soil depths
   - Bedrock layers

4. **Conservation Laws**
   - Energy conservation
   - Mass conservation (implicit through water states)

5. **Boundary Conditions**
   - Surface heat flux variations
   - Deep soil temperatures
   - Snow-soil interface

## Running the Tests

### Basic Execution
```bash
# Run all tests
pytest test_SoilTemperatureMod.py

# Run with verbose output
pytest test_SoilTemperatureMod.py -v

# Run specific test
pytest test_SoilTemperatureMod.py::test_compute_soil_temperature_nominal
```

### Coverage Analysis
```bash
# Generate coverage report
pytest test_SoilTemperatureMod.py --cov=SoilTemperatureMod --cov-report=html

# View coverage in terminal
pytest test_SoilTemperatureMod.py --cov=SoilTemperatureMod --cov-report=term-missing
```

### Parallel Execution
```bash
# Run tests in parallel (requires pytest-xdist)
pytest test_SoilTemperatureMod.py -n auto
```

### Filtering Tests
```bash
# Run only edge cases
pytest test_SoilTemperatureMod.py -k "edge"

# Run only nominal cases
pytest test_SoilTemperatureMod.py -k "nominal"

# Skip slow tests (if marked)
pytest test_SoilTemperatureMod.py -m "not slow"
```

## Test Cases

### Nominal Cases (Standard Operating Conditions)

#### 1. **Typical Summer Conditions**
- **Purpose**: Verify behavior under warm, dry soil conditions
- **Characteristics**:
  - No snow layers (snl = 0)
  - Warm soil temperatures (280-300K)
  - Moderate heat flux (50 W/m²)
  - Low soil moisture
- **Expected**: Stable temperature evolution, positive thermal conductivity

#### 2. **Winter with Snow Cover**
- **Purpose**: Test multi-layer snow pack thermal dynamics
- **Characteristics**:
  - 3-5 snow layers
  - Cold temperatures (250-273K)
  - Negative heat flux (heat loss)
  - High snow water equivalent
- **Expected**: Proper snow insulation effect, phase transition handling

#### 3. **Spring Thaw Conditions**
- **Purpose**: Verify phase transition and mixed ice/water states
- **Characteristics**:
  - Temperatures near freezing (270-276K)
  - Mixed ice and liquid water
  - Variable snow cover fraction (0.3-0.7)
  - Moderate heat flux
- **Expected**: Smooth phase transitions, energy conservation

#### 4. **Deep Soil Profile**
- **Purpose**: Test thermal diffusion through multiple soil layers
- **Characteristics**:
  - 15-25 soil layers
  - Temperature gradient from surface to depth
  - Bedrock layer at depth
  - Variable soil properties by layer
- **Expected**: Realistic temperature profile, proper bedrock treatment

### Edge Cases (Boundary Conditions & Extremes)

#### 5. **Zero Heat Flux**
- **Purpose**: Test numerical stability with no forcing
- **Characteristics**:
  - gsoi = 0 W/m²
  - Uniform initial temperature
  - No snow
- **Expected**: Minimal temperature change, no numerical artifacts

#### 6. **Extreme Cold**
- **Purpose**: Verify behavior at very low temperatures
- **Characteristics**:
  - Temperatures near 233K (-40°C)
  - All water frozen
  - High thermal conductivity (ice)
- **Expected**: Stable computation, physical thermal properties

#### 7. **Thin Surface Layer**
- **Purpose**: Test numerical stability with very thin layers
- **Characteristics**:
  - dz[0] < 1e-5 m
  - Potential for numerical instability
  - Small timestep required
- **Expected**: No division by zero, stable solution

#### 8. **Saturated Soil**
- **Purpose**: Test maximum water content conditions
- **Characteristics**:
  - h2osoi_liq at saturation (watsat = 1.0)
  - Maximum thermal conductivity
  - High heat capacity
- **Expected**: Proper thermal property calculation, no overflow

### Special Cases (Physical Constraints)

#### 9. **Energy Conservation Check**
- **Purpose**: Verify energy balance closure
- **Characteristics**:
  - Known heat flux input
  - Measured temperature change
  - Calculate energy error
- **Expected**: energy_error < tolerance (typically < 1 W/m²)

#### 10. **Bedrock Layer Transition**
- **Purpose**: Test discontinuity at soil-bedrock interface
- **Characteristics**:
  - Sharp change in thermal properties at nbedrock
  - Different heat capacity and conductivity
  - Temperature continuity required
- **Expected**: Smooth temperature profile, proper property transition

## Test Data

### Data Generation Strategy

Test data is generated using a combination of:

1. **Physically-Based Defaults**
   - Physical constants from literature (water density, ice properties, etc.)
   - Typical soil properties (sand, loam, clay ranges)
   - Realistic temperature ranges for Earth's surface

2. **Parametric Variations**
   - Systematic variation of key parameters
   - Factorial design for multi-parameter interactions
   - Random sampling within physical bounds

3. **Synthetic Profiles**
   - Exponential depth profiles for temperature
   - Layered soil property variations
   - Realistic snow density profiles

### Test Data Coverage

| Parameter | Min Value | Max Value | Typical Range | Edge Cases |
|-----------|-----------|-----------|---------------|------------|
| t_soisno | 233.15 K | 323.15 K | 250-300 K | 0 K, 233 K |
| gsoi | -500 W/m² | 500 W/m² | -100 to 100 W/m² | 0, ±500 |
| dz | 1e-6 m | 10 m | 0.01-1 m | 1e-6, 10 |
| h2osoi_liq | 0 kg/m² | 1000 kg/m² | 10-500 kg/m² | 0, 1000 |
| frac_sno_eff | 0 | 1 | 0-1 | 0, 0.5, 1 |
| dtime | 60 s | 3600 s | 300-1800 s | 60, 3600 |

### Fixture Organization

```python
@pytest.fixture
def default_params():
    """Standard physical constants"""
    
@pytest.fixture
def column_geometry():
    """Typical soil column structure"""
    
@pytest.fixture
def soil_state():
    """Representative soil properties"""
    
@pytest.fixture
def water_state():
    """Typical water/ice distribution"""
```

## Expected Behavior

### Successful Test Criteria

Tests should **PASS** when:

1. **Physical Constraints Satisfied**
   - All temperatures > 0 K
   - Thermal conductivity > 0
   - Heat capacity > 0
   - Snow fraction in [0, 1]

2. **Numerical Stability**
   - No NaN or Inf values in outputs
   - Convergence within iteration limits
   - Smooth temperature profiles (no oscillations)

3. **Conservation Laws**
   - Energy error < 1 W/m² (or specified tolerance)
   - Temperature changes consistent with heat flux
   - No mass creation/destruction

4. **Boundary Conditions**
   - Surface temperature responds to gsoi
   - Deep soil temperature stable
   - Proper interface treatment at snow-soil boundary

### Expected Failures

Tests should **FAIL** when:

1. **Invalid Inputs**
   - Negative temperatures (< 0 K)
   - Negative layer thickness
   - Invalid array shapes
   - Parameters outside physical bounds

2. **Numerical Issues**
   - Non-convergence of iterative solver
   - Timestep too large for stability
   - Extreme gradients causing oscillations

3. **Conservation Violations**
   - Energy error > tolerance
   - Temperature changes inconsistent with forcing
   - Unphysical heat flux values

### Tolerance Levels

```python
TEMPERATURE_TOLERANCE = 1e-6  # K
ENERGY_TOLERANCE = 1.0        # W/m²
CONDUCTIVITY_TOLERANCE = 1e-8 # W/m/K
RELATIVE_TOLERANCE = 1e-5     # dimensionless
```

## Extending Tests

### Adding New Test Cases

1. **Create Test Data**
```python
def test_new_scenario():
    """Test description."""
    # Setup
    geom = create_custom_geometry(...)
    t_soisno = jnp.array([...])
    
    # Execute
    result, props = compute_soil_temperature(...)
    
    # Assert
    assert jnp.all(result.t_soisno > 0)
    assert jnp.abs(result.energy_error) < ENERGY_TOLERANCE
```

2. **Use Parametrization**
```python
@pytest.mark.parametrize("snow_layers,expected_insulation", [
    (0, 1.0),
    (3, 0.5),
    (5, 0.3),
])
def test_snow_insulation(snow_layers, expected_insulation):
    """Test snow insulation effect."""
    # Implementation
```

3. **Add Property-Based Tests**
```python
from hypothesis import given, strategies as st

@given(
    temperature=st.floats(min_value=233.15, max_value=323.15),
    heat_flux=st.floats(min_value=-500, max_value=500)
)
def test_temperature_evolution_properties(temperature, heat_flux):
    """Property-based test for temperature evolution."""
    # Implementation
```

### Test Organization Best Practices

1. **Group Related Tests**
   - Use test classes for related functionality
   - Organize by physical process (conduction, phase change, etc.)

2. **Use Descriptive Names**
   - `test_<function>_<scenario>_<expected_behavior>`
   - Example: `test_compute_soil_temperature_frozen_soil_stable_solution`

3. **Document Test Purpose**
   - Clear docstrings explaining what is tested
   - Reference equations or physical principles
   - Note any assumptions or limitations

4. **Maintain Test Independence**
   - Each test should run independently
   - Use fixtures for shared setup
   - Avoid test interdependencies

## Common Issues

### Issue 1: Array Shape Mismatches

**Symptom**: `ValueError: operands could not be broadcast together`

**Cause**: Inconsistent dimensions between n_cols, nlevsno, nlevgrnd

**Solution**:
```python
# Ensure consistent shapes
n_cols = 10
nlevsno = 5
nlevgrnd = 15
n_levtot = nlevsno + nlevgrnd

# All arrays must match these dimensions
t_soisno = jnp.zeros((n_cols, n_levtot))
dz = jnp.ones((n_cols, n_levtot))
```

### Issue 2: Temperature Below Absolute Zero

**Symptom**: `AssertionError: Temperatures must be positive`

**Cause**: Numerical instability or incorrect initial conditions

**Solution**:
```python
# Clip temperatures to physical range
t_soisno = jnp.clip(t_soisno, 233.15, 323.15)

# Check timestep stability
dt_max = 0.5 * dz**2 / thermal_diffusivity
assert dtime < dt_max
```

### Issue 3: Energy Conservation Violations

**Symptom**: `energy_error > tolerance`

**Cause**: Timestep too large, incorrect boundary conditions, or numerical precision

**Solution**:
```python
# Reduce timestep
dtime = 300  # seconds (5 minutes)

# Increase solver precision
solver_tolerance = 1e-8

# Check boundary flux consistency
assert jnp.isfinite(gsoi).all()
```

### Issue 4: NaN in Thermal Properties

**Symptom**: `RuntimeWarning: invalid value encountered`

**Cause**: Division by zero in thermal conductivity calculation

**Solution**:
```python
# Avoid zero denominators
dz_safe = jnp.maximum(dz, 1e-10)

# Check for zero water content
h2osoi_total = h2osoi_liq + h2osoi_ice
h2osoi_total = jnp.maximum(h2osoi_total, 1e-10)
```

### Issue 5: Slow Test Execution

**Symptom**: Tests take too long to run

**Cause**: Large array sizes or many iterations

**Solution**:
```python
# Use smaller test arrays
n_cols_test = 5  # instead of 100
nlevgrnd_test = 10  # instead of 25

# Mark slow tests
@pytest.mark.slow
def test_large_domain():
    """Test with full-size domain."""
    pass

# Skip in quick test runs
pytest test_SoilTemperatureMod.py -m "not slow"
```

### Issue 6: JAX JIT Compilation Errors

**Symptom**: `TracerArrayConversionError` or `ConcretizationTypeError`

**Cause**: Using Python control flow with JAX arrays

**Solution**:
```python
# Use JAX control flow
from jax import lax

# Instead of: if condition:
result = lax.cond(condition, true_fun, false_fun, operand)

# Instead of: for i in range(n):
result = lax.fori_loop(0, n, body_fun, init_val)
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test SoilTemperatureMod

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install jax jaxlib pytest pytest-cov
    - name: Run tests
      run: |
        pytest test_SoilTemperatureMod.py --cov=SoilTemperatureMod --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## References

### Physical Background
- Soil thermal properties: Farouki (1981)
- Snow thermal conductivity: Jordan (1991)
- Phase change modeling: Lunardini (1981)

### Numerical Methods
- Crank-Nicolson scheme for heat equation
- Tridiagonal matrix solver
- Implicit time integration

### Testing Resources
- pytest documentation: https://docs.pytest.org/
- JAX testing guide: https://jax.readthedocs.io/en/latest/notebooks/testing_cookbook.html
- Property-based testing: https://hypothesis.readthedocs.io/

---

**Last Updated**: 2024
**Test Suite Version**: 1.0
**Maintainer**: Scientific Computing Team