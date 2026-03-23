
# ---------------------------------------------------------------------------
# TESTING AGENT
# ---------------------------------------------------------------------------

TEST_GENERATION_PROMPT = """\
You are generating validation tests for a JAX translation of a Fortran Earth system model code.

## TRANSLATED JAX MODULE
Module: {module_name}
```python
{python_code}
```

## ORIGINAL FORTRAN 
```fortran
{fortran_signatures}
```

## TASK
Generate a comprehensive pytest test suite that validates:

1. **Smoke tests**: Functions execute without errors for typical inputs
2. **Shape tests**: Output arrays have correct dimensions
3. **Physical bounds**: Outputs are within physically realistic ranges
4. **Edge cases**: Boundary conditions (zero LAI, night, frozen soil, etc.)
5. **Parity stubs**: Placeholder tests that will compare against Fortran golden I/O

## PHYSICAL BOUNDS FOR CLM-ml VARIABLES
- Temperature: 200-350 K
- Specific humidity: 0-0.04 kg/kg
- Wind speed: 0-50 m/s
- Solar radiation: 0-1400 W/m2
- Stomatal conductance: 0-2 mol/m2/s
- Photosynthesis rate: -5 to 50 umol/m2/s
- Sensible heat flux: -200 to 800 W/m2
- Latent heat flux: -100 to 600 W/m2
- Leaf temperature: 250-330 K

## OUTPUT FORMAT
A complete pytest file with:
- Import statements
- Fixture functions for test data
- Test classes organized by function
- Parameterized tests for multiple regimes
- Clear docstrings explaining what each test validates
"""
