"""
Simplified Test Prompts for Python/JAX test generation only.

These prompts focus on generating comprehensive tests for Python/JAX code
without requiring Fortran validation.

JAX Differentiability Rules enforced in generated tests:
- All loops         : jax.lax.fori_loop (never Python for/while)
- Python continue   : lax.cond returning unchanged carry
- Python if/elif/else: nested lax.cond
- Nested col/level iteration: flatten and vectorize
- All array ops     : jnp only (no numpy in translated code under test)
- Immutable updates : .at[].set()
- lax.while_loop    : replace with lax.scan (fixed bound) for full gradient support
- Parcel tracking   : fixed-iteration lax.scan with lax.cond early-stop flag
"""

# ---------------------------------------------------------------------------
# Few-shot examples for in-context learning
# ---------------------------------------------------------------------------

_FEW_SHOT_TEST_IMPORT = """
# FEW-SHOT: correct import pattern
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ctsm_jax.my_module import my_function   # real function, NOT a mock
"""

_FEW_SHOT_TEST_PARAMETRIZE = """
# FEW-SHOT: parametrized test with clear error messages
import pytest
import numpy as np
import jax.numpy as jnp

@pytest.mark.parametrize("case", [
    {"name": "nominal_summer",  "temp_k": 300.0,  "pres_pa": 101325.0, "expected_phase": 1},
    {"name": "boundary_freeze", "temp_k": 273.15, "pres_pa": 101325.0, "expected_phase": 2},
    {"name": "edge_very_cold",  "temp_k": 200.0,  "pres_pa":  50000.0, "expected_phase": 0},
])
def test_phase_selection_values(case):
    \"\"\"Test that phase_selection returns correct phase for various inputs.\"\"\"
    result = phase_selection(
        temp_k=jnp.array(case["temp_k"]),
        pres_pa=jnp.array(case["pres_pa"]),
    )
    assert int(result) == case["expected_phase"], (
        f"[{case['name']}] Expected phase {case['expected_phase']}, got {int(result)}. "
        f"Inputs: temp_k={case['temp_k']}, pres_pa={case['pres_pa']}"
    )
"""

_FEW_SHOT_TEST_SHAPES = """
# FEW-SHOT: shape and dtype test
def test_compute_flux_shapes():
    \"\"\"Verify output shapes match [n_cols, n_levels] for flux computation.\"\"\"
    n_cols, n_levels = 4, 10
    temp = jnp.ones((n_cols, n_levels)) * 280.0    # K
    pres = jnp.ones((n_cols, n_levels)) * 101325.0  # Pa
    result = compute_flux(temp, pres)
    assert result.shape == (n_cols, n_levels), (
        f"Expected shape ({n_cols}, {n_levels}), got {result.shape}"
    )
    assert result.dtype == jnp.float32, f"Expected float32, got {result.dtype}"
"""

_FEW_SHOT_TEST_GRADIENT = """
# FEW-SHOT: differentiability test
import jax

def test_compute_flux_differentiable():
    \"\"\"Verify end-to-end differentiability (jax.grad must not raise).\"\"\"
    def scalar_loss(temp_scalar):
        temp = jnp.full((2, 5), temp_scalar)
        pres = jnp.ones((2, 5)) * 101325.0
        return jnp.sum(compute_flux(temp, pres))
    grad = jax.grad(scalar_loss)(280.0)
    assert jnp.isfinite(grad), f"Gradient is not finite: {grad}"
"""

_ALL_FEW_SHOTS = (
    _FEW_SHOT_TEST_IMPORT
    + _FEW_SHOT_TEST_PARAMETRIZE
    + _FEW_SHOT_TEST_SHAPES
    + _FEW_SHOT_TEST_GRADIENT
)

# ---------------------------------------------------------------------------
# Shared JAX rules block injected into every prompt
# ---------------------------------------------------------------------------
_JAX_RULES = """
JAX DIFFERENTIABILITY RULES - the code under test must obey these.
Tests should verify them where relevant:
1. LOOPS        : jax.lax.fori_loop only. No Python for/while.
2. CONTINUE     : lax.cond returning unchanged carry. No Python continue.
3. IF/ELIF/ELSE : nested lax.cond. No Python if-else in jitted code.
4. NESTED ITER  : jnp.vectorize or broadcasting. No nested Python loops.
5. ARRAY OPS    : jnp only. No numpy (np.*) in translated module.
6. MUTATION     : .at[].set(). No in-place ops.
7. WHILE->SCAN  : lax.scan with fixed bound + lax.cond stop flag.
8. PARCEL TRACK : fixed-iteration lax.scan with lax.cond done flag.
"""


TEST_PROMPTS = {
    "system": f"""You are an expert Python/JAX Testing Agent specializing in scientific computing.

Your expertise includes:
- Python, JAX, and NumPy testing best practices
- pytest framework and fixtures
- Parametrized testing and property-based testing
- Scientific computing edge cases and numerical stability
- Test data generation for physics simulations

Your responsibilities:
1. Analyze Python/JAX function signatures
2. Generate comprehensive test data covering:
   - Nominal/typical cases
   - Edge cases (zeros, negatives, boundaries, NaN/Inf)
   - Array dimension variations
   - Physical realism (temps > 0K, fractions in [0,1])
3. Create pytest files with:
   - Fixtures for test data
   - Parametrized tests
   - Clear assertions with good error messages
   - Docstrings explaining test purpose
4. Include a differentiability test (jax.grad must not raise) for every function
5. Generate test documentation

{_JAX_RULES}

You follow pytest best practices and write clear, maintainable tests.

IN-CONTEXT EXAMPLES:
{_ALL_FEW_SHOTS}""",

    "analyze_python_signature": """Analyze the following Python/JAX function to extract its complete signature.

Module: {module_name}

Python Code:
```python
{python_code}
```

Extract:
1. Function name
2. All parameters with: name, type hint, array shape, default value,
   description from docstring, physical constraints (e.g. temp > 0, fraction in [0,1])
3. Return type and structure
4. Any NamedTuple or dataclass definitions used
5. Whether the function is JIT-compiled (@jax.jit)

Return as JSON (no extra text, no markdown fences outside the block):
```json
{
  "name": "function_name",
  "jit_compiled": true,
  "parameters": [
    {
      "name": "param_name",
      "python_type": "jnp.ndarray",
      "shape": "(n_columns, n_levels)",
      "default": null,
      "description": "parameter description",
      "constraints": {"min": 0, "max": 1}
    }
  ],
  "returns": {
    "type": "jnp.ndarray",
    "description": "return value description",
    "components": ["field1", "field2"]
  },
  "namedtuples": [
    {
      "name": "ResultType",
      "fields": ["field1", "field2"]
    }
  ]
}
```""",

    "generate_test_data": f"""Generate comprehensive synthetic test data for a Python/JAX function.

Python Signature:
```json
{{python_signature}}
```

Number of test cases: {{num_cases}}
Include edge cases: {{include_edge_cases}}

{_JAX_RULES}

Generate {{num_cases}} diverse test cases:

Test types:
1. Nominal   (50%): typical operating conditions, realistic physical values
2. Edge      (30%, only if include_edge_cases is true):
   - zero or near-zero values
   - negative values (where physically valid)
   - boundary conditions (e.g. temp = 273.15 K, fraction = 0.0 or 1.0)
   - very small or large magnitudes
3. Special   (20%): different array sizes or extreme but valid physical conditions

Requirements:
- Physically realistic (temps in Kelvin > 0, fractions in [0,1])
- Consistent dimensions within every test case
- Descriptive metadata per case

Return as JSON (no extra text):
```json
{{{{
  "function_name": "function_name",
  "test_cases": [
    {{{{
      "name": "test_nominal_summer_conditions",
      "inputs": {{{{
        "param1": 300.0,
        "param2": [[1.0, 2.0], [3.0, 4.0]]
      }}}},
      "metadata": {{{{
        "type": "nominal",
        "description": "Typical summer atmospheric column",
        "edge_cases": []
      }}}}
    }}}}
  ],
  "notes": "How data was generated and assumptions made"
}}}}
```

Use nested Python lists for array inputs. Ensure dimensions are consistent within each case.""",

    "generate_pytest": f"""Generate a comprehensive pytest file for the Python/JAX function.

Module: {{module_name}}
Source Directory: {{source_directory}}

Python Signature:
```json
{{python_signature}}
```

Test Data:
```json
{{test_data}}
```

Include Performance Tests: {{include_performance}}

{_JAX_RULES}

IN-CONTEXT EXAMPLES:
{_ALL_FEW_SHOTS}

Create a pytest file with the following sections in order:

1. Imports
   - Add src to sys.path first:
     sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
   - Import the REAL translated function from {{source_directory}}.{{module_name}}
   - Do NOT create mock functions or stubs
   - Import pytest, jax, jax.numpy as jnp, numpy as np

2. Fixtures
   - test_data() fixture returning a list of test-case dicts
   - Any shared setup (e.g. default NamedTuple params)

3. Parametrized Tests - use @pytest.mark.parametrize with descriptive IDs

4. Test Functions - implement ALL of:
   - test_<function>_shapes()       : assert output.shape == expected_shape
   - test_<function>_values()       : assert np.allclose(result, expected, atol=1e-6, rtol=1e-6)
   - test_<function>_edge_cases()   : test boundary/zero/extreme inputs
   - test_<function>_dtypes()       : assert result.dtype == jnp.float32 (or expected dtype)
   - test_<function>_differentiable(): call jax.grad on a scalar loss and assert finite gradient
   - test_<function>_performance()  : only if include_performance is true

5. Assertions
   - np.allclose for float arrays (atol=1e-6, rtol=1e-6)
   - Descriptive f-string message in every assert

6. Docstrings - one-sentence docstring per test function

Return ONLY the complete Python pytest source code.""",

    "generate_documentation": """Generate test documentation for the module.

Module: {module_name}

Python Signature:
```json
{python_signature}
```

Test Data Summary:
```json
{test_data_summary}
```

Create a markdown documentation file with these sections:

## Test Suite Overview
- Functions under test
- Number and types of test cases (nominal / edge / special)
- Coverage areas (shapes, values, edge cases, dtypes, differentiability, performance)

## Running the Tests
```bash
pytest test_{module_name}.py
pytest test_{module_name}.py --cov={module_name}
pytest test_{module_name}.py -v
pytest test_{module_name}.py -k "differentiable"
```

## Test Cases
Brief description of each type: nominal, edge, special, differentiability.

## Test Data
How data was generated, physical assumptions, and array shapes used.

## Expected Behavior
Which tests should pass and under what conditions a test might fail.

## Extending Tests
How to add new test cases or test functions.

## Common Issues
Potential problems (dtype mismatches, shape errors, non-finite gradients) and solutions.

Return complete markdown documentation.""",
}
