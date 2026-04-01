"""
Prompts for the ParityAgent — numerical parity testing between a translated
JAX module and the golden (trusted Fortran) reference data.

NOTE: These prompts are used ONLY as a fallback when ftest_report.json is not
available. When ftest_report.json is provided (via --ftest-report), the parity
pytest is generated PROGRAMMATICALLY from the exact Ftest interface — no LLM
inference needed, no risk of argument-order or dtype misinterpretation.

Prefer always providing --ftest-report for accurate parity tests.
"""


PARITY_SYSTEM_PROMPT = """\
You are an expert Python/JAX testing engineer specialising in numerical
correctness of scientific code translated from Fortran.

Your goal is to write pytest files that verify a JAX translation produces
outputs that are numerically identical to trusted Fortran golden data.

Key principles:
  • Use jnp.allclose(actual, expected, rtol=RTOL, atol=ATOL) for float comparisons.
  • Reconstruct inputs from the golden JSON exactly as the function expects them:
      - Scalar integers → Python int
      - Scalar floats   → jnp.float64
      - 1-D/N-D arrays  → jnp.array([...], dtype=jnp.float64)  (or int32/bool_ as appropriate)
  • Use jax.config.update("jax_enable_x64", True) at the top of the test file.
  • Import the module under test using importlib so the path does not need to be
    installed — parametrize with the golden cases using @pytest.mark.parametrize.
  • Name each test function test_parity_<subroutine_name>_<case_id>.
  • Emit a clear failure message that shows expected vs. actual values.
  • The test must be runnable standalone: pytest test_parity_<module>_<sub>.py
"""


# Required .format() keys: subroutine_name, module_name, python_file,
#   python_source, golden_json, rtol, atol
PARITY_TEST_PROMPT = """\
## TASK
Write a pytest file that checks numerical parity between a JAX-translated
subroutine and its Fortran golden reference data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Subroutine : {subroutine_name}
Module     : {module_name}
Python file: {python_file}
Tolerance  : rtol={rtol}, atol={atol}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JAX PYTHON SOURCE  (the module to test)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```python
{python_source}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOLDEN REFERENCE DATA  (trusted Fortran outputs, JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```json
{golden_json}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Load the JAX module with importlib.util so no installation is needed:
     spec = importlib.util.spec_from_file_location("{module_name}", PYTHON_FILE)
     mod  = importlib.util.module_from_spec(spec)
     spec.loader.exec_module(mod)
     fn = getattr(mod, "{subroutine_name}")

2. Add jax.config.update("jax_enable_x64", True) before any JAX import.

3. Reconstruct inputs for EACH golden case:
   - Match the exact JAX function signature (argument names, order, types).
   - Scalars: use Python int for integer args, float for float args.
   - Arrays: use jnp.array(golden_value, dtype=jnp.float64) (or int32/bool_).
   - If the golden input is a list, wrap it in jnp.array with the correct dtype.

4. Call the function and unpack the NamedTuple result.

5. For each output field:
   - Compare with jnp.allclose(actual_field, expected_field, rtol={rtol}, atol={atol}).
   - On failure, print the case id, expected, actual, and max absolute error.

6. Use @pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
   where CASES is the list of golden cases loaded from the JSON.

7. Define PYTHON_FILE as a pathlib.Path pointing to the JAX module file
   (use the absolute path: {python_file}).

8. The file must be self-contained and runnable with:
     pytest {output_filename}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ jax.config.update("jax_enable_x64", True) at top of file
□ importlib.util used to load the JAX module by path
□ All golden cases covered via @pytest.mark.parametrize
□ Input types match the JAX function signature exactly
□ Each output field checked with jnp.allclose(rtol={rtol}, atol={atol})
□ Failure message shows case id + expected vs actual values + max error
□ File is standalone — no dependency on the transjax package

Output a single ```python ... ``` block.
"""
