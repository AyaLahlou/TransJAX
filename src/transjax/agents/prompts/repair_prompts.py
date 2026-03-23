
REPAIR_PROMPTS = """\
You are debugging a failed JAX translation of CLM-ml Fortran code.

## ORIGINAL FORTRAN CODE
```fortran
{fortran_code}
```

## FAILED JAX TRANSLATION
```python
{failed_python_code}
```

## TEST FAILURE REPORT
```
{test_report}
```

## TASK
1. Identify the root cause of each failure
2. Compare the JAX code logic against the Fortran original line-by-line
3. Fix the JAX code while preserving the physics

## COMMON ROOT CAUSES (check these first)
1. **Array indexing**: Fortran 1-based not converted to Python 0-based
2. **Loop bounds**: off-by-one in range() vs DO loop
3. **Array mutation**: Using `array[i] = val` instead of `.at[i].set(val)`
4. **Missing float64**: Arrays defaulting to float32
5. **Division by zero**: Unprotected division where Fortran had implicit guards
6. **Intent semantics**: INTENT(INOUT) not returning modified values
7. **Column vs row major**: Array dimension ordering differs
8. **Implicit type conversion**: Fortran integer division vs Python float division

## OUTPUT FORMAT
Provide:
1. **Root Cause Analysis**: Numbered list of identified issues
2. **Fixed Code**: The corrected Python/JAX code (complete function)
3. **Explanation**: Brief note on each fix applied
"""
