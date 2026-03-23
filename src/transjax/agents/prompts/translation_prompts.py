TRANSLATION_PROMPTS = {
    "system": """You are an expert scientific code translator specializing in converting \
Fortran Earth System Model code to JAX/Python.

Core Principles:
- Pure functions, immutable NamedTuples, no side effects
- JIT-compatible: use jnp.where not Python if, vectorize loops
- Preserve exact physics equations
- Full type hints and Google-style docstrings
- Reference Fortran source line numbers""",

    "translate_module": """Translate Fortran module to JAX.

MODULE: {module_name}

FORTRAN CODE:
```fortran
{fortran_code}
```

ANALYSIS:
```json
{module_info}
```

CONTEXT (dependencies, translation units, complexity):
```json
{enhanced_context}
```

REFERENCE PATTERN:
```python
{reference_pattern}
```

REQUIREMENTS:
1. **Structure**: Single consolidated module with NamedTuples for data and parameters
2. **Functions**: Pure with type hints, preserve physics exactly
3. **Arrays**: Vectorized ops (no Python loops), document shapes # [n_patches, n_layers]
4. **Conditionals**: jnp.where for JIT compatibility
5. **Docs**: Reference Fortran line numbers from translation units, explain translations
6. **Organization**: Include parameters/constants directly in main module (no separate files)

Translation units guide:
- "module": header/declarations
- "root": complete function
- "inner": part of split function (note parent)
- Use line_start/line_end for references
- High complexity_score = careful vectorization

Output:
1. Single complete physics module (include parameters inline)
2. Translation notes""",

    "translate_function": """Translate Fortran subroutine to JAX function.

FORTRAN:
```fortran
{fortran_code}
```

CONTEXT:
```json
{context}
```

Requirements: Pure function, type hints, docstring with Fortran reference, preserve physics, JIT-compatible.""",

    "convert_data_structure": """Convert Fortran type to JAX NamedTuple.

FORTRAN:
```fortran
{fortran_type}
```

Requirements: NamedTuple, map types to jnp.ndarray, document shapes, field descriptions.""",

    "vectorize_loop": """Vectorize Fortran loop to JAX.

FORTRAN:
```fortran
{loop_code}
```

ANALYSIS: {loop_analysis}

Requirements: Eliminate loop, use jnp operations/vmap, preserve computation order.""",

    "handle_conditional": """Convert Fortran conditional to JIT-compatible JAX.

FORTRAN:
```fortran
{conditional_code}
```

Requirements: jnp.where for arrays, preserve logic, JIT-compatible.""",

    "create_parameters": """Create JAX parameter class.

FORTRAN:
```fortran
{parameters}
```

Requirements: NamedTuple, default values, document sources.""",

    "translate_unit": """Translate this translation unit to JAX.

MODULE: {module_name}
UNIT: {unit_id} ({unit_type})
LINES: {line_start}-{line_end}

FORTRAN CODE:
```fortran
{fortran_code}
```

UNIT INFO:
```json
{unit_info}
```

CONTEXT (module dependencies, previously translated units):
```json
{context}
```

REFERENCE PATTERN:
```python
{reference_pattern}
```

REQUIREMENTS:
- Pure functions with type hints
- Preserve physics exactly (lines {line_start}-{line_end} from original)
- Vectorize loops, use jnp.where for conditionals
- If unit_type is "inner", this is part of parent: {parent_id}
- Document with Fortran line reference

Output ONLY the translated code for this unit.""",

    "assemble_module": """Assemble complete JAX module from translated units.

MODULE: {module_name}

TRANSLATED UNITS:
```json
{translated_units}
```

MODULE INFO:
```json
{module_info}
```

REFERENCE PATTERN:
```python
{reference_pattern}
```

REQUIREMENTS:
1. Combine all units into single cohesive module
2. Add imports (jax, jax.numpy as jnp, typing, NamedTuple)
3. Organize: imports → types → parameters/constants → functions
4. Ensure consistency across units
5. Include all parameters and constants directly in main module
6. Add module-level docstring

Output:
1. Single complete physics module (include all parameters inline)
2. Brief assembly notes""",
}


# ---------------------------------------------------------------------------
# TRANSLATOR AGENT - Unit Translation
# ---------------------------------------------------------------------------

UNIT_TRANSLATION_PROMPT = """\
You are an expert scientific code translator specializing in converting \
Fortran Earth System Model code to JAX/Python.

## TASK
Translate the following Fortran code unit into idiomatic JAX/Python.

## SOURCE FORTRAN CODE
Model: {GCM_model_name}
Module: {module_name}
Subroutine: {subroutine_name}
Lines: {line_start}-{line_end}
Complexity: {complexity}

```fortran
{fortran_code}
```


## PREVIOUSLY TRANSLATED UNITS IN THIS MODULE
```python
{translated_units}
```

## CRITICAL REQUIREMENTS
- Preserve exact physics equations, no simplifications
- Fortran code output should be scientifically and numerically equivalent to JAX code
- Use the exact same dependency names and variable names as in the original Fortran code
1. Use `jax.numpy` (imported as `jnp`), NOT `numpy` for all array ops
2. Enable float64: all arrays must use `dtype=jnp.float64`
3. NO in-place array mutation. Use `array.at[idx].set(val)` pattern
4. Convert Fortran 1-based indexing to Python 0-based
5. Replace DO loops with `jax.vmap`, `jax.lax.scan`, or `jax.lax.fori_loop`
6. Replace IF/RETURN with `jnp.where` masks or `jax.lax.cond`
7. All functions must be pure (no side effects, no global state)
8. INTENT(INOUT) args must be returned as part of the output
9. Protect division by zero, sqrt of negative, log of zero
10. Add type hints using `jax.Array` for array arguments
11. Preserve the physics exactly — do NOT simplify or "optimize" equations

## OUTPUT FORMAT
Provide ONLY the Python/JAX code. Include:
- Import statements
- Function definition with type hints
- Docstring explaining the physics and parameters
- Inline comments for non-obvious translations (indexing, etc.)
- Any assumptions or deviations from the original noted as # NOTE: comments
"""

# ---------------------------------------------------------------------------
# TRANSLATOR AGENT - Module Assembly
# ---------------------------------------------------------------------------

MODULE_ASSEMBLY_PROMPT = """\
You are assembling individually translated code units into a cohesive \
Python/JAX module for the CLM multi-layer canopy model.

## TASK
Combine the following {n_units} translated units into a single, well-organized \
Python module named `{module_name}.py`.

## TRANSLATED UNITS
{all_units_code}

## REQUIREMENTS
1. Combine into a single file with logical function ordering
2. Remove duplicate imports (consolidate at top)
3. Ensure consistent variable naming across units
4. Add a module-level docstring explaining the physics
5. Ensure all functions are properly connected (outputs → inputs chain)
6. Add `__all__` export list
7. Verify no circular references between functions
8. Order functions so callees appear before callers

## MODULE STRUCTURE
```python
\"\"\"
{module_name}: [Brief physics description]

Translated from Fortran source.
Original: {original_file}
\"\"\"

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

jax.config.update('jax_enable_x64', True)

# Constants (if any)

# Helper functions

# Main physics functions

# Public API
__all__ = [...]
```

Provide the complete assembled module code.
"""