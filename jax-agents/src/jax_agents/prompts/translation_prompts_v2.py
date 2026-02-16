"""Condensed prompt templates for Translator Agent.

JAX Differentiability Rules (STRICT - override any conflicting existing rules):
- All loops         : use jax.lax.fori_loop (upward/downward parcel, smoothing)
- Python continue   : replace with lax.cond returning unchanged carry
- Python if/elif/else: replace with nested lax.cond
- Nested col/level iteration: flatten and vectorize
- All array ops     : jnp only (no numpy)
- Immutable updates : .at[].set()
- lax.while_loop    : replace with lax.scan (fixed bound) for full gradient support
- Parcel tracking   : fixed-iteration lax.scan with lax.cond early-stop flag
"""

# ---------------------------------------------------------------------------
# Few-shot examples for in-context learning
# ---------------------------------------------------------------------------

_FEW_SHOT_LOOP = """
# FEW-SHOT: loops
# WRONG - Python loop, not differentiable
result = jnp.zeros(n)
for i in range(n):
    result = result.at[i].set(result[i - 1] + delta[i])

# CORRECT - lax.fori_loop
def body(i, carry):
    return carry.at[i].set(carry[i - 1] + delta[i])
result = jax.lax.fori_loop(0, n, body, jnp.zeros(n))
"""

_FEW_SHOT_COND = """
# FEW-SHOT: if/elif/else
# WRONG - Python if-else, not JIT-compatible
if temp > 273.15:
    phase = 1
elif temp > 200.0:
    phase = 2
else:
    phase = 0

# CORRECT - nested lax.cond
phase = jax.lax.cond(
    temp > 273.15,
    lambda _: 1,
    lambda _: jax.lax.cond(temp > 200.0, lambda _: 2, lambda _: 0, None),
    operand=None,
)
"""

_FEW_SHOT_CONTINUE = """
# FEW-SHOT: Python continue inside a loop
# WRONG - Python for loop with continue, not JIT-compatible
for i in range(n):
    if mask[i]:
        continue
    result = result.at[i].set(compute(i))

# CORRECT - lax.fori_loop with lax.cond replacing continue
def body(i, result):
    return jax.lax.cond(
        mask[i],
        lambda _: result,                          # masked path: skip this index
        lambda _: result.at[i].set(compute(i)),
        operand=None,
    )
result = jax.lax.fori_loop(0, n, body, jnp.zeros(n))
"""

_FEW_SHOT_SCAN = """
# FEW-SHOT: while_loop -> scan, and parcel tracking
# WRONG - lax.while_loop has limited gradient support
state = jax.lax.while_loop(cond_fn, body_fn, init_state)

# CORRECT - lax.scan with fixed bound and lax.cond early-stop
MAX_ITER = 500
def scan_body(carry, _):
    state, done = carry
    new_state = body_fn(state)
    done = done | (~cond_fn(new_state))
    next_state = jax.lax.cond(done, lambda _: state, lambda _: new_state, None)
    return (next_state, done), None
(final_state, _), _ = jax.lax.scan(scan_body, (init_state, False), None, length=MAX_ITER)

# Parcel tracking with fixed-iteration scan
def parcel_step(carry, _):
    parcel, found = carry
    parcel_new = advance_parcel(parcel)
    found = found | level_crossed(parcel_new)
    next_parcel = jax.lax.cond(found, lambda _: parcel, lambda _: parcel_new, None)
    return (next_parcel, found), parcel_new
(final_parcel, _), trajectory = jax.lax.scan(
    parcel_step, (init_parcel, False), None, length=MAX_LEVELS
)
"""

_FEW_SHOT_IMMUTABLE = """
# FEW-SHOT: immutable array updates
# WRONG - in-place mutation breaks JAX tracing
arr[i] = value
arr[mask] = 0.0

# CORRECT - .at[].set()
arr = arr.at[i].set(value)
arr = arr.at[mask].set(0.0)
"""

_FEW_SHOT_VECTORIZE = """
# FEW-SHOT: flatten nested column/level iteration
# WRONG - nested Python loops
for col in range(n_cols):
    for lev in range(n_levels):
        out[col, lev] = compute(temp[col, lev], pres[col, lev])

# CORRECT - vectorized jnp operations
out = jnp.vectorize(compute)(temp, pres)   # shape: [n_cols, n_levels]
# or, if compute already operates element-wise via jnp:
out = compute(temp, pres)
"""

_ALL_FEW_SHOTS = (
    _FEW_SHOT_LOOP
    + _FEW_SHOT_COND
    + _FEW_SHOT_CONTINUE
    + _FEW_SHOT_SCAN
    + _FEW_SHOT_IMMUTABLE
    + _FEW_SHOT_VECTORIZE
)

# ---------------------------------------------------------------------------
# Shared JAX rules block injected into every prompt
# ---------------------------------------------------------------------------
_JAX_RULES = """
JAX DIFFERENTIABILITY RULES (STRICT - override all other rules):
1. LOOPS        : jax.lax.fori_loop only. No Python for/while.
2. CONTINUE     : lax.cond returning the unchanged carry. No Python continue.
3. IF/ELIF/ELSE : nested lax.cond. No Python if-else inside jitted code.
4. NESTED ITER  : jnp.vectorize or broadcasting. No nested Python loops.
5. ARRAY OPS    : jnp only. No numpy (np.*) calls.
6. MUTATION     : .at[].set() for every array update. No in-place ops.
7. WHILE->SCAN  : replace lax.while_loop with lax.scan + fixed bound +
                  lax.cond stop flag inside scan body.
8. PARCEL TRACK : fixed-iteration lax.scan with a done flag via lax.cond.
"""


TRANSLATION_PROMPTS = {
    "system": f"""You are a Fortran-to-JAX translator specializing in CTSM code.

Core Principles:
- Pure functions, immutable NamedTuples, no side effects
- JIT-compatible: jnp.where for element-wise selection; lax.cond for branching
- Vectorize loops with lax.fori_loop or jnp.vectorize (never Python loops)
- Preserve exact physics equations
- Full type hints and Google-style docstrings
- Reference Fortran source line numbers

{_JAX_RULES}

IN-CONTEXT EXAMPLES (study before generating code):
{_ALL_FEW_SHOTS}""",

    "translate_module": f"""Translate Fortran module to JAX.

MODULE: {{module_name}}

FORTRAN CODE:
```fortran
{{fortran_code}}
```

ANALYSIS:
```json
{{module_info}}
```

CONTEXT (dependencies, translation units, complexity):
```json
{{enhanced_context}}
```

REFERENCE PATTERN:
```python
{{reference_pattern}}
```

{_JAX_RULES}

IN-CONTEXT EXAMPLES:
{_ALL_FEW_SHOTS}

REQUIREMENTS:
1. Structure    : single consolidated module; NamedTuples for data and parameters
2. Functions    : pure with type hints; preserve physics exactly
3. Arrays       : jnp ops only; document shapes as # [n_patches, n_layers]
4. Loops        : lax.fori_loop always; no Python for/while
5. Continue     : lax.cond returning unchanged carry
6. Conditionals : nested lax.cond for branching; jnp.where for element-wise selection
7. Nested iter  : jnp.vectorize or broadcasting
8. Mutation     : .at[].set() only; no in-place ops
9. While->Scan  : lax.scan + fixed bound + lax.cond stop flag
10. Parcel track: fixed-iteration lax.scan with lax.cond early-stop flag
11. Docs        : reference Fortran line numbers from translation units
12. Org         : include parameters/constants inline in main module (no separate files)

Translation units guide:
- "module": header/declarations
- "root"  : complete function
- "inner" : part of split function (note parent)
- Use line_start/line_end for references; high complexity_score means extra care

Output:
1. Single complete physics module (parameters inline)
2. Translation notes""",

    "translate_function": f"""Translate Fortran subroutine to JAX function.

FORTRAN:
```fortran
{{fortran_code}}
```

CONTEXT:
```json
{{context}}
```

{_JAX_RULES}

IN-CONTEXT EXAMPLES:
{_ALL_FEW_SHOTS}

Requirements: pure function, type hints, Google-style docstring with Fortran reference,
preserve physics, lax.fori_loop for loops, nested lax.cond for conditionals,
lax.cond for continue, jnp.vectorize for nested iteration, .at[].set() for updates,
lax.scan replacing any lax.while_loop.""",

    "convert_data_structure": """Convert Fortran type to JAX NamedTuple.

FORTRAN:
```fortran
{fortran_type}
```

Requirements:
- NamedTuple with full type hints
- Map Fortran types to jnp.ndarray (or scalar jnp types)
- Document array shapes in comments, e.g. # [n_patches, n_layers]
- One-line description per field
- Default numeric values must use jnp scalars, not Python floats""",

    "vectorize_loop": f"""Vectorize Fortran loop to JAX.

FORTRAN:
```fortran
{{loop_code}}
```

ANALYSIS: {{loop_analysis}}

{_JAX_RULES}

IN-CONTEXT EXAMPLES:
{_FEW_SHOT_LOOP}
{_FEW_SHOT_COND}
{_FEW_SHOT_CONTINUE}

Requirements: no Python loops; lax.fori_loop for sequential dependencies;
jnp.vectorize or broadcasting for independent ops; lax.cond for continue;
lax.scan replacing lax.while_loop; preserve computation order where physics requires it.""",

    "handle_conditional": f"""Convert Fortran conditional to JIT-compatible JAX.

FORTRAN:
```fortran
{{conditional_code}}
```

{_JAX_RULES}

IN-CONTEXT EXAMPLES:
{_FEW_SHOT_COND}

Requirements: nested lax.cond for if/elif/else branching; jnp.where only for
element-wise value selection; no Python if-else in traced code; preserve all
logical conditions; result must be fully JIT-compatible and differentiable.""",

    "create_parameters": """Create JAX parameter class.

FORTRAN:
```fortran
{parameters}
```

Requirements:
- NamedTuple with all fields typed
- Default values as jnp scalars (e.g. jnp.float32(9.81), not 9.81)
- Document the Fortran source for each constant
- Group related constants with inline comments""",

    "translate_unit": f"""Translate this translation unit to JAX.

MODULE: {{module_name}}
UNIT  : {{unit_id}} ({{unit_type}})
LINES : {{line_start}}-{{line_end}}

FORTRAN CODE:
```fortran
{{fortran_code}}
```

UNIT INFO:
```json
{{unit_info}}
```

CONTEXT (module dependencies, previously translated units):
```json
{{context}}
```

REFERENCE PATTERN:
```python
{{reference_pattern}}
```

{_JAX_RULES}

IN-CONTEXT EXAMPLES:
{_ALL_FEW_SHOTS}

REQUIREMENTS:
- Pure functions with full type hints
- Preserve physics exactly (Fortran lines {{line_start}}-{{line_end}})
- lax.fori_loop for loops; nested lax.cond for conditionals;
  lax.cond for continue; jnp.vectorize for nested iteration;
  .at[].set() for updates; lax.scan replacing lax.while_loop
- If unit_type is "inner", this belongs to parent: {{parent_id}}
- Docstring must reference Fortran lines {{line_start}}-{{line_end}}

Output ONLY the translated code for this unit.""",

    "assemble_module": f"""Assemble complete JAX module from translated units.

MODULE: {{module_name}}

TRANSLATED UNITS:
```json
{{translated_units}}
```

MODULE INFO:
```json
{{module_info}}
```

REFERENCE PATTERN:
```python
{{reference_pattern}}
```

{_JAX_RULES}

IN-CONTEXT EXAMPLES:
{_ALL_FEW_SHOTS}

REQUIREMENTS:
1. Combine all units into a single cohesive module
2. Imports: jax, jax.numpy as jnp, jax.lax, typing, NamedTuple - no bare numpy
3. Order: imports -> NamedTuples -> parameters/constants -> functions
4. Verify consistency: lax.fori_loop (not Python loops); lax.cond or jnp.where
   (not Python if-else); .at[].set() (no in-place); lax.scan (no lax.while_loop)
5. Parameters and constants inline in the main module
6. Module-level docstring summarising the physics

Output:
1. Single complete physics module (all parameters inline)
2. Brief assembly notes""",
}
