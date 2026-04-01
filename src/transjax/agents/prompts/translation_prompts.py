"""
Translation prompts for the Fortran → JAX translator.

Design principles
-----------------
1. Concrete rules beat vague principles.  Tell Claude *exactly* what to produce.
2. Pre-parsed interface contracts eliminate the #1 source of hallucinations
   (wrong argument count / intent / dtype).
3. Explicit JAX anti-pattern table prevents the most common JIT-incompatibility bugs.
4. Single-responsibility: one prompt per concern, no catch-all mega-prompts.
5. Assembly uses full source text (not JSON blobs) so Claude preserves exact code.
"""


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

TRANSLATOR_SYSTEM_PROMPT = """\
You are an expert scientific software engineer translating Fortran Earth System \
Model (ESM) code into GPU-accelerated Python using JAX.

════════════════════════════════════════════════════════════════
MISSION
════════════════════════════════════════════════════════════════
Produce Python/JAX code that is:
  (A) Numerically identical to the Fortran — same equations, same constants,
      same branch logic, same rounding order where it matters.
  (B) A drop-in Python interface that mirrors the Fortran calling convention:
      all intent(in)/intent(inout) become function arguments; all
      intent(out)/intent(inout) are returned as a NamedTuple.
  (C) JIT-compilable with `@jax.jit` without any modification.
  (D) GPU-ready: float64 everywhere (Fortran `real(r8)` standard).

════════════════════════════════════════════════════════════════
FORTRAN → JAX DTYPE TABLE  (memorise this)
════════════════════════════════════════════════════════════════
  Fortran type          │ JAX/NumPy dtype       │ Python type hint
  ─────────────────────────────────────────────────────────────
  real(r8) / real*8     │ jnp.float64           │ jax.Array
  real(r4) / real*4     │ jnp.float32           │ jax.Array
  real      (default)   │ jnp.float64           │ jax.Array  ← always use f64
  integer               │ jnp.int32             │ int | jax.Array
  logical               │ jnp.bool_             │ jax.Array
  complex(r8)           │ jnp.complex128        │ jax.Array

════════════════════════════════════════════════════════════════
JAX ANTI-PATTERN TABLE  (never do these in JIT code)
════════════════════════════════════════════════════════════════
  ✗ WRONG                                │ ✓ CORRECT
  ───────────────────────────────────────────────────────────────
  import numpy as np; np.sin(x)          │ import jax.numpy as jnp; jnp.sin(x)
  arr[i] = value                         │ arr = arr.at[i].set(value)
  for i in range(n): ...                 │ jax.vmap / jax.lax.fori_loop / jnp ops
  if scalar_condition: return x          │ jnp.where(cond, x, y)
  if arr[i] > 0: branch()               │ jax.lax.cond(arr[i] > 0, f_true, f_false, ())
  global state / module-level mutation   │ pass as function argument, return new value
  dtype=float  (Python float = f64 OK)   │ dtype=jnp.float64  (explicit)
  np.zeros(n)                            │ jnp.zeros(n, dtype=jnp.float64)
  x ** 0.5  where x might be negative   │ jnp.sqrt(jnp.maximum(x, 0.0))
  jnp.log(x)  where x might be ≤ 0      │ jnp.log(jnp.maximum(x, _EPS))
  1/x  where x might be 0               │ x / jnp.where(jnp.abs(x) > 0, x*x, 1.0)
  DO i=1,n  (Fortran 1-based index)      │ Python 0-based: range(n), arr[0:n]

════════════════════════════════════════════════════════════════
MANDATORY MODULE HEADER  (copy this exactly)
════════════════════════════════════════════════════════════════
  import jax
  import jax.numpy as jnp
  from functools import partial
  from typing import NamedTuple
  jax.config.update("jax_enable_x64", True)

════════════════════════════════════════════════════════════════
FUNCTION SIGNATURE RULES
════════════════════════════════════════════════════════════════
  1. Arguments: all intent(in) + intent(inout) → positional args (same order).
  2. Returns:   all intent(out) + intent(inout) → fields of a NamedTuple.
  3. Scalars passed as intent(in): keep as Python scalars (int / float).
  4. Arrays: annotate as jax.Array.  Scalar outputs: jnp.float64 etc.
  5. Add @partial(jax.jit, static_argnames=(...)) for scalar-integer args (e.g. n).
  6. Never return None.  Never mutate an input array.

════════════════════════════════════════════════════════════════
PHYSICAL FIDELITY RULES
════════════════════════════════════════════════════════════════
  • Do NOT simplify, factorise, or "clean up" equations.
  • Preserve intermediate variable names from Fortran where possible.
  • Constants (pi, grav, Cp_air, etc.) must use the same numerical values;
    note them as # Fortran constant: NAME = VALUE.
  • SAVE variables / module-level state → accept as an extra input and return
    updated value in the NamedTuple.
  • EQUIVALENCE / COMMON blocks → unpack explicitly as separate arguments.

Output ONLY valid Python — no prose outside code blocks.
"""


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TRANSLATION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

# Required .format() keys: gcm_model_name, module_name, source_file, line_start,
#   line_end, complexity, subroutine_name, interface_table, python_signature,
#   return_fields, fortran_code, already_translated
UNIT_TRANSLATION_PROMPT = """\
## TASK
Translate one Fortran subroutine/function to JAX/Python.
Mirror the interface EXACTLY — same argument order, all outputs returned.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESM model  : {gcm_model_name}
Module     : {module_name}
Source file: {source_file}
Lines      : {line_start}–{line_end}
Complexity : {complexity}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRE-PARSED FORTRAN INTERFACE CONTRACT
(extracted by static analysis — reproduce this interface exactly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Subroutine name : {subroutine_name}

Arguments (in calling order):
{interface_table}

Required Python signature:
{python_signature}

Return NamedTuple fields (intent out + inout):
{return_fields}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL FORTRAN SOURCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```fortran
{fortran_code}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALREADY TRANSLATED IN THIS MODULE (use these, do not redefine)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{already_translated}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSLATION CHECKLIST — verify every point before outputting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ Function name exactly matches subroutine name (snake_case is fine but consistent)
□ Argument list matches the "Required Python signature" above exactly
□ Every intent(out)/intent(inout) field appears in the return NamedTuple
□ No Python for-loops inside the function body (use jnp ops / jax.vmap / jax.lax)
□ No in-place mutation: arr[i]=v → arr.at[i].set(v)
□ No bare `if cond:` on arrays: use jnp.where or jax.lax.cond
□ All array literals use dtype=jnp.float64
□ No `import numpy` — only `import jax.numpy as jnp`
□ Division and sqrt/log are guarded against invalid inputs
□ Fortran 1-based indices converted to 0-based (DO i=1,n → range(n), arr[i-1])
□ Module constants reproduced with same numerical value (comment source line)
□ @partial(jax.jit) decorator added (static_argnames for any integer-shape arg)
□ NamedTuple for return values is defined above the function
□ Module header: jax.config.update("jax_enable_x64", True)

Output a single ```python ... ``` block containing:
  1. NamedTuple definition for return type (if any outputs)
  2. The translated function with @partial(jax.jit, ...) decorator
  3. Inline # comments for non-obvious translations (indexing, math equivalences)
  4. # NOTE: tags for any assumptions or deviations from the Fortran
"""


# ─────────────────────────────────────────────────────────────────────────────
# WHOLE-MODULE PROMPT  (--mode whole: single LLM call for the entire file)
# ─────────────────────────────────────────────────────────────────────────────

# Required .format() keys: gcm_model_name, module_name, source_file,
#   n_routines, interface_summary, fortran_code
WHOLE_MODULE_PROMPT = """\
## TASK
Translate a complete Fortran module to a single JAX/Python file in one pass.
Every subroutine and function in the file must appear in the output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESM model  : {gcm_model_name}
Module     : {module_name}
Source file: {source_file}
Routines   : {n_routines} subroutine(s)/function(s)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRE-PARSED INTERFACE CONTRACTS
(extracted by static analysis — reproduce every interface exactly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{interface_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL FORTRAN SOURCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```fortran
{fortran_code}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce ONE ```python ... ``` block containing the complete translated module:

1. MODULE HEADER (exactly once):
     import jax
     import jax.numpy as jnp
     from functools import partial
     from typing import NamedTuple
     jax.config.update("jax_enable_x64", True)

2. MODULE DOCSTRING:
   \"\"\"
   {module_name}: <one-line physics description>

   Translated from Fortran ESM source.
   Original: {source_file}
   \"\"\"

3. For each subroutine / function (in source order):
   a. One NamedTuple for its return values (if any intent(out)/intent(inout)):
        class <Name>Result(NamedTuple):
            <field>: jax.Array  # or scalar type
   b. The translated function with @partial(jax.jit, ...) decorator.
      Signature must match the pre-parsed interface contract exactly.

4. __all__ listing every translated function name.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSLATION CHECKLIST — verify every point before outputting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ Every subroutine/function from the Fortran source has a Python equivalent
□ Each function signature matches the corresponding interface contract above
□ Every intent(out)/intent(inout) field is in the return NamedTuple
□ No Python for-loops inside function bodies (use jnp ops / jax.lax)
□ No in-place mutation: arr[i]=v → arr.at[i].set(v)
□ No bare `if cond:` on arrays: use jnp.where or jax.lax.cond
□ All array literals use dtype=jnp.float64
□ No `import numpy` — only `import jax.numpy as jnp`
□ Division and sqrt/log guarded against invalid inputs
□ Fortran 1-based indices converted to 0-based
□ Module constants reproduced with same numerical value (comment source line)
□ jax.config.update("jax_enable_x64", True) appears exactly once
□ NamedTuples defined before first use; callees appear before callers
□ __all__ lists all public functions
"""


# ─────────────────────────────────────────────────────────────────────────────
# MODULE ASSEMBLY PROMPT
# ─────────────────────────────────────────────────────────────────────────────

# Required .format() keys: n_units, module_name, gcm_model_name, original_file,
#   public_api, all_units_source
MODULE_ASSEMBLY_PROMPT = """\
## TASK
Assemble {n_units} individually translated JAX units into one coherent Python module.
Do NOT change the physics or function signatures — only reorganise and deduplicate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE METADATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Module name   : {module_name}
ESM model     : {gcm_model_name}
Original file : {original_file}
Public API    : {public_api}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSLATED UNITS (full source, in translation order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{all_units_source}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSEMBLY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. MODULE HEADER (exactly once at top):
     import jax
     import jax.numpy as jnp
     from functools import partial
     from typing import NamedTuple, Tuple
     jax.config.update("jax_enable_x64", True)

2. ORDER:  imports → NamedTuples/types → constants → helper functions → main functions

3. DEDUPLICATION:
   • Merge identical imports into one block.
   • If two units define the same NamedTuple, keep one definition only.
   • If two units define the same constant with the same value, keep one.
   • If two units define the same helper function, keep one.

4. DO NOT change any function body, signature, decorator, or docstring.
   The only allowed edits are removing duplicates and reordering.

5. Add a module docstring:
   \"\"\"
   {module_name}: [one-line physics description inferred from the code]

   Translated from Fortran ESM source.
   Original: {original_file}
   \"\"\"

6. Add at the bottom:
   __all__ = {public_api}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSEMBLY CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ jax.config.update("jax_enable_x64", True) appears exactly once
□ No `import numpy` — only jax.numpy
□ All function signatures unchanged from unit translations
□ All NamedTuples defined before first use
□ callees appear before callers
□ __all__ lists all public functions

Output a single ```python ... ``` block containing the complete assembled module.
"""


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY COMPAT — kept so existing call sites that reference the old dict
# do not crash.  New code should use the named constants above.
# ─────────────────────────────────────────────────────────────────────────────

TRANSLATION_PROMPTS = {
    "system":           TRANSLATOR_SYSTEM_PROMPT,
    "translate_unit":   UNIT_TRANSLATION_PROMPT,
    "assemble_module":  MODULE_ASSEMBLY_PROMPT,
    "whole_module":     WHOLE_MODULE_PROMPT,
    # These sub-tasks are handled inline by translator.py rather than via
    # dedicated prompts.  The keys are kept so legacy call sites don't crash.
    "translate_module":       "# handled by translate_module() in translator.py",
    "translate_function":     "# handled by translate_module() in translator.py",
    "convert_data_structure": "# handled inline by translator",
    "vectorize_loop":         "# handled inline by translator",
    "handle_conditional":     "# handled inline by translator",
    "create_parameters":      "# handled inline by translator",
}
