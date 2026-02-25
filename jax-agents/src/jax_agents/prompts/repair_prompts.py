"""
Prompts for the Repair Agent.

This agent focuses on fixing failed Python/JAX translations by:
1. Analyzing test failures and error messages
2. Comparing with original Fortran code
3. Identifying root causes (including JAX differentiability violations)
4. Generating corrected Python code

JAX Differentiability Rules (STRICT - violations are always a critical root cause):
- All loops         : jax.lax.fori_loop (never Python for/while)
- Python continue   : lax.cond returning unchanged carry
- Python if/elif/else: nested lax.cond
- Nested col/level iteration: flatten and vectorize
- All array ops     : jnp only (no numpy)
- Immutable updates : .at[].set()
- lax.while_loop    : replace with lax.scan (fixed bound) for full gradient support
- Parcel tracking   : fixed-iteration lax.scan with lax.cond early-stop flag
"""

# ---------------------------------------------------------------------------
# Few-shot examples for in-context learning
# ---------------------------------------------------------------------------

_FEW_SHOT_LOOP_BUG = """
# FEW-SHOT: loop bug and fix
# BUG - Python loop breaks JIT and grad
def compute_profile(delta, n):
    result = jnp.zeros(n)
    for i in range(n):                         # Python loop
        result[i] = result[i - 1] + delta[i]  # in-place mutation
    return result

# FIXED - lax.fori_loop + .at[].set()
def compute_profile(delta, n):
    def body(i, result):
        return result.at[i].set(result[i - 1] + delta[i])
    return jax.lax.fori_loop(0, n, body, jnp.zeros(n))
"""

_FEW_SHOT_COND_BUG = """
# FEW-SHOT: if/elif/else bug and fix
# BUG - Python if-else inside traced code
def classify_phase(temp):
    if temp > 273.15:
        return 1
    elif temp > 200.0:
        return 2
    else:
        return 0

# FIXED - nested lax.cond
def classify_phase(temp):
    return jax.lax.cond(
        temp > 273.15,
        lambda _: 1,
        lambda _: jax.lax.cond(temp > 200.0, lambda _: 2, lambda _: 0, None),
        operand=None,
    )
"""

_FEW_SHOT_WHILE_BUG = """
# FEW-SHOT: while_loop -> scan fix
# BUG - lax.while_loop has limited gradient support
state = jax.lax.while_loop(lambda s: ~converged(s), step, init)

# FIXED - lax.scan with fixed bound + lax.cond early-stop
MAX_ITER = 500
def scan_body(carry, _):
    state, done = carry
    new_state = step(state)
    done = done | converged(new_state)
    next_state = jax.lax.cond(done, lambda _: state, lambda _: new_state, None)
    return (next_state, done), None
(final_state, _), _ = jax.lax.scan(scan_body, (init, False), None, length=MAX_ITER)
"""

_FEW_SHOT_MUTATION_BUG = """
# FEW-SHOT: in-place mutation bug and fix
# BUG - in-place mutation breaks JAX tracing
arr[i] = value
arr[mask] = 0.0

# FIXED - .at[].set()
arr = arr.at[i].set(value)
arr = arr.at[mask].set(0.0)
"""

_FEW_SHOT_CONTINUE_BUG = """
# FEW-SHOT: continue bug and fix
# BUG - Python continue inside a loop
for i in range(n):
    if skip[i]:
        continue
    result = result.at[i].set(compute(x[i]))

# FIXED - lax.fori_loop + lax.cond replaces continue
def body(i, result):
    return jax.lax.cond(
        skip[i],
        lambda _: result,                           # masked path: skip this index
        lambda _: result.at[i].set(compute(x[i])),
        operand=None,
    )
result = jax.lax.fori_loop(0, n, body, jnp.zeros(n))
"""

_ALL_FEW_SHOTS = (
    _FEW_SHOT_LOOP_BUG
    + _FEW_SHOT_COND_BUG
    + _FEW_SHOT_WHILE_BUG
    + _FEW_SHOT_MUTATION_BUG
    + _FEW_SHOT_CONTINUE_BUG
)

# ---------------------------------------------------------------------------
# Shared JAX rules block injected into every prompt
# ---------------------------------------------------------------------------
_JAX_RULES = """
JAX DIFFERENTIABILITY RULES (STRICT - violations are ALWAYS a critical root cause):
1. LOOPS        : jax.lax.fori_loop only. No Python for/while.
2. CONTINUE     : lax.cond returning unchanged carry. No Python continue.
3. IF/ELIF/ELSE : nested lax.cond. No Python if-else in traced code.
4. NESTED ITER  : jnp.vectorize or broadcasting. No nested Python loops.
5. ARRAY OPS    : jnp only. No numpy (np.*) in translated code.
6. MUTATION     : .at[].set(). No in-place ops.
7. WHILE->SCAN  : replace lax.while_loop with lax.scan + fixed bound + lax.cond stop flag.
8. PARCEL TRACK : fixed-iteration lax.scan with lax.cond done flag.
"""


REPAIR_PROMPTS = {
    "system": f"""You are an expert Repair Agent specializing in debugging and fixing Fortran-to-JAX translations.

Your expertise:
- Deep understanding of Fortran semantics and Python/JAX
- Debugging complex numerical computing code
- Root cause analysis of test failures
- Identifying translation errors, logic bugs, and JAX differentiability violations

Your responsibilities:
1. Analyze test failures and error messages carefully
2. Compare failed Python code with the original Fortran implementation
3. Identify root causes - always check JAX differentiability violations first
4. Generate corrected Python code that passes all tests
5. Produce clear root cause analysis reports

You follow these principles:
- Preserve scientific accuracy - physics must match exactly
- Use JAX best practices (pure functions, immutable state, JIT-compatible code)
- Fix only what is broken - do not refactor unrelated code
- Provide clear before/after explanations for every change

{_JAX_RULES}

IN-CONTEXT EXAMPLES (study before diagnosing failures):
{_ALL_FEW_SHOTS}""",

    "analyze_failure": f"""Analyze the test failure and identify all root causes.

**Fortran Subroutine (Original):**
{{fortran_code}}

**Failed Python Function:**
{{python_code}}

**Test Report:**
{{test_report}}

{_JAX_RULES}

IN-CONTEXT EXAMPLES:
{_ALL_FEW_SHOTS}

Perform a two-pass analysis:

Pass 1 - JAX Differentiability Audit:
  Scan the Python code line-by-line for violations of the eight JAX rules above.
  Any violation is automatically severity "critical".

Pass 2 - Semantic / Logic Audit:
  Compare Python semantics against Fortran for indexing errors, type mismatches,
  off-by-one errors, and incorrect physics.

Return as JSON (no extra text):
{{{{
    "failed_tests": ["list of failed test names"],
    "error_summary": "brief one-sentence summary of what went wrong",
    "jax_rule_violations": [
        {{{{
            "rule": "LOOPS|CONTINUE|IF_ELSE|NESTED_ITER|ARRAY_OPS|MUTATION|WHILE_SCAN|PARCEL_TRACK",
            "location": "function name and approximate line description",
            "offending_code": "the exact snippet that violates the rule",
            "explanation": "why this violates the rule and breaks differentiability"
        }}}}
    ],
    "root_causes": [
        {{{{
            "issue": "description of the issue",
            "location": "where in the code",
            "severity": "critical|major|minor",
            "explanation": "detailed explanation"
        }}}}
    ],
    "required_fixes": [
        "ordered list of fixes - JAX violations first, then semantic issues"
    ]
}}}}
""",

    "generate_fix": f"""Generate a corrected version of the Python function based on the root cause analysis.

**Fortran Subroutine (Reference):**
{{fortran_code}}

**Failed Python Function:**
{{python_code}}

**Root Cause Analysis:**
{{root_cause_analysis}}

**Test Report:**
{{test_report}}

{_JAX_RULES}

IN-CONTEXT EXAMPLES:
{_ALL_FEW_SHOTS}

Fix strategy:
1. Address every item in required_fixes in order
2. For each JAX violation apply the correct fix:
   - Python loop       : lax.fori_loop
   - Python continue   : lax.cond returning unchanged carry
   - Python if/elif/else: nested lax.cond
   - Nested iteration  : jnp.vectorize or broadcasting
   - numpy ops         : jnp equivalents
   - in-place mutation : .at[].set()
   - lax.while_loop    : lax.scan + fixed bound + lax.cond stop flag
   - parcel tracking   : fixed-iteration lax.scan with lax.cond done flag
3. Preserve the original function structure where not broken
4. Add a short inline comment for every non-trivial fix, e.g.:
   # FIX: replaced Python for-loop with lax.fori_loop for JAX differentiability
5. Ensure type hints remain correct

Provide ONLY the complete corrected Python function code.
No JSON, no markdown fences, no explanations outside the code.
""",

    "root_cause_report": f"""Generate a comprehensive root cause analysis report.

**Original Fortran Code:**
{{fortran_code}}

**Failed Python Code:**
{{failed_python_code}}

**Corrected Python Code:**
{{corrected_python_code}}

**Failure Analysis:**
{{failure_analysis}}

**Test Results (After Fix):**
{{test_results}}

{_JAX_RULES}

Generate a root cause analysis report in markdown using exactly this structure:

# Root Cause Analysis: {{module_name}}

## 1. Executive Summary
One-paragraph overview of the issue, overall severity, and impact on downstream code.

## 2. Failure Analysis
- Which tests failed and what error messages/assertion failures were observed
- Symptoms: shape errors, wrong values, ConcretizationTypeError, non-finite gradients, etc.
- When and where in execution the failure occurred

## 3. Root Cause Identification

### 3a. JAX Differentiability Violations
For each violation: rule violated, offending code snippet, explanation of why it
breaks JAX tracing or gradient computation.

### 3b. Semantic / Logic Errors
For each bug: issue description, location, and Fortran vs Python comparison.

## 4. Fix Implementation
For each fix: what changed, why it resolves the issue, and a short before/after snippet.

## 5. Test Results
List of test names with PASS/FAIL status after the fix.

## 6. Lessons Learned
Key takeaways for future Fortran-to-JAX translations and JAX pitfalls to watch for.
""",

    "verify_fix": f"""Verify whether the corrected code fully resolves all identified issues.

**Original Failure Analysis:**
{{failure_analysis}}

**Corrected Python Code:**
{{corrected_code}}

**Required Fixes:**
{{required_fixes}}

{_JAX_RULES}

Perform a two-pass verification:

Pass 1 - JAX Differentiability Re-audit:
  Re-scan the corrected code for any remaining violations of the eight JAX rules.
  A single remaining Python loop or if-else is a blocking issue.

Pass 2 - Fix Completeness Check:
  Confirm every item in required_fixes has been addressed.

Return as JSON (no extra text):
{{{{
    "all_issues_addressed": true,
    "remaining_jax_violations": [
        {{{{
            "rule": "rule name",
            "location": "where in the code",
            "offending_code": "the snippet",
            "explanation": "why it still violates the rule"
        }}}}
    ],
    "addressed_issues": ["list of fixed issues from required_fixes"],
    "remaining_concerns": ["any semantic or physics issues still present"],
    "confidence_level": "high|medium|low",
    "recommendations": ["any additional improvements recommended"]
}}}}
""",
}
