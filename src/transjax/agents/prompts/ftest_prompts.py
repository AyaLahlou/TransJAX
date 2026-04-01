"""
Prompts for the FtestAgent — functional test suite builder for Fortran ESM codebases.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

FTEST_SYSTEM_PROMPT = """\
You are an expert in building functional test suites for Fortran Earth System
Model (ESM) codebases targeting HPC environments (nvfortran + NetCDF).

Your job is to analyse Fortran subroutine signatures and generate:

1. Thin Fortran "test driver" programs that
   - accept inputs via a Fortran NAMELIST block read from stdin
   - call the subroutine under test
   - write every intent(out)/intent(inout) scalar variable to stdout as KEY=VALUE

2. Python pytest files that
   - invoke the compiled Fortran driver as a subprocess
   - feed controlled namelist inputs
   - assert on the KEY=VALUE output with appropriate scientific tolerances

Rules:
- Only scalar real/integer/logical variables in NAMELIST (skip arrays, write
  ARRAY_SKIPPED=1 to stdout instead of crashing).
- Use Fortran 90 free-form source, compatible with nvfortran.
- Python tests must be self-contained, importable, and run with `pytest`.
- Produce complete, compilable Fortran code and complete, runnable pytest code.
- Output code only — no prose before or after the code blocks.
"""

# ---------------------------------------------------------------------------
# Interface analysis
# ---------------------------------------------------------------------------

FTEST_ANALYZE_SUBROUTINE_PROMPT = """\
Analyse this Fortran subroutine and extract its calling interface.

```fortran
{fortran_code}
```

Module: {module_name}
File:   {file_path}

Return a single JSON object with EXACTLY this structure (no extra keys):

{{
  "subroutine_name": "<name>",
  "module_name": "<module or null>",
  "intent_in": [
    {{"name": "<var>", "type": "<Fortran type>", "dimensions": "<'' for scalar, '(:)' etc for array>", "default_value": "<sensible test default as string>", "description": "<brief>"}}
  ],
  "intent_out": [
    {{"name": "<var>", "type": "<Fortran type>", "dimensions": ""}}
  ],
  "intent_inout": [
    {{"name": "<var>", "type": "<Fortran type>", "dimensions": "", "default_value": "<string>"}}
  ],
  "use_statements": ["<module1>", "<module2>"],
  "kind_parameters": {{"r8": "selected_real_kind(12,300)", "i4": "selected_int_kind(9)"}},
  "notes": "<any special compilation or calling notes>"
}}

Constraints:
- Include only scalar variables (dimensions == "") in intent_in/inout for NAMELIST.
- For array variables, still list them in intent_out but set dimensions to the shape.
- Provide a sensible numeric default_value for every intent_in/inout scalar.
- Output ONLY the raw JSON object — no markdown, no prose.
"""

# ---------------------------------------------------------------------------
# Fortran driver generation
# ---------------------------------------------------------------------------

FTEST_DRIVER_PROMPT = """\
Generate a Fortran 90 test-driver program for the subroutine described below.

Subroutine : {subroutine_name}
Module     : {module_name}
Compiler   : nvfortran (HPC)

Interface JSON:
{interface_json}

Requirements:
1.  Program name: test_{subroutine_name}
2.  If module_name is not null, add:  USE {module_name}
3.  Declare all intent(in) and intent(inout) scalar variables as local variables
    using the types listed in the interface JSON.  Declare kind parameters if needed.
4.  Declare all intent(out) scalar variables as local variables.
5.  Set each intent(in)/intent(inout) variable to its default_value before reading
    the namelist (fallback if the namelist omits a variable).
6.  Define a NAMELIST /inputs/ containing only scalar intent(in) and intent(inout)
    variables.
7.  Read the namelist from stdin with error handling:
        integer :: ios
        read(*, nml=inputs, iostat=ios)
        if (ios /= 0) stop 'ERROR: namelist read failed'
8.  Call the subroutine.
9.  For each scalar intent(out)/intent(inout) variable write:
        write(*, '(A,G0)') '<varname>=', <varname>
10. For any array intent(out) variable write:
        write(*, '(A)') '<varname>=ARRAY_SKIPPED'
11. End the program cleanly.

Output ONLY the Fortran source code inside a ```fortran … ``` block.
"""

# ---------------------------------------------------------------------------
# Python pytest generation
# ---------------------------------------------------------------------------

FTEST_PYTEST_PROMPT = """\
Generate a Python pytest file for the Fortran subroutine test driver described below.

Subroutine : {subroutine_name}
Module     : {module_name}
Driver exe : ${{FTEST_DRIVER_DIR}}/test_{subroutine_name}

Interface JSON:
{interface_json}

Requirements:
1.  Top imports: subprocess, os, pathlib.Path, pytest, math.
2.  Retrieve the driver directory:
        DRIVER_DIR = pathlib.Path(
            os.environ.get("FTEST_DRIVER_DIR",
                           str(pathlib.Path(__file__).parent.parent / "drivers" / "bin"))
        )
        DRIVER = DRIVER_DIR / "test_{subroutine_name}"
3.  Do NOT redefine run_driver — it is imported from conftest.py by pytest
    automatically.  Use it directly in test functions as `run_driver(DRIVER, inputs)`.
4.  Write exactly three test functions:

    def test_{subroutine_name}_smoke():
        \"\"\"Driver runs without crashing for typical inputs.\"\"\"
        # typical mid-range values, assert return is a dict

    def test_{subroutine_name}_typical():
        \"\"\"Outputs are within physically plausible ranges for typical inputs.\"\"\"
        # assert specific output keys exist and are finite

    @pytest.mark.parametrize("inputs", [
        {{...}},   # zeros / near-zero
        {{...}},   # extremes
    ])
    def test_{subroutine_name}_edge(inputs):
        \"\"\"Driver handles edge/boundary inputs gracefully.\"\"\"
        # assert driver does not crash (returncode check is inside run_driver)

5.  Use `pytest.importorskip` or `pytest.skip` if the driver binary is missing:
        if not DRIVER.exists():
            pytest.skip("driver not compiled — run `make all` in ftest output dir")
6.  Use `math.isfinite` or `abs(x - expected) < rtol * abs(expected)` for float
    comparisons (rtol = 1e-5).
7.  Fill in realistic ESM physical values (temperatures ~270–300 K, pressures
    ~1e5 Pa, etc.) for the test inputs based on the variable names/descriptions.

Output ONLY the Python source code inside a ```python … ``` block.
"""
