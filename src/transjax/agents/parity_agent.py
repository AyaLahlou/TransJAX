"""
ParityAgent — numerical parity testing between a JAX translation and golden
Fortran reference data.

Architecture
------------
Primary path (recommended — with --ftest-report):
  ftest_report.json provides the exact subroutine interface (argument names,
  Fortran types, array dimensions, intent). The parity pytest is built
  PROGRAMMATICALLY from this interface + the golden JSON. No LLM call is made,
  so there is no risk of interface misinterpretation, and generation is instant.

Fallback path (without --ftest-report):
  Claude infers the interface from the JAX source code. Less accurate and
  more expensive, but works when ftest has not been run.

Workflow
--------
For each golden JSON in the golden directory matching the Python module:
  1. Generate — build a standalone pytest that loads the JAX module by path,
     reconstructs golden inputs as JAX arrays with the correct dtypes, calls
     the function, and asserts each output field matches within jnp.allclose
     tolerance.
  2. Execute  — run the generated pytest with subprocess, capture pass/fail.

Output
------
<output_dir>/
  test_parity_<module>_<subroutine>.py   generated pytest file
  parity_report.json                      aggregated results
"""

import json
import logging
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from transjax.agents.base_agent import BaseAgent
from transjax.agents.prompts.parity_prompts import PARITY_SYSTEM_PROMPT, PARITY_TEST_PROMPT
from transjax.agents.utils.config_loader import get_llm_config

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Fortran type → JAX dtype mapping (used in programmatic test generation)
# ---------------------------------------------------------------------------

def _fortran_type_to_jax_dtype(fortran_type: str) -> str:
    """Return the jnp dtype string for a Fortran type declaration."""
    t = fortran_type.lower().strip()
    if re.search(r"real\s*\(\s*r?8\s*\)|real\s*\*\s*8|double\s+precision", t):
        return "jnp.float64"
    if re.search(r"real\s*\(\s*r?4\s*\)|real\s*\*\s*4", t):
        return "jnp.float32"
    if re.search(r"\breal\b", t):
        return "jnp.float64"       # default real → float64
    if re.search(r"\binteger\b", t):
        return "jnp.int32"
    if re.search(r"\blogical\b", t):
        return "jnp.bool_"
    if re.search(r"\bcomplex\b", t):
        return "jnp.complex128"
    return "jnp.float64"           # safe default


def _is_array(dimensions: str) -> bool:
    """Return True if the dimensions string indicates an array."""
    return bool(dimensions and dimensions.strip())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParitySubroutineResult:
    """Outcome for all golden cases of one subroutine."""
    subroutine_name: str
    module_name: str
    golden_file: Path
    test_file: Path
    cases_total: int = 0
    cases_passed: int = 0
    cases_failed: int = 0
    test_generated: bool = False
    generation_error: str = ""
    pytest_stdout: str = ""
    pytest_returncode: int = -1
    used_ftest_interface: bool = False   # True = programmatic, False = LLM fallback

    @property
    def all_passed(self) -> bool:
        return self.cases_passed == self.cases_total and self.cases_total > 0

    @property
    def status(self) -> str:
        if not self.test_generated:
            return "generation_failed"
        return "passed" if self.pytest_returncode == 0 else "failed"


@dataclass
class ParityRunResult:
    """Aggregated outcome of a full parity test run."""
    python_file: Path
    golden_dir: Path
    output_dir: Path
    ftest_report_path: Optional[Path] = None
    subroutine_results: List[ParitySubroutineResult] = field(default_factory=list)

    @property
    def total_subroutines(self) -> int:
        return len(self.subroutine_results)

    @property
    def subroutines_passed(self) -> int:
        return sum(1 for r in self.subroutine_results if r.status == "passed")

    @property
    def subroutines_failed(self) -> int:
        return sum(1 for r in self.subroutine_results if r.status == "failed")

    @property
    def total_cases(self) -> int:
        return sum(r.cases_total for r in self.subroutine_results)

    @property
    def cases_passed(self) -> int:
        return sum(r.cases_passed for r in self.subroutine_results)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ParityAgent(BaseAgent):
    """
    Generates and runs numerical parity tests for a JAX-translated module.

    PRIMARY path (with ftest_report):
      Reads the Ftest interface for each subroutine (argument names, Fortran
      types, array dimensions). Uses this to generate the parity pytest
      PROGRAMMATICALLY — no LLM call, guaranteed interface accuracy.

    FALLBACK path (without ftest_report):
      Sends the JAX source + golden JSON to Claude and asks it to infer the
      interface and generate the test.  Slower and less reliable.
    """

    DEFAULT_RTOL = 1e-10
    DEFAULT_ATOL = 1e-12

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        llm_config = get_llm_config()
        super().__init__(
            name="ParityAgent",
            role=(
                "Expert Python/JAX testing engineer who writes numerically accurate "
                "parity tests verifying that JAX translations reproduce Fortran "
                "golden reference outputs to machine precision."
            ),
            model=model or llm_config.get("model", "claude-sonnet-4-6"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 16000),
        )

    # ---------------------------------------------------------------------- #
    # Public entry point                                                       #
    # ---------------------------------------------------------------------- #

    def run(
        self,
        python_file: Path,
        golden_dir: Path,
        output_dir: Path,
        ftest_report_path: Optional[Path] = None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        verbose: bool = False,
    ) -> ParityRunResult:
        """
        Generate and run parity tests for *python_file* against golden data.

        Args:
            python_file:       Translated JAX/Python module to test.
            golden_dir:        Directory containing golden JSON files.
            output_dir:        Where to write generated test files and report.
            ftest_report_path: Path to ftest_report.json (from ``transjax ftest``).
                               When provided, parity tests are built programmatically
                               from the Ftest interface — no LLM inference needed.
                               When omitted, Claude infers the interface from JAX
                               source (less accurate, slower).
            rtol:              Relative tolerance for jnp.allclose (default 1e-10).
            atol:              Absolute tolerance for jnp.allclose (default 1e-12).
            verbose:           Print extra output including full pytest logs.

        Returns:
            ParityRunResult with per-subroutine outcomes.
        """
        python_file = Path(python_file).resolve()
        golden_dir  = Path(golden_dir).resolve()
        output_dir  = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        run_result = ParityRunResult(
            python_file=python_file,
            golden_dir=golden_dir,
            output_dir=output_dir,
            ftest_report_path=ftest_report_path,
        )

        # Load Ftest interfaces — the single source of truth for argument metadata
        ftest_interfaces: Dict[str, Dict[str, Any]] = {}
        if ftest_report_path:
            ftest_report_path = Path(ftest_report_path).resolve()
            if ftest_report_path.exists():
                ftest_interfaces = self._load_ftest_interfaces(ftest_report_path)
                console.print(
                    f"[dim]Loaded {len(ftest_interfaces)} subroutine interface(s) "
                    f"from {ftest_report_path.name}[/dim]"
                )
            else:
                logger.warning("ftest_report not found: %s", ftest_report_path)
                console.print(f"[yellow]ftest_report not found: {ftest_report_path}[/yellow]")
        else:
            console.print(
                "[yellow]No --ftest-report provided. Falling back to LLM-based test "
                "generation (less accurate). Provide --ftest-report for best results.[/yellow]"
            )

        info_mode = (
            "[green]programmatic (ftest interface)[/green]"
            if ftest_interfaces else
            "[yellow]LLM inference (fallback)[/yellow]"
        )
        console.print(
            Panel.fit(
                f"[white]Python module:[/white] {python_file}\n"
                f"[white]Golden dir:   [/white] {golden_dir}\n"
                f"[white]Output dir:   [/white] {output_dir}\n"
                f"[white]Tolerance:    [/white] rtol={rtol}, atol={atol}\n"
                f"[white]Test gen:     [/white] {info_mode}",
                title="[bold cyan]TransJAX Parity Test[/bold cyan]",
            )
        )

        if not python_file.exists():
            console.print(f"[red]Python file not found: {python_file}[/red]")
            return run_result
        python_source = python_file.read_text()

        module_name  = python_file.stem
        golden_files = self._find_golden_files(golden_dir, module_name)

        if not golden_files:
            console.print(
                f"[yellow]No golden files found for module '{module_name}' "
                f"in {golden_dir}[/yellow]"
            )
            return run_result

        console.print(f"[cyan]Found {len(golden_files)} golden file(s)[/cyan]")

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
            task = prog.add_task("Testing...", total=len(golden_files))
            for golden_file in golden_files:
                prog.update(task, description=f"[cyan]{golden_file.name}[/cyan]")
                result = self._process_golden_file(
                    golden_file=golden_file,
                    python_file=python_file,
                    python_source=python_source,
                    module_name=module_name,
                    output_dir=output_dir,
                    rtol=rtol,
                    atol=atol,
                    verbose=verbose,
                    ftest_interfaces=ftest_interfaces,
                )
                run_result.subroutine_results.append(result)
                prog.advance(task)

        self._print_summary(run_result)
        self._write_report(run_result)
        return run_result

    # ---------------------------------------------------------------------- #
    # Ftest interface loading                                                   #
    # ---------------------------------------------------------------------- #

    def _load_ftest_interfaces(
        self, report_path: Path
    ) -> Dict[str, Dict[str, Any]]:
        """
        Read ftest_report.json and return a dict mapping subroutine name →
        interface dict (as produced by FtestAgent._analyze_interface).

        Interface schema:
          {
            "subroutine_name": "...",
            "module_name": "...",
            "intent_in":    [{"name": ..., "type": ..., "dimensions": ..., ...}],
            "intent_out":   [{"name": ..., "type": ..., "dimensions": ..., ...}],
            "intent_inout": [{"name": ..., "type": ..., "dimensions": ..., ...}],
            ...
          }
        """
        try:
            report = json.loads(report_path.read_text())
        except Exception as exc:
            logger.warning("Could not load ftest_report.json: %s", exc)
            return {}

        interfaces: Dict[str, Dict[str, Any]] = {}
        for sub in report.get("subroutines", []):
            if sub.get("status") == "ok" and "interface" in sub:
                name = sub.get("name") or sub.get("subroutine_name", "")
                if name:
                    interfaces[name.lower()] = sub["interface"]
        return interfaces

    # ---------------------------------------------------------------------- #
    # Per-golden-file processing                                               #
    # ---------------------------------------------------------------------- #

    def _process_golden_file(
        self,
        golden_file: Path,
        python_file: Path,
        python_source: str,
        module_name: str,
        output_dir: Path,
        rtol: float,
        atol: float,
        verbose: bool,
        ftest_interfaces: Dict[str, Dict[str, Any]],
    ) -> ParitySubroutineResult:
        try:
            golden_data = json.loads(golden_file.read_text())
        except Exception as exc:
            logger.error("Cannot read golden file %s: %s", golden_file, exc)
            return ParitySubroutineResult(
                subroutine_name=golden_file.stem,
                module_name=module_name,
                golden_file=golden_file,
                test_file=output_dir / f"test_parity_{golden_file.stem}.py",
                generation_error=str(exc),
            )

        subroutine_name = golden_data.get("subroutine", golden_file.stem.split("_", 1)[-1])
        n_cases = len(golden_data.get("cases", []))

        test_filename = f"test_parity_{module_name}_{subroutine_name}.py"
        test_file = output_dir / test_filename

        # Look up interface (case-insensitive)
        interface = ftest_interfaces.get(subroutine_name.lower())

        result = ParitySubroutineResult(
            subroutine_name=subroutine_name,
            module_name=module_name,
            golden_file=golden_file,
            test_file=test_file,
            cases_total=n_cases,
            used_ftest_interface=interface is not None,
        )

        # ── Step 1: Generate the pytest ──────────────────────────────────────
        path_label = "[green]ftest interface[/green]" if interface else "[yellow]LLM fallback[/yellow]"
        console.print(
            f"\n[bold]  Step 1[/bold] Generate parity test — "
            f"[cyan]{subroutine_name}[/cyan] ({n_cases} case(s)) via {path_label}"
        )
        try:
            if interface:
                test_code = self._generate_parity_test_programmatic(
                    subroutine_name=subroutine_name,
                    module_name=module_name,
                    python_file=python_file,
                    golden_data=golden_data,
                    test_filename=test_filename,
                    rtol=rtol,
                    atol=atol,
                    interface=interface,
                )
            else:
                test_code = self._generate_parity_test_with_llm(
                    subroutine_name=subroutine_name,
                    module_name=module_name,
                    python_file=python_file,
                    python_source=python_source,
                    golden_data=golden_data,
                    test_filename=test_filename,
                    rtol=rtol,
                    atol=atol,
                )
            test_file.write_text(test_code)
            result.test_generated = True
            console.print(f"  [green]✓[/green] Written: {test_file}")
        except Exception as exc:
            result.generation_error = str(exc)
            console.print(f"  [red]✗ Generation failed: {exc}[/red]")
            return result

        # ── Step 2: Run the pytest ────────────────────────────────────────────
        console.print(f"  [bold]Step 2[/bold] Run parity test — [cyan]{subroutine_name}[/cyan]")
        returncode, stdout = self._run_pytest(test_file, verbose=verbose)
        result.pytest_returncode = returncode
        result.pytest_stdout = stdout

        passed, failed = self._parse_pytest_counts(stdout)
        result.cases_passed = passed
        result.cases_failed = failed

        if returncode == 0:
            console.print(f"  [green]✓ All {n_cases} case(s) passed[/green]")
        else:
            console.print(f"  [red]✗ {failed} case(s) failed / {n_cases} total[/red]")
            if verbose:
                console.print(stdout)

        return result

    # ---------------------------------------------------------------------- #
    # PRIMARY: Programmatic test generation from Ftest interface               #
    # ---------------------------------------------------------------------- #

    def _generate_parity_test_programmatic(
        self,
        subroutine_name: str,
        module_name: str,
        python_file: Path,
        golden_data: Dict[str, Any],
        test_filename: str,
        rtol: float,
        atol: float,
        interface: Dict[str, Any],
    ) -> str:
        """
        Build a parity pytest programmatically from the Ftest interface.

        No LLM call is made.  The interface provides exact argument names,
        Fortran types, and array dimensions — so there is no risk of Claude
        misinterpreting the JAX function signature.

        Input reconstruction rules (from Ftest interface):
          • ``dimensions == ""``   → scalar Python literal
          • ``dimensions != ""``   → jnp.array([...], dtype=<jax_dtype>)
          • Fortran real(r8)       → jnp.float64
          • Fortran integer        → jnp.int32
          • Fortran logical        → bool
        """
        intent_in    = interface.get("intent_in",    [])
        intent_out   = interface.get("intent_out",   [])
        intent_inout = interface.get("intent_inout", [])

        # Build dtype-aware input reconstruction for each intent(in) variable
        input_lines: List[str] = []
        for var in intent_in:
            name  = var["name"]
            ftype = var.get("type", "real")
            dims  = var.get("dimensions", "")
            dtype = _fortran_type_to_jax_dtype(ftype)

            if _is_array(dims):
                input_lines.append(
                    f'    if "{name}" in raw:\n'
                    f'        val = raw["{name}"]\n'
                    f'        inputs["{name}"] = jnp.array(val, dtype={dtype}) '
                    f'if isinstance(val, list) else jnp.asarray(val, dtype={dtype})'
                )
            else:
                # Scalar — keep as Python primitive for JIT static-arg friendliness
                if dtype == "jnp.int32":
                    input_lines.append(
                        f'    if "{name}" in raw:\n'
                        f'        inputs["{name}"] = int(raw["{name}"])'
                    )
                elif dtype == "jnp.bool_":
                    input_lines.append(
                        f'    if "{name}" in raw:\n'
                        f'        inputs["{name}"] = bool(raw["{name}"])'
                    )
                else:
                    input_lines.append(
                        f'    if "{name}" in raw:\n'
                        f'        inputs["{name}"] = {dtype}(raw["{name}"])'
                    )

        # Output variable names (intent_out + intent_inout)
        output_vars = [v["name"] for v in intent_out + intent_inout]
        output_vars_repr = repr(output_vars)

        input_block = "\n".join(input_lines) if input_lines else "    pass  # no intent(in) vars"

        # Embed golden data inline so the test is fully self-contained.
        # Escape triple-quotes so the JSON can be embedded in a triple-quoted string.
        golden_json_str = json.dumps(golden_data, indent=2)
        golden_json_escaped = golden_json_str.replace('"""', '\\"\\"\\"')

        # Human-readable interface summary for the docstring
        interface_vars = ", ".join(
            f"{v['name']}({v.get('type','?')})" for v in intent_in
        ) or "(none)"

        # Subroutine name used as the source-file label inside the embedded comment
        golden_source_label = str(golden_data.get("subroutine", subroutine_name))

        # Build the test file using an f-string for values known at this point.
        # Placeholders for content that contains { or } (JSON, input_block) are
        # double-braced so the f-string emits them as literal text; they are then
        # substituted via str.replace() below.
        code = textwrap.dedent(f'''\
            """
            Parity test: {subroutine_name} — JAX vs. Fortran golden reference.

            Auto-generated by TransJAX ParityAgent (programmatic, from ftest interface).
            Do not edit by hand; re-run `transjax test-parity` to regenerate.

            Tolerance: rtol={rtol}, atol={atol}
            """

            import importlib.util
            import json
            from pathlib import Path
            from typing import Any, Dict

            import jax
            import jax.numpy as jnp
            import pytest

            # Enable float64 — must come before any JAX computation
            jax.config.update("jax_enable_x64", True)

            # ---------------------------------------------------------------------------
            # Load JAX module under test
            # ---------------------------------------------------------------------------
            _PYTHON_FILE = Path({str(python_file)!r})
            _spec = importlib.util.spec_from_file_location({module_name!r}, _PYTHON_FILE)
            _mod  = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _fn   = getattr(_mod, {subroutine_name!r})

            # ---------------------------------------------------------------------------
            # Golden reference data  (embedded from {golden_source_label})
            # ---------------------------------------------------------------------------
            _GOLDEN = json.loads("""
            <<GOLDEN_JSON>>
            """)

            CASES = _GOLDEN.get("cases", [])

            # Tolerance
            RTOL = {rtol!r}
            ATOL = {atol!r}

            # ---------------------------------------------------------------------------
            # Input reconstruction
            # Interface source: ftest_report.json ({subroutine_name})
            # ---------------------------------------------------------------------------

            def _build_inputs(raw: Dict[str, Any]) -> Dict[str, Any]:
                """Convert golden JSON inputs to JAX arrays with correct dtypes.

                Dtype mapping follows the Fortran interface extracted by FtestAgent:
                {interface_vars}
                """
                inputs: Dict[str, Any] = {{}}
            <<INPUT_BLOCK>>
                return inputs

            # Expected output field names (intent(out) + intent(inout))
            _OUTPUT_FIELDS = {output_vars_repr}

            # ---------------------------------------------------------------------------
            # Parity test
            # ---------------------------------------------------------------------------

            @pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
            def test_parity_{subroutine_name}(case):
                """Verify JAX output matches Fortran golden data within tolerance."""
                inputs = _build_inputs(case["inputs"])
                result = _fn(**inputs)

                for field_name in _OUTPUT_FIELDS:
                    if field_name not in case["outputs"]:
                        continue  # golden does not cover this output — skip

                    expected = case["outputs"][field_name]
                    expected_arr = jnp.asarray(expected, dtype=jnp.float64)

                    # Handle NamedTuple, dict, or bare array return types
                    if hasattr(result, field_name):
                        actual_arr = jnp.asarray(getattr(result, field_name))
                    elif isinstance(result, dict) and field_name in result:
                        actual_arr = jnp.asarray(result[field_name])
                    else:
                        pytest.fail(
                            f"Case {{case['id']}}: output field '{{field_name}}' not found "
                            f"in result (type: {{type(result).__name__}})"
                        )

                    matches = jnp.allclose(actual_arr, expected_arr, rtol=RTOL, atol=ATOL)
                    if not matches:
                        diff = jnp.abs(actual_arr - expected_arr)
                        pytest.fail(
                            f"Case {{case['id']}}: {{field_name}} mismatch\\n"
                            f"  expected : {{expected_arr.tolist()}}\\n"
                            f"  actual   : {{actual_arr.tolist()}}\\n"
                            f"  max |err|: {{float(jnp.max(diff)):.6e}}"
                        )
        ''')

        # Substitute the two blocks that contain { / } and can't live in the f-string.
        code = code.replace("<<GOLDEN_JSON>>", golden_json_escaped)
        code = code.replace("<<INPUT_BLOCK>>", input_block)

        return code

    # ---------------------------------------------------------------------- #
    # FALLBACK: LLM-based test generation                                      #
    # ---------------------------------------------------------------------- #

    def _generate_parity_test_with_llm(
        self,
        subroutine_name: str,
        module_name: str,
        python_file: Path,
        python_source: str,
        golden_data: Dict[str, Any],
        test_filename: str,
        rtol: float,
        atol: float,
    ) -> str:
        """
        Ask Claude to write the parity pytest (fallback when ftest_report
        is not available).

        Claude infers the function interface from the JAX source code.
        This is less reliable than the programmatic path — prefer providing
        --ftest-report whenever possible.
        """
        trimmed = dict(golden_data)
        if len(trimmed.get("cases", [])) > 60:
            trimmed["cases"] = trimmed["cases"][:60]
            trimmed["_note"] = "truncated to 60 cases for prompt"

        prompt = PARITY_TEST_PROMPT.format(
            subroutine_name=subroutine_name,
            module_name=module_name,
            python_file=str(python_file),
            python_source=python_source,
            golden_json=json.dumps(trimmed, indent=2),
            rtol=rtol,
            atol=atol,
            output_filename=test_filename,
        )

        response = self.query_claude(
            prompt=prompt,
            system_prompt=PARITY_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
        )
        return self._extract_code(response)

    # ---------------------------------------------------------------------- #
    # Test execution                                                            #
    # ---------------------------------------------------------------------- #

    def _run_pytest(self, test_file: Path, verbose: bool = False) -> Tuple[int, str]:
        """Run pytest on *test_file*, return (returncode, combined stdout+stderr)."""
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", "--no-header"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return proc.returncode, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            return 1, "pytest timed out after 120 seconds"
        except Exception as exc:
            return 1, f"Failed to run pytest: {exc}"

    def _parse_pytest_counts(self, stdout: str) -> Tuple[int, int]:
        """Extract (passed, failed) counts from pytest -v output."""
        m = re.search(r"(\d+)\s+passed", stdout)
        passed = int(m.group(1)) if m else 0
        m = re.search(r"(\d+)\s+failed", stdout)
        failed = int(m.group(1)) if m else 0
        return passed, failed

    # ---------------------------------------------------------------------- #
    # Golden file discovery                                                     #
    # ---------------------------------------------------------------------- #

    def _find_golden_files(self, golden_dir: Path, module_name: str) -> List[Path]:
        """
        Return golden JSON files in *golden_dir* belonging to *module_name*.

        Match priority (case-insensitive):
          1. Files whose name starts with ``<module_name>_``
          2. Files whose name contains  ``<module_name>``
          3. Fallback: all .json files in the directory
        """
        if not golden_dir.exists():
            return []
        all_json = sorted(golden_dir.glob("*.json"))
        lower = module_name.lower()
        prefix = [f for f in all_json if f.name.lower().startswith(lower + "_")]
        if prefix:
            return prefix
        substr = [f for f in all_json if lower in f.name.lower()]
        if substr:
            return substr
        return all_json

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    def _extract_code(self, response: str) -> str:
        if "```python" in response:
            start = response.find("```python") + 9
            end   = response.find("```", start)
            return response[start:end if end != -1 else None].strip()
        if "```" in response:
            start = response.find("```") + 3
            end   = response.find("```", start)
            return response[start:end if end != -1 else None].strip()
        return response.strip()

    # ---------------------------------------------------------------------- #
    # Reporting                                                                #
    # ---------------------------------------------------------------------- #

    def _print_summary(self, run: ParityRunResult) -> None:
        table = Table(title="Parity Test Results", show_lines=True)
        table.add_column("Subroutine",  style="cyan",  no_wrap=True)
        table.add_column("Interface",   justify="center")
        table.add_column("Cases",       justify="right")
        table.add_column("Passed",      justify="right", style="green")
        table.add_column("Failed",      justify="right", style="red")
        table.add_column("Status",      justify="center")

        for r in run.subroutine_results:
            iface_str = "[green]ftest[/green]" if r.used_ftest_interface else "[yellow]LLM[/yellow]"
            status_str = {
                "passed":           "[green]✓ PASS[/green]",
                "failed":           "[red]✗ FAIL[/red]",
                "generation_failed":"[yellow]⚠ GEN ERR[/yellow]",
            }.get(r.status, r.status)
            table.add_row(
                r.subroutine_name, iface_str,
                str(r.cases_total), str(r.cases_passed), str(r.cases_failed),
                status_str,
            )

        console.print()
        console.print(table)
        console.print(
            Panel.fit(
                f"[white]Subroutines:[/white]  {run.total_subroutines}  "
                f"([green]{run.subroutines_passed} passed[/green] / "
                f"[red]{run.subroutines_failed} failed[/red])\n"
                f"[white]Cases total:[/white]  {run.total_cases}  "
                f"([green]{run.cases_passed} passed[/green])",
                title="[bold]Summary[/bold]",
            )
        )

    def _write_report(self, run: ParityRunResult) -> None:
        report = {
            "python_file":      str(run.python_file),
            "golden_dir":       str(run.golden_dir),
            "output_dir":       str(run.output_dir),
            "ftest_report":     str(run.ftest_report_path) if run.ftest_report_path else None,
            "summary": {
                "subroutines_total":  run.total_subroutines,
                "subroutines_passed": run.subroutines_passed,
                "subroutines_failed": run.subroutines_failed,
                "cases_total":        run.total_cases,
                "cases_passed":       run.cases_passed,
            },
            "subroutines": [
                {
                    "subroutine":         r.subroutine_name,
                    "module":             r.module_name,
                    "golden_file":        str(r.golden_file),
                    "test_file":          str(r.test_file),
                    "status":             r.status,
                    "used_ftest_interface": r.used_ftest_interface,
                    "cases_total":        r.cases_total,
                    "cases_passed":       r.cases_passed,
                    "cases_failed":       r.cases_failed,
                    "pytest_returncode":  r.pytest_returncode,
                    **({"generation_error": r.generation_error} if r.generation_error else {}),
                }
                for r in run.subroutine_results
            ],
        }
        report_path = run.output_dir / "parity_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        console.print(f"[dim]Report written to {report_path}[/dim]")
