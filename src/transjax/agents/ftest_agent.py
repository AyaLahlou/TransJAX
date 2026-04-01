"""
FtestAgent — Functional testing framework builder for Fortran ESM codebases.

Generates thin Fortran test-driver programs and Python pytest suites that
compile and run them, enabling isolated subroutine testing on HPC systems
(nvfortran + NetCDF).

Output layout
-------------
<output_dir>/
├── drivers/
│   ├── test_<subroutine>.f90   # Fortran driver: namelist in → KEY=VALUE out
│   └── bin/                    # Compiled binaries (populated by `make all`)
├── tests/
│   ├── conftest.py             # run_driver() helper used by all test files
│   └── test_<subroutine>.py   # pytest file per subroutine
├── Makefile                    # Compiles drivers against the existing model build
└── ftest_report.json           # Summary of generated tests
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from transjax.agents.base_agent import BaseAgent
from transjax.agents.prompts.ftest_prompts import (
    FTEST_ANALYZE_SUBROUTINE_PROMPT,
    FTEST_DRIVER_PROMPT,
    FTEST_PYTEST_PROMPT,
    FTEST_SYSTEM_PROMPT,
)
from transjax.agents.utils.config_loader import get_llm_config

logger = logging.getLogger(__name__)
console = Console()

# Maximum Fortran source characters sent to Claude for interface analysis.
_MAX_SOURCE_CHARS = 8000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubroutineInfo:
    """A subroutine discovered in the Fortran codebase."""
    name: str
    module_name: str
    file_path: Path
    source_code: str
    interface: Optional[Dict[str, Any]] = None


@dataclass
class FtestResult:
    """Generated test artefacts for a single subroutine."""
    subroutine_name: str
    module_name: str
    driver_code: str
    pytest_code: str
    driver_path: Path
    pytest_path: Path
    status: str = "ok"          # "ok" | "skipped" | "error"
    error_message: str = ""
    interface: Optional[Dict[str, Any]] = None  # Parsed calling interface (for golden)


@dataclass
class FtestFrameworkResult:
    """Aggregated outcome of a full Ftest run."""
    fortran_dir: Path
    output_dir: Path
    subroutines_found: int = 0
    tests_generated: int = 0
    skipped: int = 0
    errors: int = 0
    results: List[FtestResult] = field(default_factory=list)
    makefile_path: Optional[Path] = None
    conftest_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FtestAgent(BaseAgent):
    """
    Builds a functional testing framework for Fortran ESM subroutines.

    For each subroutine found in the codebase the agent:
    1. Extracts the calling interface (via Claude).
    2. Generates a thin Fortran test-driver (namelist → KEY=VALUE).
    3. Generates a Python pytest file that compiles/runs the driver.

    Drivers link against the *existing* model build — no separate compilation
    of the model is required.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        llm_config = get_llm_config()
        super().__init__(
            name="FtestAgent",
            role=(
                "Expert functional test suite builder for Fortran Earth System Model "
                "codebases.  Generates thin Fortran test-driver programs and Python "
                "pytest suites that compile and run them in isolation on HPC systems "
                "(nvfortran + NetCDF)."
            ),
            model=model or llm_config.get("model", "claude-sonnet-4-6"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 48000),
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        fortran_dir: Path,
        output_dir: Path,
        build_dir: Optional[Path] = None,
        compiler: str = "nvfortran",
        netcdf_inc: Optional[str] = None,
        netcdf_lib: Optional[str] = None,
        module_filter: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> FtestFrameworkResult:
        """
        Build the full Ftest framework for a Fortran codebase.

        Args:
            fortran_dir:   Root of the Fortran source tree.
            output_dir:    Where to write drivers/, tests/, Makefile, report.
            build_dir:     Existing model build directory containing .o files
                           that drivers should link against.
            compiler:      Fortran compiler command (default: nvfortran).
            netcdf_inc:    Path to NetCDF include directory (-I flag).
            netcdf_lib:    Path to NetCDF lib directory (-L flag).
            module_filter: If given, only process modules whose names appear
                           in this list (case-insensitive).
            verbose:       Enable DEBUG-level logging.

        Returns:
            FtestFrameworkResult summarising what was generated.
        """
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        fw_result = FtestFrameworkResult(
            fortran_dir=fortran_dir,
            output_dir=output_dir,
        )

        # Create output directory tree
        drivers_dir = output_dir / "drivers"
        bin_dir = drivers_dir / "bin"
        tests_dir = output_dir / "tests"
        for d in (drivers_dir, bin_dir, tests_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 1 — discover subroutines
        console.print(f"[cyan]Scanning:[/cyan] {fortran_dir}")
        subroutines = self._scan_subroutines(fortran_dir, module_filter)
        fw_result.subroutines_found = len(subroutines)
        console.print(
            f"[green]Found {len(subroutines)} subroutine(s) across "
            f"{len({s.module_name for s in subroutines})} module(s)[/green]"
        )

        if not subroutines:
            console.print("[yellow]No subroutines found — check the path.[/yellow]")
            return fw_result

        # 2 — generate test artefacts
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "Generating Ftest framework...", total=len(subroutines)
            )
            for sub in subroutines:
                progress.update(
                    task,
                    description=f"[cyan]{sub.module_name}[/cyan]::{sub.name}",
                )
                ftest_res = self._generate_ftest(sub, drivers_dir, tests_dir)
                fw_result.results.append(ftest_res)
                if ftest_res.status == "ok":
                    fw_result.tests_generated += 1
                elif ftest_res.status == "skipped":
                    fw_result.skipped += 1
                else:
                    fw_result.errors += 1
                progress.advance(task)

        # 3 — write Makefile
        makefile_path = output_dir / "Makefile"
        self._write_makefile(
            makefile_path,
            fw_result.results,
            drivers_dir,
            bin_dir,
            build_dir,
            compiler,
            netcdf_inc,
            netcdf_lib,
        )
        fw_result.makefile_path = makefile_path

        # 4 — write conftest.py
        conftest_path = tests_dir / "conftest.py"
        self._write_conftest(conftest_path, bin_dir)
        fw_result.conftest_path = conftest_path

        # 5 — write JSON report
        self._write_report(output_dir / "ftest_report.json", fw_result)

        return fw_result

    # -----------------------------------------------------------------------
    # Fortran scanning
    # -----------------------------------------------------------------------

    def _scan_subroutines(
        self,
        fortran_dir: Path,
        module_filter: Optional[List[str]],
    ) -> List[SubroutineInfo]:
        """Walk *fortran_dir* and collect subroutine definitions."""
        fortran_exts = {".f90", ".f95", ".f03", ".f08", ".F90", ".F95", ".f", ".FOR"}
        subroutines: List[SubroutineInfo] = []

        for f_path in sorted(fortran_dir.rglob("*")):
            if f_path.suffix not in fortran_exts:
                continue
            try:
                source = f_path.read_text(errors="replace")
            except OSError as exc:
                logger.warning("Cannot read %s: %s", f_path, exc)
                continue

            module_name = self._extract_module_name(source) or f_path.stem

            if module_filter:
                lower_filter = [m.lower() for m in module_filter]
                if module_name.lower() not in lower_filter:
                    continue

            found = self._extract_subroutines(source, module_name, f_path)
            if found:
                logger.debug("  %s: %d subroutine(s)", f_path.name, len(found))
            subroutines.extend(found)

        return subroutines

    _MODULE_RE = re.compile(r"^\s*module\s+(\w+)\s*$", re.IGNORECASE | re.MULTILINE)
    _SUB_START_RE = re.compile(
        r"^\s*(?:(?:pure|elemental|recursive)\s+)*subroutine\s+(\w+)\s*(?:\(|$)",
        re.IGNORECASE,
    )
    _SUB_END_RE = re.compile(r"^\s*end\s+subroutine\b", re.IGNORECASE)

    def _extract_module_name(self, source: str) -> Optional[str]:
        m = self._MODULE_RE.search(source)
        return m.group(1) if m else None

    def _extract_subroutines(
        self,
        source: str,
        module_name: str,
        file_path: Path,
    ) -> List[SubroutineInfo]:
        lines = source.splitlines()
        results: List[SubroutineInfo] = []
        i = 0
        while i < len(lines):
            m = self._SUB_START_RE.match(lines[i])
            if m:
                sub_name = m.group(1)
                depth = 1
                j = i + 1
                while j < len(lines) and depth > 0:
                    if self._SUB_START_RE.match(lines[j]):
                        depth += 1
                    if self._SUB_END_RE.match(lines[j]):
                        depth -= 1
                    j += 1
                results.append(SubroutineInfo(
                    name=sub_name,
                    module_name=module_name,
                    file_path=file_path,
                    source_code="\n".join(lines[i:j]),
                ))
                i = j
            else:
                i += 1
        return results

    # -----------------------------------------------------------------------
    # Code generation (per subroutine)
    # -----------------------------------------------------------------------

    def _generate_ftest(
        self,
        sub: SubroutineInfo,
        drivers_dir: Path,
        tests_dir: Path,
    ) -> FtestResult:
        driver_path = drivers_dir / f"test_{sub.name}.f90"
        pytest_path = tests_dir / f"test_{sub.name}.py"

        try:
            # A — parse interface via Claude
            sub.interface = self._analyze_interface(sub)

            # B — generate Fortran driver
            driver_code = self._generate_driver(sub)

            # C — generate Python pytest
            pytest_code = self._generate_pytest(sub)

            driver_path.write_text(driver_code)
            pytest_path.write_text(pytest_code)

            console.print(
                f"  [green]✓[/green] {sub.name}  "
                f"[dim]({driver_path.name}, {pytest_path.name})[/dim]"
            )
            return FtestResult(
                subroutine_name=sub.name,
                module_name=sub.module_name,
                driver_code=driver_code,
                pytest_code=pytest_code,
                driver_path=driver_path,
                pytest_path=pytest_path,
                status="ok",
                interface=sub.interface,
            )

        except Exception as exc:
            logger.error("Failed to generate ftest for %s: %s", sub.name, exc)
            console.print(f"  [red]✗[/red] {sub.name}: {exc}")
            return FtestResult(
                subroutine_name=sub.name,
                module_name=sub.module_name,
                driver_code="",
                pytest_code="",
                driver_path=driver_path,
                pytest_path=pytest_path,
                status="error",
                error_message=str(exc),
            )

    def _analyze_interface(self, sub: SubroutineInfo) -> Dict[str, Any]:
        prompt = FTEST_ANALYZE_SUBROUTINE_PROMPT.format(
            fortran_code=sub.source_code[:_MAX_SOURCE_CHARS],
            module_name=sub.module_name,
            file_path=str(sub.file_path),
        )
        response = self.query_claude(
            prompt, system_prompt=FTEST_SYSTEM_PROMPT, max_tokens=4096
        )
        return self._extract_json(response)

    def _generate_driver(self, sub: SubroutineInfo) -> str:
        prompt = FTEST_DRIVER_PROMPT.format(
            subroutine_name=sub.name,
            module_name=sub.module_name,
            interface_json=json.dumps(sub.interface, indent=2),
        )
        response = self.query_claude(
            prompt, system_prompt=FTEST_SYSTEM_PROMPT, max_tokens=4096
        )
        return self._extract_code_block(response, "fortran") or response.strip()

    def _generate_pytest(self, sub: SubroutineInfo) -> str:
        prompt = FTEST_PYTEST_PROMPT.format(
            subroutine_name=sub.name,
            module_name=sub.module_name,
            interface_json=json.dumps(sub.interface, indent=2),
        )
        response = self.query_claude(
            prompt, system_prompt=FTEST_SYSTEM_PROMPT, max_tokens=6144
        )
        return self._extract_code_block(response, "python") or response.strip()

    # -----------------------------------------------------------------------
    # Static file generation
    # -----------------------------------------------------------------------

    def _write_makefile(
        self,
        makefile_path: Path,
        results: List[FtestResult],
        drivers_dir: Path,
        bin_dir: Path,
        build_dir: Optional[Path],
        compiler: str,
        netcdf_inc: Optional[str],
        netcdf_lib: Optional[str],
    ) -> None:
        ok = [r for r in results if r.status == "ok"]

        inc_flags = f" -I{netcdf_inc}" if netcdf_inc else ""
        lib_flags = f" -L{netcdf_lib} -lnetcdff -lnetcdf" if netcdf_lib else ""

        build_objs_line = (
            f"BUILD_OBJS := $(wildcard $(BUILD_DIR)/*.o)\n"
            if build_dir else
            "# BUILD_OBJS: set BUILD_DIR to the model build directory\nBUILD_OBJS :=\n"
        )
        build_dir_default = str(build_dir) if build_dir else "./model_build"

        all_targets = " \\\n    ".join(
            f"$(BIN_DIR)/test_{r.subroutine_name}" for r in ok
        )

        per_target = "\n".join(
            f"$(BIN_DIR)/test_{r.subroutine_name}: $(DRV_DIR)/test_{r.subroutine_name}.f90 | $(BIN_DIR)\n"
            f"\t$(FC) $(FCFLAGS) -o $@ $< $(BUILD_OBJS) $(LDFLAGS)\n"
            for r in ok
        )

        makefile = (
            f"# Ftest Makefile — generated by TransJAX FtestAgent\n"
            f"#\n"
            f"# Usage:\n"
            f"#   make all          — compile all Fortran test drivers\n"
            f"#   make clean        — remove compiled binaries\n"
            f"#   make test         — run pytest suite (requires `make all` first)\n"
            f"#\n"
            f"# Override on the command line, e.g.:\n"
            f"#   make FC=gfortran BUILD_DIR=/path/to/build NETCDF_INC=/usr/include\n"
            f"\n"
            f"FC        ?= {compiler}\n"
            f"FCFLAGS   ?= -O0 -g{inc_flags}\n"
            f"LDFLAGS   ?= {lib_flags.strip()}\n"
            f"BUILD_DIR ?= {build_dir_default}\n"
            f"DRV_DIR   := {drivers_dir.name}\n"
            f"BIN_DIR   := {bin_dir.relative_to(makefile_path.parent)}\n"
            f"\n"
            f"{build_objs_line}"
            f"\n"
            f".PHONY: all clean test\n"
            f"\n"
            f"all: {all_targets}\n"
            f"\n"
            f"$(BIN_DIR):\n"
            f"\tmkdir -p $(BIN_DIR)\n"
            f"\n"
            f"{per_target}\n"
            f"clean:\n"
            f"\trm -f $(BIN_DIR)/test_*\n"
            f"\n"
            f"test: all\n"
            f"\tpython -m pytest tests/ -v\n"
        )

        makefile_path.write_text(makefile)
        console.print(f"[green]Makefile:[/green]  {makefile_path}")

    def _write_conftest(self, conftest_path: Path, bin_dir: Path) -> None:
        """Write conftest.py providing the run_driver() helper to all test files."""
        conftest = '''\
"""
conftest.py — Ftest pytest helpers (generated by TransJAX FtestAgent).

Provides ``run_driver``, used by every test module in this directory.
Drivers must be compiled before running tests::

    cd <ftest_output>
    make all
    python -m pytest tests/ -v
"""

import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest

# Directory that contains compiled driver binaries.
# Override with the FTEST_DRIVER_DIR environment variable.
_DEFAULT_BIN_DIR = Path(__file__).parent.parent / "drivers" / "bin"
FTEST_DRIVER_DIR = Path(os.environ.get("FTEST_DRIVER_DIR", str(_DEFAULT_BIN_DIR)))


def run_driver(driver_exe: Path, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a compiled Fortran test-driver and return its outputs.

    Constructs a Fortran NAMELIST string from *inputs*, pipes it to the driver
    on stdin, and parses ``KEY=VALUE`` lines from stdout into a Python dict.

    Args:
        driver_exe: Path to the compiled driver binary.
        inputs:     Dict of NAMELIST variable names → Python values.

    Returns:
        Dict of output variable names → coerced Python values
        (int, float, bool, or str).

    Raises:
        ``pytest.skip`` if the binary does not exist.
        ``pytest.fail``  if the driver exits with a non-zero status.
    """
    if not driver_exe.exists():
        pytest.skip(
            f"Driver binary not found: {driver_exe}\\n"
            "Compile with `make all` in the ftest output directory."
        )

    # Build Fortran NAMELIST block
    lines = ["&inputs"]
    for key, val in inputs.items():
        if isinstance(val, bool):
            lines.append(f"  {key} = {'T' if val else 'F'},")
        elif isinstance(val, str):
            lines.append(f"  {key} = '{val}',")
        else:
            lines.append(f"  {key} = {val},")
    lines.append("/")
    namelist_input = "\\n".join(lines) + "\\n"

    proc = subprocess.run(
        [str(driver_exe)],
        input=namelist_input,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if proc.returncode != 0:
        pytest.fail(
            f"Driver {driver_exe.name} exited {proc.returncode}\\n"
            f"stderr: {proc.stderr.strip()}"
        )

    # Parse KEY=VALUE output
    outputs: Dict[str, Any] = {}
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, _, raw = line.partition("=")
        key = key.strip()
        raw = raw.strip()
        try:
            outputs[key] = int(raw)
        except ValueError:
            try:
                outputs[key] = float(raw)
            except ValueError:
                upper = raw.upper()
                if upper in ("T", ".TRUE.", "TRUE"):
                    outputs[key] = True
                elif upper in ("F", ".FALSE.", "FALSE"):
                    outputs[key] = False
                else:
                    outputs[key] = raw

    return outputs
'''
        conftest_path.write_text(conftest)
        console.print(f"[green]conftest.py:[/green] {conftest_path}")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------

    def _write_report(
        self, report_path: Path, fw_result: FtestFrameworkResult
    ) -> None:
        report = {
            "fortran_dir": str(fw_result.fortran_dir),
            "output_dir": str(fw_result.output_dir),
            "summary": {
                "subroutines_found": fw_result.subroutines_found,
                "tests_generated": fw_result.tests_generated,
                "skipped": fw_result.skipped,
                "errors": fw_result.errors,
            },
            "subroutines": [
                {
                    "name": r.subroutine_name,
                    "module": r.module_name,
                    "status": r.status,
                    "driver": str(r.driver_path),
                    "pytest": str(r.pytest_path),
                    **({"interface": r.interface} if r.interface else {}),
                    **({"error": r.error_message} if r.error_message else {}),
                }
                for r in fw_result.results
            ],
        }
        report_path.write_text(json.dumps(report, indent=2))
        console.print(f"[green]Report:[/green]    {report_path}")

    # -----------------------------------------------------------------------
    # Parsing helpers
    # -----------------------------------------------------------------------

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract the first JSON object from a Claude response."""
        # 1. ```json … ``` block
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        # 2. Bare JSON starting from first {
        stripped = response.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        # 3. Find first balanced { … } block
        start = response.find("{")
        if start != -1:
            depth = 0
            for i, ch in enumerate(response[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response[start : i + 1])
                        except json.JSONDecodeError:
                            break
        raise ValueError(
            f"Could not extract JSON from Claude response "
            f"(first 400 chars):\n{response[:400]}"
        )

    def _extract_code_block(self, response: str, language: str) -> Optional[str]:
        """Extract the first fenced code block of *language* from *response*."""
        m = re.search(
            rf"```{re.escape(language)}\s*(.*?)\s*```",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        # Fallback: any fenced block
        m = re.search(r"```\w*\s*(.*?)\s*```", response, re.DOTALL)
        return m.group(1).strip() if m else None
