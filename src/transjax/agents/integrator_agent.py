"""
IntegratorAgent — build and test the system integration for a fully
translated JAX/Python Earth System Model.

Given:
  • A directory of Fortran source files (read-only reference)
  • A directory of translated JAX/Python modules

The agent:
  1. Scans the Fortran codebase to understand the model's call structure.
  2. Collects public function signatures from all translated JAX modules.
  3. Asks Claude to generate model_run.py, test_integration.py, and
     System_integration_README.md.
  4. Writes the generated files to <output_dir>/integration/.
  5. Runs the integration test (pytest test_integration.py).
  6. If the test fails, delegates to IntegrationRepairAgent for iterative
     repair until the test passes or the repair limit is reached.

Output directory layout
-----------------------
<output_dir>/
└── integration/
    ├── model_run.py
    ├── test_integration.py
    └── docs/
        ├── System_integration_README.md
        └── integration_repair_PASS|FAIL.md  (if repair was needed)
"""

import ast
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel

from transjax.agents.base_agent import BaseAgent
from transjax.agents.integration_repair_agent import (
    IntegrationRepairAgent,
    IntegrationRepairResult,
)
from transjax.agents.prompts.integration_prompts import (
    INTEGRATION_BUILD_PROMPT,
    INTEGRATOR_SYSTEM_PROMPT,
)
from transjax.agents.utils.config_loader import get_llm_config

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IntegrationResult:
    """Outcome of the full integration build-and-test run."""
    integration_dir: Path
    fortran_dir: Path
    jax_src_dir: Path
    final_status: str = "unknown"       # "PASS" | "FAIL" | "SKIPPED"
    model_run_path: Optional[Path] = None
    test_path: Optional[Path] = None
    readme_path: Optional[Path] = None
    repair_result: Optional[IntegrationRepairResult] = None
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.final_status == "PASS"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class IntegratorAgent(BaseAgent):
    """
    Builds and validates the Python system integration for a translated ESM.

    Generates model_run.py (driver), test_integration.py (pytest wrapper),
    and System_integration_README.md by asking Claude to mirror the Fortran
    model's call sequence using the translated JAX modules.  Delegates
    iterative repair to IntegrationRepairAgent when the initial test fails.
    """

    DEFAULT_MAX_REPAIR_ITERATIONS = 5

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        llm_config = get_llm_config()
        super().__init__(
            name="IntegratorAgent",
            role=(
                "Expert Earth System Model software engineer who understands how "
                "individual ESM components are assembled into a running simulation "
                "and can faithfully reproduce that assembly in Python/JAX."
            ),
            model=model or llm_config.get("model", "claude-sonnet-4-6"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 48000),
        )

    # ---------------------------------------------------------------------- #
    # Public entry point                                                       #
    # ---------------------------------------------------------------------- #

    def run(
        self,
        fortran_dir: Path,
        jax_src_dir: Path,
        output_dir: Path,
        gcm_model_name: str = "generic ESM",
        max_repair_iterations: int = DEFAULT_MAX_REPAIR_ITERATIONS,
        verbose: bool = False,
    ) -> IntegrationResult:
        """
        Build and test the system integration.

        Args:
            fortran_dir:          Root of the Fortran source tree (read-only).
            jax_src_dir:          Directory of translated JAX modules.
            output_dir:           Root output directory.  Integration files are
                                  written to <output_dir>/integration/.
            gcm_model_name:       ESM name injected into prompts and docs.
            max_repair_iterations: Maximum IntegrationRepairAgent iterations if
                                  the initial test fails.
            verbose:              Print full subprocess output.

        Returns:
            IntegrationResult.
        """
        fortran_dir  = Path(fortran_dir).resolve()
        jax_src_dir  = Path(jax_src_dir).resolve()
        output_dir   = Path(output_dir).resolve()

        integration_dir = output_dir / "integration"
        integration_dir.mkdir(parents=True, exist_ok=True)
        (integration_dir / "docs").mkdir(exist_ok=True)

        result = IntegrationResult(
            integration_dir=integration_dir,
            fortran_dir=fortran_dir,
            jax_src_dir=jax_src_dir,
        )

        console.print(
            Panel(
                f"[bold cyan]IntegratorAgent[/bold cyan]\n"
                f"  Model : {gcm_model_name}\n"
                f"  JAX src : {jax_src_dir}\n"
                f"  Output  : {integration_dir}",
                expand=False,
            )
        )

        # Step 1 — gather context
        console.print("  [cyan]Scanning Fortran codebase structure…[/cyan]")
        fortran_summary = self._scan_fortran_structure(fortran_dir)

        console.print("  [cyan]Collecting JAX module interfaces…[/cyan]")
        module_interfaces = self._collect_module_interfaces(jax_src_dir)
        n_modules = len(list(jax_src_dir.glob("*.py")))

        if n_modules == 0:
            result.final_status = "FAIL"
            result.error = f"No translated Python modules found in {jax_src_dir}"
            console.print(f"  [red]✗ {result.error}[/red]")
            return result

        # Step 2 — ask Claude to build the integration
        console.print("  [cyan]Asking Claude to build the system integration…[/cyan]")
        prompt = INTEGRATION_BUILD_PROMPT.format(
            gcm_model_name=gcm_model_name,
            jax_src_dir=str(jax_src_dir),
            fortran_structure_summary=fortran_summary,
            module_interfaces=module_interfaces,
            n_modules=n_modules,
        )

        response = self.query_claude(
            prompt=prompt,
            system_prompt=INTEGRATOR_SYSTEM_PROMPT,
        )

        # Step 3 — parse and write the generated files
        model_run_code, test_code, readme_md = self._parse_response(response)

        if not model_run_code:
            result.final_status = "FAIL"
            result.error = "Claude did not return model_run.py content"
            console.print(f"  [red]✗ {result.error}[/red]")
            return result

        model_run_path = integration_dir / "model_run.py"
        test_path      = integration_dir / "test_integration.py"
        readme_path    = integration_dir / "docs" / "System_integration_README.md"

        model_run_path.write_text(model_run_code)
        console.print(f"  [dim]  Wrote: {model_run_path}[/dim]")

        if test_code:
            test_path.write_text(test_code)
            console.print(f"  [dim]  Wrote: {test_path}[/dim]")

        if readme_md:
            readme_path.write_text(readme_md)
            console.print(f"  [dim]  Wrote: {readme_path}[/dim]")

        result.model_run_path = model_run_path
        result.test_path      = test_path if test_code else None
        result.readme_path    = readme_path if readme_md else None

        if not test_code:
            console.print("  [yellow]⚠ No test_integration.py generated; skipping test run.[/yellow]")
            result.final_status = "SKIPPED"
            return result

        # Step 4 — run the integration test
        console.print("  [cyan]Running integration test…[/cyan]")
        returncode, output = self._run_integration_test(integration_dir, verbose)

        if returncode == 0:
            console.print("  [bold green]✓ Integration test passed![/bold green]")
            result.final_status = "PASS"
            return result

        console.print(f"  [red]✗ Integration test failed (exit {returncode})[/red]")
        if verbose:
            console.print(output)

        # Step 5 — delegate to IntegrationRepairAgent
        console.print(
            f"\n  [cyan]Invoking IntegrationRepairAgent "
            f"(max {max_repair_iterations} iterations)…[/cyan]"
        )
        repair_agent = IntegrationRepairAgent(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        repair_result = repair_agent.run(
            integration_dir=integration_dir,
            jax_src_dir=jax_src_dir,
            gcm_model_name=gcm_model_name,
            max_iterations=max_repair_iterations,
            verbose=verbose,
        )

        result.repair_result  = repair_result
        result.final_status   = repair_result.final_status

        if result.passed:
            console.print("  [bold green]✓ Integration repair succeeded![/bold green]")
        else:
            console.print(
                f"  [bold red]✗ Integration repair did not converge after "
                f"{repair_result.total_iterations} iteration(s).[/bold red]"
            )

        return result

    # ---------------------------------------------------------------------- #
    # Fortran structure scanner                                                #
    # ---------------------------------------------------------------------- #

    def _scan_fortran_structure(self, fortran_dir: Path) -> str:
        """
        Produce a concise summary of the Fortran codebase structure.

        Identifies likely driver/program files and lists all modules with
        their public subroutine/function names.  Uses plain text parsing
        (no Fortran compiler or parser needed).
        """
        f_extensions = {".f90", ".F90", ".f", ".F", ".for", ".FOR", ".f95", ".F95"}
        fortran_files = sorted(
            f for f in fortran_dir.rglob("*") if f.suffix in f_extensions
        )

        if not fortran_files:
            return "(no Fortran source files found)"

        sections: List[str] = [
            f"Fortran source root: {fortran_dir}",
            f"Total source files : {len(fortran_files)}",
            "",
        ]

        # Identify driver files (contain PROGRAM statement)
        drivers: List[str] = []
        module_summaries: List[str] = []

        for ffile in fortran_files:
            try:
                text = ffile.read_text(errors="replace")
            except OSError:
                continue

            lines = text.splitlines()
            rel = ffile.relative_to(fortran_dir)

            # Detect PROGRAM blocks
            is_driver = any(
                re.match(r"^\s*PROGRAM\s+\w+", ln, re.IGNORECASE) for ln in lines
            )
            if is_driver:
                drivers.append(str(rel))

            # Extract MODULE name
            module_match = None
            for ln in lines:
                m = re.match(r"^\s*MODULE\s+(\w+)", ln, re.IGNORECASE)
                if m and not re.match(r"^\s*MODULE\s+PROCEDURE", ln, re.IGNORECASE):
                    module_match = m.group(1)
                    break

            if module_match is None:
                continue

            # Extract SUBROUTINE / FUNCTION names from this module
            routines: List[str] = []
            for ln in lines:
                sub = re.match(
                    r"^\s*(?:(?:PURE|ELEMENTAL|RECURSIVE)\s+)?"
                    r"(?:SUBROUTINE|FUNCTION)\s+(\w+)\s*\(",
                    ln, re.IGNORECASE,
                )
                if sub:
                    routines.append(sub.group(1))

            summary = f"  MODULE {module_match}  ({rel})"
            if routines:
                summary += "\n    " + "  ".join(routines[:12])
                if len(routines) > 12:
                    summary += f"  … (+{len(routines) - 12} more)"
            module_summaries.append(summary)

        if drivers:
            sections.append("Driver / Program files:")
            for d in drivers[:10]:
                sections.append(f"  {d}")
            sections.append("")

        sections.append(f"Modules ({len(module_summaries)}):")
        sections.extend(module_summaries[:50])
        if len(module_summaries) > 50:
            sections.append(f"  … (+{len(module_summaries) - 50} more modules)")

        return "\n".join(sections)

    # ---------------------------------------------------------------------- #
    # JAX module interface collector                                           #
    # ---------------------------------------------------------------------- #

    def _collect_module_interfaces(self, jax_src_dir: Path) -> str:
        """
        Collect public function signatures from all translated modules via AST.
        """
        py_files = sorted(jax_src_dir.glob("*.py"))
        if not py_files:
            return "(no translated modules found)"

        sections: List[str] = []
        for py_file in py_files:
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                sections.append(f"### {py_file.name}\n  (syntax error — could not parse)")
                continue

            funcs: List[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    args = [a.arg for a in node.args.args]
                    funcs.append(f"  def {node.name}({', '.join(args)})")

            if funcs:
                sections.append(f"### {py_file.stem}\n" + "\n".join(funcs))
            else:
                sections.append(f"### {py_file.stem}\n  (no public functions found)")

        return "\n\n".join(sections)

    # ---------------------------------------------------------------------- #
    # Response parsing                                                         #
    # ---------------------------------------------------------------------- #

    def _parse_response(self, response: str) -> Tuple[str, str, str]:
        """
        Extract model_run.py, test_integration.py, and README from Claude's response.

        Returns:
            Tuple of (model_run_code, test_code, readme_md).
            Any missing section is returned as an empty string.
        """
        model_run_code = self._extract_section_code(response, "model_run.py")
        test_code      = self._extract_section_code(response, "test_integration.py")
        readme_md      = self._extract_section_markdown(response, "System_integration_README.md")

        return model_run_code, test_code, readme_md

    def _extract_section_code(self, response: str, section_name: str) -> str:
        """Extract a ```python ... ``` block under a ### <section_name> heading."""
        pattern = rf"###\s+{re.escape(section_name)}\s*\n(.*?)(?=###|\Z)"
        m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if not m:
            return ""
        body = m.group(1)
        # Extract code block
        for tag in ("```python", "```"):
            if tag in body:
                start = body.find(tag) + len(tag)
                end   = body.find("```", start)
                if end != -1:
                    return body[start:end].strip()
        return body.strip()

    def _extract_section_markdown(self, response: str, section_name: str) -> str:
        """Extract a ```markdown ... ``` block under a ### <section_name> heading."""
        pattern = rf"###\s+{re.escape(section_name)}\s*\n(.*?)(?=###|\Z)"
        m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if not m:
            return ""
        body = m.group(1)
        for tag in ("```markdown", "```"):
            if tag in body:
                start = body.find(tag) + len(tag)
                end   = body.find("```", start)
                if end != -1:
                    return body[start:end].strip()
        return body.strip()

    # ---------------------------------------------------------------------- #
    # Test runner                                                              #
    # ---------------------------------------------------------------------- #

    def _run_integration_test(
        self, integration_dir: Path, verbose: bool
    ) -> Tuple[int, str]:
        """Run pytest test_integration.py; return (returncode, combined output)."""
        test_file = integration_dir / "test_integration.py"
        if not test_file.exists():
            return 1, "test_integration.py not found"

        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v", "--tb=short", "--no-header",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(integration_dir),
            )
            return proc.returncode, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            return 1, "Integration test timed out after 300 s"
        except Exception as exc:
            return 1, f"Failed to run pytest: {exc}"
