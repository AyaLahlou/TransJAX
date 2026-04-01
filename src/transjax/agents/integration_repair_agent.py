"""
IntegrationRepairAgent — iterative debugging of a translated ESM integration.

Given a failing integration test, the agent:
  1. Reads the error log and current integration/module source files.
  2. Asks Claude to diagnose the failure and return corrected file(s).
  3. Writes the corrected files.
  4. Re-runs the integration test.
  5. Repeats until the test passes OR the iteration limit is reached.

Files the agent may modify
--------------------------
  • <integration_dir>/model_run.py
  • <integration_dir>/test_integration.py
  • <jax_src_dir>/<module>.py   (only for genuine translation bugs)

Files that are always read-only
--------------------------------
  • Fortran source
  • Golden data JSON files
  • Parity test files
"""

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
from transjax.agents.prompts.integration_repair_prompts import (
    INTEGRATION_REPAIR_PROMPT,
    INTEGRATION_REPAIR_SYSTEM_PROMPT,
)
from transjax.agents.utils.config_loader import get_llm_config

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RepairIteration:
    """Record of one integration repair iteration."""
    iteration: int
    root_cause: str = ""
    fix_strategy: str = ""
    files_changed: List[str] = field(default_factory=list)
    test_returncode: int = -1
    error_snippet: str = ""

    @property
    def passed(self) -> bool:
        return self.test_returncode == 0


@dataclass
class IntegrationRepairResult:
    """Aggregated outcome of the full integration repair run."""
    integration_dir: Path
    jax_src_dir: Path
    final_status: str = "unknown"      # "PASS" | "FAIL"
    iterations: List[RepairIteration] = field(default_factory=list)
    report_path: Optional[Path] = None

    @property
    def total_iterations(self) -> int:
        return len(self.iterations)

    @property
    def passed(self) -> bool:
        return self.final_status == "PASS"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class IntegrationRepairAgent(BaseAgent):
    """
    Iteratively repairs a failing system integration for a translated ESM.

    The agent reads error logs from the integration test, asks Claude to
    diagnose and fix the problem, writes the corrected files, and re-runs
    the test until all assertions pass or the iteration limit is reached.
    """

    DEFAULT_MAX_ITERATIONS = 5

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        llm_config = get_llm_config()
        super().__init__(
            name="IntegrationRepairAgent",
            role=(
                "Senior research software engineer with deep expertise in Earth "
                "System Model codebases and Python/JAX translations. Diagnoses and "
                "fixes system integration failures iteratively."
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
        integration_dir: Path,
        jax_src_dir: Path,
        gcm_model_name: str = "generic ESM",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        verbose: bool = False,
    ) -> IntegrationRepairResult:
        """
        Repair the integration until tests pass or the limit is reached.

        Args:
            integration_dir:  Directory containing model_run.py and
                              test_integration.py.
            jax_src_dir:      Directory of translated JAX modules (may be
                              modified if a module-level bug is found).
            gcm_model_name:   ESM name for prompt context.
            max_iterations:   Maximum repair attempts.
            verbose:          Print full error output.

        Returns:
            IntegrationRepairResult with per-iteration logs.
        """
        integration_dir = Path(integration_dir).resolve()
        jax_src_dir     = Path(jax_src_dir).resolve()

        result = IntegrationRepairResult(
            integration_dir=integration_dir,
            jax_src_dir=jax_src_dir,
        )

        model_run_file   = integration_dir / "model_run.py"
        test_file        = integration_dir / "test_integration.py"

        # Pre-flight: make sure we have files to work with
        if not model_run_file.exists():
            logger.error("model_run.py not found in %s", integration_dir)
            result.final_status = "FAIL"
            return result

        module_interfaces = self._collect_module_interfaces(jax_src_dir)

        for i in range(1, max_iterations + 1):
            console.print(f"\n[bold cyan]━━ Integration Repair Iteration {i}/{max_iterations} ━━[/bold cyan]")
            iter_log = RepairIteration(iteration=i)

            # Run the integration test and collect errors
            returncode, error_log = self._run_integration(integration_dir, verbose)
            if returncode == 0:
                console.print("[bold green]✓ Integration test passed![/bold green]")
                result.final_status = "PASS"
                result.report_path = self._write_report(result)
                return result

            iter_log.error_snippet = error_log[:4000]
            console.print(f"  [red]Test failed (exit {returncode})[/red]")
            if verbose:
                console.print(error_log)

            # Build prompt
            model_run_source = model_run_file.read_text() if model_run_file.exists() else "(missing)"
            test_source      = test_file.read_text()      if test_file.exists()      else "(missing)"

            prompt = INTEGRATION_REPAIR_PROMPT.format(
                gcm_model_name=gcm_model_name,
                iteration=i,
                max_iterations=max_iterations,
                error_log=error_log[:6000] + ("\n… (truncated)" if len(error_log) > 6000 else ""),
                model_run_source=model_run_source,
                test_source=test_source,
                module_interfaces=module_interfaces,
            )

            console.print(f"  [cyan]Asking Claude to diagnose and fix the integration…[/cyan]")
            response = self.query_claude(
                prompt=prompt,
                system_prompt=INTEGRATION_REPAIR_SYSTEM_PROMPT,
            )

            iter_log.root_cause  = self._extract_section(response, "ROOT CAUSE")
            iter_log.fix_strategy = self._extract_section(response, "FIX STRATEGY")
            console.print(f"  [dim]Root cause:[/dim] {iter_log.root_cause[:120]}")
            console.print(f"  [dim]Fix strategy:[/dim] {iter_log.fix_strategy[:120]}")

            # Write corrected files
            changed = self._apply_changes(response, integration_dir, jax_src_dir)
            iter_log.files_changed = changed

            if changed:
                console.print(f"  [green]✓ Updated {len(changed)} file(s): {', '.join(changed)}[/green]")
            else:
                console.print("  [yellow]⚠ No file changes extracted from response[/yellow]")

            iter_log.test_returncode = returncode  # still the pre-fix result
            result.iterations.append(iter_log)

        # Final check after loop
        returncode, _ = self._run_integration(integration_dir, verbose=False)
        result.final_status = "PASS" if returncode == 0 else "FAIL"

        if result.final_status == "FAIL":
            console.print(
                f"\n[bold red]✗ Integration repair did not converge after "
                f"{max_iterations} iteration(s).[/bold red]"
            )
        result.report_path = self._write_report(result)
        return result

    # ---------------------------------------------------------------------- #
    # Test runner                                                              #
    # ---------------------------------------------------------------------- #

    def _run_integration(
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

    # ---------------------------------------------------------------------- #
    # Response parsing                                                         #
    # ---------------------------------------------------------------------- #

    def _extract_section(self, response: str, section: str) -> str:
        """Extract text under a ### SECTION heading (up to the next ### or end)."""
        pattern = rf"###\s+{re.escape(section)}\s*\n(.*?)(?=###|\Z)"
        m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _apply_changes(
        self,
        response: str,
        integration_dir: Path,
        jax_src_dir: Path,
    ) -> List[str]:
        """
        Parse all ``#### <filename>`` subsections from CHANGES, write each file.

        Returns list of filenames that were written.
        """
        changed: List[str] = []

        # Find the CHANGES section
        changes_section = self._extract_section(response, "CHANGES")
        if not changes_section:
            return changed

        # Each subsection starts with #### <filename>
        parts = re.split(r"####\s+", changes_section)
        for part in parts:
            if not part.strip():
                continue
            lines = part.split("\n", 1)
            filename = lines[0].strip()
            body = lines[1] if len(lines) > 1 else ""
            code = self._extract_code_block(body)
            if not code:
                continue

            # Resolve the file path: integration_dir first, then jax_src_dir
            target = self._resolve_target(filename, integration_dir, jax_src_dir)
            if target is None:
                logger.warning("Cannot resolve file path for '%s' — skipping", filename)
                continue

            target.write_text(code)
            changed.append(filename)
            console.print(f"  [dim]  Wrote: {target}[/dim]")

        return changed

    def _extract_code_block(self, text: str) -> str:
        """Extract the first ```python ... ``` or ``` ... ``` block."""
        for tag in ("```python", "```"):
            if tag in text:
                start = text.find(tag) + len(tag)
                end   = text.find("```", start)
                if end != -1:
                    return text[start:end].strip()
        return ""

    def _resolve_target(
        self,
        filename: str,
        integration_dir: Path,
        jax_src_dir: Path,
    ) -> Optional[Path]:
        """
        Map a bare filename (e.g. 'model_run.py' or 'CanopyFluxesMod.py')
        to an absolute path.  Priority:
          1. File lives in integration_dir
          2. File lives in jax_src_dir
        Rejects paths that try to escape these directories.
        """
        # Strip leading path separators / directory prefixes that Claude might emit
        name = Path(filename).name   # e.g. 'CanopyFluxesMod.py'
        candidates = [
            integration_dir / name,
            jax_src_dir / name,
        ]
        for c in candidates:
            if c.exists() or c.suffix == ".py":
                # Safety check: must stay within allowed roots
                try:
                    c.resolve().relative_to(integration_dir.resolve())
                    return c
                except ValueError:
                    pass
                try:
                    c.resolve().relative_to(jax_src_dir.resolve())
                    return c
                except ValueError:
                    pass
        return None

    # ---------------------------------------------------------------------- #
    # Module interface summary                                                 #
    # ---------------------------------------------------------------------- #

    def _collect_module_interfaces(self, jax_src_dir: Path) -> str:
        """
        Collect public function signatures from all translated modules.

        Uses AST parsing so we get accurate signatures without importing the
        modules (which might fail if dependencies are absent).
        """
        import ast

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

            funcs = []
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
    # Markdown report                                                           #
    # ---------------------------------------------------------------------- #

    def _write_report(self, result: IntegrationRepairResult) -> Path:
        docs_dir = result.integration_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        fname = f"integration_repair_{result.final_status}.md"
        report_path = docs_dir / fname
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        badge = "✅ PASS" if result.final_status == "PASS" else "❌ FAIL"

        lines: List[str] = [
            "# Integration Repair Report",
            "",
            f"**Status:** {badge}  ",
            f"**Date/time:** {now}  ",
            f"**Repair iterations:** {result.total_iterations}  ",
            "",
        ]

        for it in result.iterations:
            it_badge = "✅ PASS" if it.passed else "❌ FAIL"
            lines += [
                f"## Iteration {it.iteration}  {it_badge}",
                "",
                f"**Root cause:** {it.root_cause or '(not provided)'}",
                "",
                f"**Fix strategy:** {it.fix_strategy or '(not provided)'}",
                "",
                f"**Files changed:** {', '.join(it.files_changed) or '(none)'}",
                "",
            ]
            if it.error_snippet and result.final_status == "FAIL":
                lines += [
                    "**Error log (truncated):**",
                    "```",
                    it.error_snippet[:1500],
                    "```",
                    "",
                ]

        report_path.write_text("\n".join(lines))
        console.print(f"[dim]Integration repair report → {report_path}[/dim]")
        return report_path
