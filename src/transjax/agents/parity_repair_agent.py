"""
ParityRepairAgent — iterative numerical parity gatekeeper and repair orchestrator.

Workflow
--------
1. Run parity tests via ParityAgent.
2. PASS → write a green-light markdown report and return.
3. FAIL → enter a repair loop (up to *max_iterations*):
     a. Collect failing test logs and golden data summary.
     b. Ask Claude to diagnose the discrepancy and return a corrected module.
     c. Write the corrected code back to *python_file* (ONLY this file).
     d. Re-run parity tests.
     e. Repeat until all tests pass OR the iteration limit is reached.
4. Always write a repair-summary markdown in docs/.

Hard constraints (enforced by the repair prompt):
  • Claude may only edit the Python module — never golden data, test files,
    Fortran source, or any other file.
  • Function signatures, NamedTuple fields, and JAX JIT decorators must be
    preserved across iterations.
"""

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from transjax.agents.base_agent import BaseAgent
from transjax.agents.parity_agent import ParityAgent, ParityRunResult
from transjax.agents.prompts.parity_repair_prompts import (
    PARITY_REPAIR_PROMPT,
    PARITY_REPAIR_SYSTEM_PROMPT,
)
from transjax.agents.utils.config_loader import get_llm_config

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IterationLog:
    """Record of one repair iteration."""
    iteration: int
    root_cause: str = ""
    fix_applied: str = ""
    subroutines_passed: int = 0
    subroutines_failed: int = 0
    cases_passed: int = 0
    cases_total: int = 0
    parity_returncode: int = -1   # 0 = all tests passed this iteration

    @property
    def all_passed(self) -> bool:
        return self.parity_returncode == 0


@dataclass
class ParityRepairResult:
    """Aggregated outcome of the full parity-repair run."""
    python_file: Path
    fortran_file: Path
    golden_dir: Path
    output_dir: Path
    module_name: str
    final_status: str = "unknown"    # "PASS" | "FAIL"
    initial_run: Optional[ParityRunResult] = None
    iterations: List[IterationLog] = field(default_factory=list)
    final_run: Optional[ParityRunResult] = None
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

class ParityRepairAgent(BaseAgent):
    """
    Runs parity tests and, on failure, iteratively asks Claude to repair the
    JAX module until all parity tests pass or the iteration limit is reached.

    Only the Python module is ever modified.  Golden data, Fortran source,
    and generated test files are read-only.
    """

    DEFAULT_RTOL = 1e-10
    DEFAULT_ATOL = 1e-12
    DEFAULT_MAX_ITERATIONS = 5

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        llm_config = get_llm_config()
        super().__init__(
            name="ParityRepairAgent",
            role=(
                "Expert Fortran-to-JAX translator and numerical analyst who "
                "diagnoses parity failures between a JAX translation and its "
                "Fortran golden reference data, then applies targeted fixes."
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
        python_file: Path,
        fortran_file: Path,
        golden_dir: Path,
        output_dir: Path,
        ftest_report_path: Optional[Path] = None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        verbose: bool = False,
    ) -> ParityRepairResult:
        """
        Run parity tests; repair the JAX module if tests fail.

        Args:
            python_file:       Translated JAX/Python module to test and repair.
            fortran_file:      Original Fortran source (read-only reference for Claude).
            golden_dir:        Directory with golden JSON files.
            output_dir:        Where test files and the parity report are written.
            ftest_report_path: Path to ftest_report.json (from ``transjax ftest``).
            rtol:              Relative tolerance for jnp.allclose.
            atol:              Absolute tolerance for jnp.allclose.
            max_iterations:    Maximum repair iterations (default 5).
            verbose:           Print DEBUG logs and full pytest output.

        Returns:
            ParityRepairResult with per-iteration logs and final status.
        """
        python_file  = Path(python_file).resolve()
        fortran_file = Path(fortran_file).resolve()
        golden_dir   = Path(golden_dir).resolve()
        output_dir   = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        module_name = python_file.stem

        result = ParityRepairResult(
            python_file=python_file,
            fortran_file=fortran_file,
            golden_dir=golden_dir,
            output_dir=output_dir,
            module_name=module_name,
        )

        console.print(
            Panel.fit(
                f"[white]Python module:[/white]  {python_file}\n"
                f"[white]Fortran source:[/white] {fortran_file}\n"
                f"[white]Golden dir:   [/white]  {golden_dir}\n"
                f"[white]Output dir:   [/white]  {output_dir}\n"
                f"[white]Tolerance:    [/white]  rtol={rtol}, atol={atol}\n"
                f"[white]Max iterations:[/white] {max_iterations}",
                title="[bold cyan]TransJAX Parity Repair[/bold cyan]",
            )
        )

        # ── Save a backup of the original Python module ───────────────────────
        backup_path = output_dir / f"{module_name}_original_backup.py"
        if not backup_path.exists():
            shutil.copy2(python_file, backup_path)
            console.print(f"[dim]Backup saved: {backup_path}[/dim]")

        # ── Step 1: Initial parity run ────────────────────────────────────────
        console.print("\n[bold]Step 1[/bold] — Initial parity test run")
        parity_agent = self._make_parity_agent()
        initial_run = parity_agent.run(
            python_file=python_file,
            golden_dir=golden_dir,
            output_dir=output_dir,
            ftest_report_path=ftest_report_path,
            rtol=rtol,
            atol=atol,
            verbose=verbose,
        )
        result.initial_run = initial_run

        # ── PASS: done ────────────────────────────────────────────────────────
        if initial_run.subroutines_failed == 0:
            console.print("\n[bold green]✓ All parity tests passed — no repair needed.[/bold green]")
            result.final_status = "PASS"
            result.final_run = initial_run
            result.report_path = self._write_report(result, rtol, atol)
            return result

        console.print(
            f"\n[yellow]{initial_run.subroutines_failed} subroutine(s) failed parity. "
            f"Starting repair loop (max {max_iterations} iteration(s)).[/yellow]"
        )

        # ── Step 2: Repair loop ───────────────────────────────────────────────
        fortran_source = fortran_file.read_text() if fortran_file.exists() else "(Fortran source not found)"
        current_run = initial_run

        for i in range(1, max_iterations + 1):
            console.print(f"\n[bold cyan]━━ Repair Iteration {i}/{max_iterations} ━━[/bold cyan]")
            iter_log = IterationLog(iteration=i)

            failing_summary = self._build_failing_summary(current_run, verbose)
            golden_summary  = self._build_golden_summary(golden_dir, module_name)
            python_source   = python_file.read_text()

            prompt = PARITY_REPAIR_PROMPT.format(
                module_name=module_name,
                iteration=i,
                python_source=python_source,
                fortran_source=fortran_source,
                failing_tests_summary=failing_summary,
                golden_summary=golden_summary,
                rtol=rtol,
                atol=atol,
            )

            console.print(f"  [cyan]Asking Claude to diagnose and fix {module_name}…[/cyan]")
            response = self.query_claude(
                prompt=prompt,
                system_prompt=PARITY_REPAIR_SYSTEM_PROMPT,
            )

            iter_log.root_cause = self._extract_section(response, "ROOT CAUSE")
            iter_log.fix_applied = self._extract_section(response, "FIX APPLIED")
            new_code = self._extract_code(response)

            console.print(f"  [dim]Root cause:[/dim] {iter_log.root_cause[:120]}")
            console.print(f"  [dim]Fix applied:[/dim] {iter_log.fix_applied[:120]}")

            if not new_code:
                logger.warning("Iteration %d: Claude returned no code block — skipping write.", i)
                iter_log.fix_applied = "(no code returned by Claude)"
                result.iterations.append(iter_log)
                continue

            # Write the corrected module (ONLY this file may be changed)
            python_file.write_text(new_code)
            console.print(f"  [green]✓ Wrote corrected module → {python_file}[/green]")

            # Re-run parity tests with fresh test generation
            console.print(f"  [bold]Re-running parity tests…[/bold]")
            current_run = parity_agent.run(
                python_file=python_file,
                golden_dir=golden_dir,
                output_dir=output_dir,
                ftest_report_path=ftest_report_path,
                rtol=rtol,
                atol=atol,
                verbose=verbose,
            )

            iter_log.subroutines_passed = current_run.subroutines_passed
            iter_log.subroutines_failed = current_run.subroutines_failed
            iter_log.cases_passed       = current_run.cases_passed
            iter_log.cases_total        = current_run.total_cases
            iter_log.parity_returncode  = 0 if current_run.subroutines_failed == 0 else 1

            result.iterations.append(iter_log)

            if current_run.subroutines_failed == 0:
                console.print(
                    f"\n[bold green]✓ All parity tests passed after {i} repair iteration(s).[/bold green]"
                )
                break
            else:
                console.print(
                    f"  [yellow]{current_run.subroutines_failed} subroutine(s) still failing.[/yellow]"
                )

        result.final_run = current_run
        result.final_status = "PASS" if current_run.subroutines_failed == 0 else "FAIL"

        if result.final_status == "FAIL":
            console.print(
                f"\n[bold red]✗ Parity repair did not converge after {max_iterations} iteration(s).[/bold red]"
            )

        result.report_path = self._write_report(result, rtol, atol)
        return result

    # ---------------------------------------------------------------------- #
    # Prompt helpers                                                           #
    # ---------------------------------------------------------------------- #

    def _build_failing_summary(self, run: ParityRunResult, verbose: bool) -> str:
        """Collect failure output from all failed subroutine parity tests."""
        lines: List[str] = []
        for sub in run.subroutine_results:
            if sub.status != "failed":
                continue
            lines.append(f"### {sub.subroutine_name} ({sub.cases_failed}/{sub.cases_total} cases failed)")
            stdout = sub.pytest_stdout or "(no pytest output captured)"
            # Include full output when verbose, otherwise cap at 3000 chars to
            # avoid blowing the prompt budget on repeated boilerplate.
            if not verbose and len(stdout) > 3000:
                stdout = stdout[:3000] + "\n… (truncated)"
            lines.append("```\n" + stdout + "\n```")
        return "\n\n".join(lines) if lines else "(no failing subroutines)"

    def _build_golden_summary(self, golden_dir: Path, module_name: str) -> str:
        """Summarise golden cases for all JSON files matching *module_name*."""
        lines: List[str] = []
        json_files = sorted(golden_dir.glob("*.json")) if golden_dir.exists() else []
        lower = module_name.lower()
        matching = [f for f in json_files if lower in f.name.lower()]
        if not matching:
            matching = json_files  # fallback: all JSON in dir

        for jf in matching[:10]:   # cap at 10 files to keep prompt reasonable
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue
            sub  = data.get("subroutine", jf.stem)
            cases = data.get("cases", [])
            lines.append(f"**{sub}** — {len(cases)} case(s) in `{jf.name}`")
            # Show first case I/O as a compact sample
            if cases:
                sample = cases[0]
                inp_keys  = list(sample.get("inputs", {}).keys())
                out_keys  = list(sample.get("outputs", {}).keys())
                lines.append(f"  inputs:  {inp_keys}")
                lines.append(f"  outputs: {out_keys}")
                # Show a snippet of actual values for the first output
                first_out_key = out_keys[0] if out_keys else None
                if first_out_key:
                    val = sample["outputs"][first_out_key]
                    val_repr = repr(val)[:200]
                    lines.append(f"  {first_out_key} (case 0): {val_repr}")

        return "\n".join(lines) if lines else "(no golden data found)"

    # ---------------------------------------------------------------------- #
    # Response parsing                                                         #
    # ---------------------------------------------------------------------- #

    def _extract_section(self, response: str, section: str) -> str:
        """Extract text under a ### SECTION heading (up to the next ### or end)."""
        pattern = rf"###\s+{re.escape(section)}\s*\n(.*?)(?=###|\Z)"
        m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return ""

    def _extract_code(self, response: str) -> str:
        """Extract the first ```python ... ``` block from the response."""
        if "```python" in response:
            start = response.find("```python") + 9
            end   = response.find("```", start)
            return response[start:end if end != -1 else None].strip()
        if "```" in response:
            start = response.find("```") + 3
            end   = response.find("```", start)
            return response[start:end if end != -1 else None].strip()
        return ""

    # ---------------------------------------------------------------------- #
    # Markdown report                                                           #
    # ---------------------------------------------------------------------- #

    def _write_report(self, result: ParityRepairResult, rtol: float, atol: float) -> Path:
        """Write the repair summary markdown to docs/<module>_parity_repair_<STATUS>.md."""
        docs_dir = result.output_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{result.module_name}_numerical_parity_repair_{result.final_status}.md"
        report_path = docs_dir / fname

        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_badge = "✅ PASS" if result.final_status == "PASS" else "❌ FAIL"

        lines: List[str] = [
            f"# Numerical Parity Repair — `{result.module_name}`",
            "",
            f"**Status:** {status_badge}  ",
            f"**Date/time:** {now}  ",
            f"**Tolerance:** rtol={rtol}, atol={atol}  ",
            f"**Repair iterations:** {result.total_iterations}  ",
            f"**Python module:** `{result.python_file}`  ",
            f"**Fortran source:** `{result.fortran_file}`  ",
            "",
        ]

        # ── Initial run summary ───────────────────────────────────────────────
        if result.initial_run:
            r = result.initial_run
            lines += [
                "## Initial Parity Run",
                "",
                f"| Subroutines | Passed | Failed | Cases | Cases Passed |",
                f"|-------------|--------|--------|-------|--------------|",
                f"| {r.total_subroutines} | {r.subroutines_passed} | "
                f"{r.subroutines_failed} | {r.total_cases} | {r.cases_passed} |",
                "",
            ]
            for sub in r.subroutine_results:
                icon = "✅" if sub.status == "passed" else "❌"
                lines.append(f"- {icon} `{sub.subroutine_name}` — {sub.status}")
            lines.append("")

        # ── Iteration log ─────────────────────────────────────────────────────
        if result.iterations:
            lines += ["## Repair Iterations", ""]
            for it in result.iterations:
                it_status = "✅ PASS" if it.all_passed else "❌ FAIL"
                lines += [
                    f"### Iteration {it.iteration}  {it_status}",
                    "",
                    f"**Root cause:** {it.root_cause or '(not provided)'}",
                    "",
                    f"**Fix applied:** {it.fix_applied or '(not provided)'}",
                    "",
                    f"| Subroutines Passed | Subroutines Failed | Cases Passed | Cases Total |",
                    f"|--------------------|--------------------|--------------|-------------|",
                    f"| {it.subroutines_passed} | {it.subroutines_failed} "
                    f"| {it.cases_passed} | {it.cases_total} |",
                    "",
                ]

        # ── Final run summary ─────────────────────────────────────────────────
        if result.final_run and result.final_run is not result.initial_run:
            r = result.final_run
            lines += [
                "## Final Parity Run",
                "",
                f"| Subroutines | Passed | Failed | Cases | Cases Passed |",
                f"|-------------|--------|--------|-------|--------------|",
                f"| {r.total_subroutines} | {r.subroutines_passed} | "
                f"{r.subroutines_failed} | {r.total_cases} | {r.cases_passed} |",
                "",
            ]
            for sub in r.subroutine_results:
                icon = "✅" if sub.status == "passed" else "❌"
                lines.append(f"- {icon} `{sub.subroutine_name}` — {sub.status}")
            lines.append("")

        # ── Residual discrepancies on FAIL ────────────────────────────────────
        if result.final_status == "FAIL" and result.final_run:
            lines += ["## Residual Discrepancies", ""]
            for sub in result.final_run.subroutine_results:
                if sub.status != "failed":
                    continue
                lines += [
                    f"### `{sub.subroutine_name}`",
                    "",
                    "```",
                    (sub.pytest_stdout or "(no output)")[:2000],
                    "```",
                    "",
                ]

        report_path.write_text("\n".join(lines))
        console.print(f"\n[dim]Repair report → {report_path}[/dim]")
        return report_path

    # ---------------------------------------------------------------------- #
    # Factory helper                                                           #
    # ---------------------------------------------------------------------- #

    def _make_parity_agent(self) -> ParityAgent:
        """Create a ParityAgent with the same model as this agent."""
        return ParityAgent(
            model=self.model,
            temperature=self.temperature,
        )
