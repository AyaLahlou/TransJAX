"""
Repair Agent for fixing failed JAX translations.

This agent focuses on debugging and fixing failed Python/JAX translations:
1. Analyzes test failures and error messages
2. Compares with original Fortran code
3. Identifies root causes
4. Generates corrected Python code
5. Iteratively fixes until tests pass
6. Provides comprehensive root cause analysis
"""

import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from transjax.agents.base_agent import BaseAgent
from transjax.agents.utils.config_loader import get_llm_config

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class RalphIteration:
    """Structured log entry for one RALPH iteration."""
    iteration: int
    stage: str                          # run | assess | patch | halt
    elapsed_s: float = 0.0
    passed: int = 0
    failed: int = 0
    root_causes: List[str] = field(default_factory=list)
    lines_changed: int = 0
    halt_reason: str = ""               # filled only on halt stage


@dataclass
class RepairResult:
    """Complete result of repair process."""
    module_name: str
    original_python_code: str
    corrected_python_code: str
    root_cause_analysis: str
    failure_analysis: Dict[str, Any]
    iterations: int
    final_test_report: str
    all_tests_passed: bool

    def save(self, output_dir: Path) -> Dict[str, Path]:
        """Save all repair artifacts to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save corrected Python code
        corrected_path = output_dir / f"{self.module_name}_corrected.py"
        with open(corrected_path, 'w') as f:
            f.write(self.corrected_python_code)
        saved_files["corrected_code"] = corrected_path
        console.print(f"[green]✓ Saved corrected code to {corrected_path}[/green]")

        # Save root cause analysis
        rca_path = output_dir / f"root_cause_analysis_{self.module_name}.md"
        with open(rca_path, 'w') as f:
            f.write(self.root_cause_analysis)
        saved_files["root_cause_analysis"] = rca_path
        console.print(f"[green]✓ Saved root cause analysis to {rca_path}[/green]")

        # Save failure analysis
        failure_path = output_dir / f"failure_analysis_{self.module_name}.json"
        with open(failure_path, 'w') as f:
            json.dump(self.failure_analysis, f, indent=2)
        saved_files["failure_analysis"] = failure_path
        console.print(f"[green]✓ Saved failure analysis to {failure_path}[/green]")

        # Save final test report
        test_report_path = output_dir / f"final_test_report_{self.module_name}.txt"
        with open(test_report_path, 'w') as f:
            f.write(self.final_test_report)
        saved_files["test_report"] = test_report_path
        console.print(f"[green]✓ Saved final test report to {test_report_path}[/green]")

        return saved_files


class RepairAgent(BaseAgent):
    """
    Agent for repairing failed JAX translations.

    Responsibilities:
    - Analyze test failures and identify root causes
    - Generate corrected Python code
    - Run tests and verify fixes
    - Iterate until tests pass (or max iterations reached)
    - Generate comprehensive root cause analysis reports
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_repair_iterations: int = 5,
    ):
        """
        Initialize Repair Agent.

        Args:
            model: Claude model to use (defaults to config.yaml)
            temperature: Sampling temperature (defaults to config.yaml)
            max_tokens: Maximum tokens in response (defaults to config.yaml)
            max_repair_iterations: Maximum number of repair iterations
        """
        llm_config = get_llm_config()

        super().__init__(
            name="RepairAgent",
            role="JAX translation debugger and fixer",
            model=model or llm_config.get("model", "claude-sonnet-4-6"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 32000),
        )

        self.max_repair_iterations = max_repair_iterations

    def repair_translation(
        self,
        module_name: str,
        fortran_code: str,
        failed_python_code: str,
        test_report: str,
        test_file_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> RepairResult:
        """
        Repair a failed Python/JAX translation using the RALPH loop.

        RALPH: Run → Assess → Loop → Patch → Halt

        Args:
            module_name: Name of the module being repaired
            fortran_code: Original Fortran subroutine/function
            failed_python_code: Failed Python translation
            test_report: Test report showing failures
            test_file_path: Optional path to pytest file for running tests
            output_dir: Optional directory to save outputs

        Returns:
            RepairResult with corrected code and analysis
        """
        console.print(f"\n[bold cyan]🔧 RALPH loop — repairing {module_name}[/bold cyan]")

        current_python_code = failed_python_code
        iteration = 0
        all_tests_passed = False
        ralph_log: List[RalphIteration] = []
        test_results_history: List[Dict[str, Any]] = [{"iteration": 0, "report": test_report}]

        # Initial assess before first patch
        failure_analysis = self._ralph_assess(
            fortran_code, current_python_code, test_report, module_name,
            ralph_log=ralph_log, iteration=0,
        )

        while not self._ralph_halt(
            iteration, self.max_repair_iterations, all_tests_passed, ralph_log
        ):
            iteration += 1
            console.print(
                f"\n[bold cyan]RALPH iteration {iteration}/{self.max_repair_iterations}[/bold cyan]"
            )

            # ── Patch ──────────────────────────────────────────────────
            corrected_code = self._ralph_patch(
                fortran_code, current_python_code, failure_analysis, test_report,
                module_name, ralph_log=ralph_log, iteration=iteration,
            )

            if test_file_path:
                # ── Run ────────────────────────────────────────────────
                test_success, new_test_report = self._ralph_run(
                    corrected_code, test_file_path, module_name,
                    ralph_log=ralph_log, iteration=iteration,
                )
                test_results_history.append({"iteration": iteration, "report": new_test_report})

                if test_success:
                    all_tests_passed = True
                    current_python_code = corrected_code
                    test_report = new_test_report
                    self._ralph_halt(
                        iteration, self.max_repair_iterations, True, ralph_log
                    )
                    break

                # ── Assess ─────────────────────────────────────────────
                failure_analysis = self._ralph_assess(
                    fortran_code, corrected_code, new_test_report, module_name,
                    ralph_log=ralph_log, iteration=iteration,
                )
                current_python_code = corrected_code
                test_report = new_test_report
            else:
                console.print("[yellow]⚠ No test file — accepting patch without verification[/yellow]")
                current_python_code = corrected_code
                all_tests_passed = True
                break

        if not all_tests_passed:
            console.print(f"[red]⚠ RALPH: max iterations ({self.max_repair_iterations}) reached[/red]")
            ralph_log.append(RalphIteration(
                iteration=iteration,
                stage="halt",
                halt_reason="max_iterations_reached",
            ))

        # Generate RCA report
        console.print("[cyan]Generating root cause analysis report...[/cyan]")
        rca_report = self._generate_root_cause_report(
            fortran_code,
            failed_python_code,
            current_python_code,
            failure_analysis,
            test_report,
            module_name,
            test_results_history,
        )

        result = RepairResult(
            module_name=module_name,
            original_python_code=failed_python_code,
            corrected_python_code=current_python_code,
            root_cause_analysis=rca_report,
            failure_analysis=failure_analysis,
            iterations=iteration,
            final_test_report=test_report,
            all_tests_passed=all_tests_passed,
        )

        console.print("[green]✓ RALPH repair complete![/green]")

        if output_dir:
            result.save(output_dir)
            self._save_ralph_log(module_name, ralph_log, output_dir)

        return result

    # ------------------------------------------------------------------
    # RALPH stage methods
    # ------------------------------------------------------------------

    def _ralph_run(
        self,
        python_code: str,
        test_file_path: Path,
        module_name: str,
        ralph_log: List[RalphIteration],
        iteration: int,
    ) -> Tuple[bool, str]:
        """RALPH stage R — Run tests and record outcome."""
        console.print("[cyan][R] Run tests...[/cyan]")
        t0 = time.monotonic()
        success, test_report = self._run_tests(python_code, test_file_path, module_name)
        elapsed = time.monotonic() - t0

        passed_count = test_report.count(" passed")
        failed_count = test_report.count(" failed")

        ralph_log.append(RalphIteration(
            iteration=iteration,
            stage="run",
            elapsed_s=round(elapsed, 2),
            passed=passed_count,
            failed=failed_count,
        ))
        icon = "[green]✓[/green]" if success else "[red]✗[/red]"
        console.print(
            f"  {icon} tests in {elapsed:.1f}s — "
            f"passed={passed_count} failed={failed_count}"
        )
        return success, test_report

    def _ralph_assess(
        self,
        fortran_code: str,
        python_code: str,
        test_report: str,
        module_name: str,
        ralph_log: List[RalphIteration],
        iteration: int,
    ) -> Dict[str, Any]:
        """RALPH stage A — Assess failures with claude."""
        console.print("[cyan][A] Assess failures...[/cyan]")
        failure_analysis = self._analyze_failure(
            fortran_code, python_code, test_report, module_name
        )
        root_causes = failure_analysis.get("root_causes", [])
        console.print(f"  Found {len(root_causes)} root cause(s)")
        ralph_log.append(RalphIteration(
            iteration=iteration,
            stage="assess",
            root_causes=[str(rc) for rc in root_causes],
        ))
        return failure_analysis

    def _ralph_patch(
        self,
        fortran_code: str,
        python_code: str,
        failure_analysis: Dict[str, Any],
        test_report: str,
        module_name: str,
        ralph_log: List[RalphIteration],
        iteration: int,
    ) -> str:
        """RALPH stage P — Patch the code with claude."""
        console.print("[cyan][P] Patch code...[/cyan]")
        corrected_code = self._generate_fix(
            fortran_code, python_code, failure_analysis, test_report, module_name
        )
        lines_changed = abs(len(corrected_code.splitlines()) - len(python_code.splitlines()))
        ralph_log.append(RalphIteration(
            iteration=iteration,
            stage="patch",
            lines_changed=lines_changed,
        ))
        console.print(f"  Patch applied (~{lines_changed} lines changed)")
        return corrected_code

    def _ralph_halt(
        self,
        iteration: int,
        max_iterations: int,
        all_tests_passed: bool,
        ralph_log: List[RalphIteration],
    ) -> bool:
        """RALPH stage H — check halt conditions."""
        if all_tests_passed:
            ralph_log.append(RalphIteration(
                iteration=iteration,
                stage="halt",
                halt_reason="all_passed",
            ))
            console.print(f"[green][H] Halt — all tests passed (iteration {iteration})[/green]")
            return True
        if iteration >= max_iterations:
            # The main loop will append halt log entry
            return True
        return False

    def _save_ralph_log(
        self,
        module_name: str,
        ralph_log: List[RalphIteration],
        output_dir: Path,
    ) -> None:
        """Write ralph_log.json alongside other repair artifacts."""
        log_path = output_dir / f"{module_name}_ralph_log.json"
        data = {
            "module": module_name,
            "iterations": [
                {
                    "iteration": e.iteration,
                    "stage": e.stage,
                    "elapsed_s": e.elapsed_s,
                    "passed": e.passed,
                    "failed": e.failed,
                    "root_causes": e.root_causes,
                    "lines_changed": e.lines_changed,
                    "halt_reason": e.halt_reason,
                }
                for e in ralph_log
            ],
        }
        with open(log_path, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]✓ Saved RALPH log → {log_path}[/green]")

    def _analyze_failure(
        self,
        fortran_code: str,
        python_code: str,
        test_report: str,
        module_name: str,
    ) -> Dict[str, Any]:
        """Analyze test failure and identify root causes."""
        from transjax.agents.prompts.repair_prompts import REPAIR_PROMPTS

        prompt = REPAIR_PROMPTS["analyze_failure"].format(
            fortran_code=fortran_code,
            python_code=python_code,
            test_report=test_report,
        )

        response = self.query_claude(
            prompt=prompt,
            system_prompt=REPAIR_PROMPTS["system"],
        )

        # Parse JSON response
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse failure analysis: {e}")
            logger.error(f"Response: {response}")
            # Return a basic structure if parsing fails
            return {
                "failed_tests": ["unknown"],
                "error_summary": "Failed to parse error analysis",
                "root_causes": [],
                "required_fixes": ["Review test report manually"]
            }

    def _generate_fix(
        self,
        fortran_code: str,
        python_code: str,
        failure_analysis: Dict[str, Any],
        test_report: str,
        module_name: str,
    ) -> str:
        """Generate corrected Python code."""
        from transjax.agents.prompts.repair_prompts import REPAIR_PROMPTS

        prompt = REPAIR_PROMPTS["generate_fix"].format(
            fortran_code=fortran_code,
            python_code=python_code,
            root_cause_analysis=json.dumps(failure_analysis, indent=2),
            test_report=test_report,
        )

        response = self.query_claude(
            prompt=prompt,
            system_prompt=REPAIR_PROMPTS["system"],
        )

        # Extract Python code
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return response[start:end].strip()
        else:
            return response.strip()

    def _verify_fix(
        self,
        failure_analysis: Dict[str, Any],
        corrected_code: str,
        required_fixes: List[str],
    ) -> Dict[str, Any]:
        """Verify if the corrected code addresses all issues."""
        from transjax.agents.prompts.repair_prompts import REPAIR_PROMPTS

        prompt = REPAIR_PROMPTS["verify_fix"].format(
            failure_analysis=json.dumps(failure_analysis, indent=2),
            corrected_code=corrected_code,
            required_fixes=json.dumps(required_fixes, indent=2),
        )

        response = self.query_claude(
            prompt=prompt,
            system_prompt=REPAIR_PROMPTS["system"],
        )

        # Parse JSON response
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse verification: {e}")
            return {
                "all_issues_addressed": False,
                "confidence_level": "low",
                "recommendations": ["Manual review required"]
            }

    def _run_tests(
        self,
        python_code: str,
        test_file_path: Path,
        module_name: str,
    ) -> Tuple[bool, str]:
        """
        Run tests with the corrected Python code.

        Args:
            python_code: Corrected Python code
            test_file_path: Path to pytest file
            module_name: Module name

        Returns:
            Tuple of (success: bool, test_report: str)
        """
        try:
            # Create temporary file with corrected code
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, dir=test_file_path.parent
            ) as tmp_file:
                tmp_file.write(python_code)
                tmp_file_path = Path(tmp_file.name)

            try:
                # Run pytest
                result = subprocess.run(
                    ['pytest', str(test_file_path), '-v', '--tb=short'],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )

                test_report = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                success = result.returncode == 0

                return success, test_report

            finally:
                # Clean up temporary file
                tmp_file_path.unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            return False, "Error: Tests timed out after 5 minutes"
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False, f"Error running tests: {str(e)}"

    def _generate_root_cause_report(
        self,
        fortran_code: str,
        failed_python_code: str,
        corrected_python_code: str,
        failure_analysis: Dict[str, Any],
        final_test_report: str,
        module_name: str,
        test_results_history: List[Dict[str, Any]],
    ) -> str:
        """Generate comprehensive root cause analysis report."""
        from transjax.agents.prompts.repair_prompts import REPAIR_PROMPTS

        # Build test results summary
        test_results_summary = "\n".join([
            f"Iteration {h['iteration']}: {'PASSED' if 'PASSED' in h['report'] or 'passed' in h['report'].lower() else 'FAILED'}"
            for h in test_results_history
        ])

        prompt = REPAIR_PROMPTS["root_cause_report"].format(
            fortran_code=fortran_code,
            failed_python_code=failed_python_code,
            corrected_python_code=corrected_python_code,
            failure_analysis=json.dumps(failure_analysis, indent=2),
            test_results=f"{test_results_summary}\n\nFinal Report:\n{final_test_report}",
        )

        response = self.query_claude(
            prompt=prompt,
            system_prompt=REPAIR_PROMPTS["system"],
        )

        return response

