"""
Repair Agent for fixing failed JAX translations.

This agent focuses on debugging and fixing failed Python/JAX translations by
analyzing test reports and iteratively refining the output.
"""

import json
import logging
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from jax_agents.base_agent import BaseAgent
from jax_agents.utils.config_loader import get_llm_config
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

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
        
        # Mapping of filenames to content
        artifacts = {
            f"{self.module_name}_corrected.py": self.corrected_python_code,
            f"root_cause_analysis_{self.module_name}.md": self.root_cause_analysis,
            f"final_test_report_{self.module_name}.txt": self.final_test_report
        }

        for filename, content in artifacts.items():
            path = output_dir / filename
            path.write_text(content)
            saved_files[filename] = path
            console.print(f"[green]✓ Saved {filename}[/green]")
        
        # Save structured failure analysis
        failure_path = output_dir / f"failure_analysis_{self.module_name}.json"
        failure_path.write_text(json.dumps(self.failure_analysis, indent=2))
        saved_files["failure_analysis"] = failure_path
        
        return saved_files

class RepairAgent(BaseAgent):
    """
    Agent for repairing failed JAX translations using iterative testing.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_repair_iterations: int = 5,
    ):
        llm_config = get_llm_config()
        
        super().__init__(
            name="RepairAgent",
            role="JAX translation debugger and fixer",
            model=model or llm_config.get("model", "claude-sonnet-4-5"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 32000),
        )
        self.max_repair_iterations = max_repair_iterations

    def _extract_block(self, text: str, block_type: str = "json") -> str:
        """Helper to extract content from markdown code blocks reliably."""
        pattern = rf"```(?:{block_type})?\n?(.*?)\n?```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def repair_translation(
        self,
        module_name: str,
        fortran_code: str,
        failed_python_code: str,
        test_report: str,
        test_file_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> RepairResult:
        """Main repair loop: Analyze -> Fix -> Verify."""
        console.print(f"\n[bold cyan]🔧 Repairing {module_name}[/bold cyan]")
        
        current_python_code = failed_python_code
        iteration = 0
        all_tests_passed = False
        test_results_history = [{"iteration": 0, "report": test_report}]

        # Initial Analysis
        console.print("[cyan]Step 1: Analyzing test failures...[/cyan]")
        failure_analysis = self._analyze_failure(fortran_code, current_python_code, test_report)
        
        # Repair Loop
        while iteration < self.max_repair_iterations and not all_tests_passed:
            iteration += 1
            console.print(f"\n[bold cyan]Iteration {iteration}/{self.max_repair_iterations}[/bold cyan]")
            
            corrected_code = self._generate_fix(
                fortran_code, current_python_code, failure_analysis, test_report
            )
            
            # If we have a test suite, run it
            if test_file_path:
                console.print("[cyan]Running validation tests...[/cyan]")
                test_success, new_report = self._run_tests(corrected_code, test_file_path)
                
                test_results_history.append({"iteration": iteration, "report": new_report})
                test_report = new_report
                current_python_code = corrected_code

                if test_success:
                    console.print(f"[green]✓ Tests passed on iteration {iteration}![/green]")
                    all_tests_passed = True
                else:
                    console.print("[yellow]Tests still failing, re-analyzing...[/yellow]")
                    failure_analysis = self._analyze_failure(fortran_code, corrected_code, new_report)
            else:
                console.print("[yellow]⚠ No test file; assuming fix is correct.[/yellow]")
                current_python_code = corrected_code
                all_tests_passed = True

        # Generate Final Report
        rca_report = self._generate_root_cause_report(
            fortran_code, failed_python_code, current_python_code, 
            failure_analysis, test_report, test_results_history
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

        if output_dir:
            result.save(output_dir)
        return result

    def _analyze_failure(self, fortran: str, python: str, report: str) -> Dict[str, Any]:
        from jax_agents.prompts.repair_prompts import REPAIR_PROMPTS
        
        response = self.query_claude(
            prompt=REPAIR_PROMPTS["analyze_failure"].format(
                fortran_code=fortran, python_code=python, test_report=report
            ),
            system_prompt=REPAIR_PROMPTS["system"],
        )
        
        try:
            return json.loads(self._extract_block(response, "json"))
        except json.JSONDecodeError:
            logger.error("Failed to parse failure analysis JSON")
            return {"error": "Parse error", "root_causes": ["Manual review required"]}

    def _generate_fix(self, fortran: str, python: str, analysis: Dict[str, Any], report: str) -> str:
        from jax_agents.prompts.repair_prompts import REPAIR_PROMPTS
        
        response = self.query_claude(
            prompt=REPAIR_PROMPTS["generate_fix"].format(
                fortran_code=fortran, 
                python_code=python, 
                root_cause_analysis=json.dumps(analysis, indent=2),
                test_report=report
            ),
            system_prompt=REPAIR_PROMPTS["system"],
        )
        return self._extract_block(response, "python")

    def _run_tests(self, code: str, test_path: Path) -> Tuple[bool, str]:
        """Executes pytest in a controlled environment with the fix."""
        # Note: We write the fix to a temporary file in the same dir to handle imports
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=test_path.parent, delete=False) as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)
        
        try:
            res = subprocess.run(
                ['pytest', str(test_path), '-v', '--tb=short'],
                capture_output=True, text=True, timeout=300
            )
            return (res.returncode == 0), f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        except Exception as e:
            return False, f"Test execution failed: {str(e)}"
        finally:
            tmp_path.unlink(missing_ok=True)

    def _generate_root_cause_report(self, fortran: str, old_py: str, new_py: str, 
                                   analysis: Dict[str, Any], report: str, history: List[Any]) -> str:
        from jax_agents.prompts.repair_prompts import REPAIR_PROMPTS
        
        hist_str = "\n".join([f"Iter {h['iteration']}: {'Passed' if 'PASSED' in h['report'] else 'Failed'}" for h in history])
        
        return self.query_claude(
            prompt=REPAIR_PROMPTS["root_cause_report"].format(
                fortran_code=fortran,
                failed_python_code=old_py,
                corrected_python_code=new_py,
                failure_analysis=json.dumps(analysis, indent=2),
                test_results=f"{hist_str}\n\nFinal Report:\n{report}"
            ),
            system_prompt=REPAIR_PROMPTS["system"]
        )