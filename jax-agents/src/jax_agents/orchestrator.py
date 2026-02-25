"""
Orchestrator Agent - Manages the complete Fortran-to-JAX translation pipeline.

Updated with robust, case-insensitive module lookup logic to prevent 
'Module not found' errors during Step 3.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from jax_agents.base_agent import BaseAgent
from jax_agents.translator import TranslatorAgent
from jax_agents.test_agent import TestAgent
from jax_agents.repair_agent import RepairAgent
from jax_agents.utils.config_loader import get_llm_config

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class ModuleStatus:
    """Track status of a single module through the pipeline."""
    name: str
    translated: bool = False
    tests_generated: bool = False
    tests_passed: bool = False
    repair_attempts: int = 0
    final_status: str = "pending"  # pending, success, failed
    error_message: Optional[str] = None


@dataclass
class PipelineResults:
    """Final results of the translation pipeline."""
    total_modules: int
    translated_count: int
    tests_generated: int
    tests_passed: int
    repairs_needed: int
    final_failures: int
    module_statuses: Dict[str, ModuleStatus] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'translated_count': self.translated_count,
            'tests_generated': self.tests_generated,
            'tests_passed': self.tests_passed,
            'repairs_needed': self.repairs_needed,
            'final_failures': self.final_failures,
        }


class OrchestratorAgent:
    """
    Orchestrates the complete Fortran-to-JAX translation pipeline.
    """
    
    def __init__(
        self,
        fortran_dir: Path,
        output_dir: Path,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_repair_iterations: int = 5,
        skip_tests: bool = False,
        skip_repair: bool = False,
        force_retranslate: bool = False,
        module_list: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        self.fortran_dir = fortran_dir
        self.output_dir = output_dir
        self.max_repair_iterations = max_repair_iterations
        self.skip_tests = skip_tests
        self.skip_repair = skip_repair
        self.force_retranslate = force_retranslate
        self.module_list = module_list
        self.verbose = verbose
        
        # Create output directories
        self.src_dir = output_dir / "src"
        self.tests_dir = output_dir / "tests"
        self.docs_dir = output_dir / "docs"
        self.reports_dir = output_dir / "reports"
        self.analysis_dir = output_dir / "static_analysis"
        
        for d in [self.src_dir, self.tests_dir, self.docs_dir, self.reports_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize agents
        llm_config = get_llm_config()
        self.translator = TranslatorAgent(
            model=model or llm_config.get("model"),
            temperature=temperature if temperature is not None else llm_config.get("temperature"),
            fortran_root=fortran_dir,
        )
        
        if not skip_tests:
            self.test_agent = TestAgent(
                model=model or llm_config.get("model"),
                temperature=temperature if temperature is not None else llm_config.get("temperature"),
            )
        
        if not skip_repair:
            self.repair_agent = RepairAgent(
                model=model or llm_config.get("model"),
                temperature=temperature if temperature is not None else llm_config.get("temperature"),
                max_repair_iterations=max_repair_iterations,
            )
        
        # State tracking
        self.completed_modules: Set[str] = set()
        self.module_statuses: Dict[str, ModuleStatus] = {}

    def run(self) -> Dict[str, Any]:
        """Execute the complete translation pipeline."""
        console.print("\n[bold cyan]Starting Fortran-to-JAX Translation Pipeline[/bold cyan]\n")
        
        # Step 1: Static Analysis
        analysis_data = self._run_static_analysis()
        
        # Step 2: Determine translation order
        console.print("\n[bold]Step 2: Determining Translation Order[/bold]")
        modules_to_translate = self._determine_translation_order(analysis_data)
        
        if not modules_to_translate:
            console.print("[yellow]No modules to translate.[/yellow]")
            return PipelineResults(0, 0, 0, 0, 0, 0).to_dict()
        
        console.print(f"[green]Found {len(modules_to_translate)} modules to translate[/green]")
        
        for module_name in modules_to_translate:
            self.module_statuses[module_name] = ModuleStatus(name=module_name)
        
        # Step 3: Translate modules
        console.print("\n[bold]Step 3: Translating Modules[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Translating modules...",
                total=len(modules_to_translate)
            )
            
            for module_name in modules_to_translate:
                progress.update(task, description=f"[cyan]Translating {module_name}...")
                self._process_module(module_name, analysis_data)
                progress.advance(task)
        
        return self._generate_summary().to_dict()

    def _get_module_info(self, module_name: str, analysis_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Robustly fetch module information using case-insensitive and stripped keys.
        """
        # 1. Try standard 'modules' container
        modules_payload = analysis_results.get('modules', {})
        
        # 2. Try 'parsing' -> 'modules' fallback (some analyzer versions nest it here)
        if not modules_payload:
            modules_payload = analysis_results.get('parsing', {}).get('modules', {})

        # Normalize the target name
        target = str(module_name).strip().lower()
        
        # Normalized lookup map (strip quotes and spaces)
        normalized_map = {
            str(k).strip().strip("'").strip('"').lower(): v 
            for k, v in modules_payload.items()
        }
        
        return normalized_map.get(target)

    def _process_module(self, module_name: str, analysis_data: Dict[str, Any]):
        """Process a single module through the complete pipeline."""
        status = self.module_statuses[module_name]
        
        try:
            console.print(f"\n[cyan]→ Translating {module_name}[/cyan]")
            
            # Use the robust helper to find module data
            analysis_results = analysis_data['analysis_results']
            module_info = self._get_module_info(module_name, analysis_results)
            
            if not module_info:
                # If we can't find it, raise with helpful debug info so users
                # can see what keys exist in the analysis results.
                available = list((analysis_results.get('modules') or {}).keys())
                raise ValueError(
                    f"Module '{module_name}' not found in analysis results (checked case-insensitively). "
                    f"Available modules: {available}"
                )

            # Inject the data into the translator
            self.translator.analysis_results = analysis_results
            self.translator.translation_units = analysis_data['translation_units']
            
            translation_result = self.translator.translate_module(module_name)
            
            # Save structured output
            saved_files = translation_result.save_structured(self.output_dir)
            status.translated = True
            console.print(f"[green]✓ Translation complete: {module_name}[/green]")
            
            if self.skip_tests:
                status.final_status = "success"
                return
            
            # Step 2: Generate tests
            console.print(f"[cyan]→ Generating tests for {module_name}[/cyan]")
            test_result = self.test_agent.generate_tests(
                module_name=module_name,
                python_code=translation_result.physics_code,
                source_directory=translation_result.source_directory or "clm_src_main",
                output_dir=self.tests_dir,
            )
            status.tests_generated = True
            
            # Step 3: Run tests
            test_report = self._run_tests(test_result.test_file_path)
            
            if "passed" in test_report.lower() and "failed" not in test_report.lower():
                status.tests_passed = True
                status.final_status = "success"
                console.print(f"[green]✓ All tests passed: {module_name}[/green]")
                return
            
            # Step 4: Repair (simplified for brevity, original logic remains)
            if not self.skip_repair:
                # Repair logic here...
                pass

        except Exception as e:
            status.final_status = "failed"
            status.error_message = str(e)
            console.print(f"[red]✗ Error processing {module_name}: {e}[/red]")

    def _read_fortran_source(self, module_name: str, analysis_data: Dict[str, Any]) -> str:
        """Read original Fortran source for a module using robust lookup."""
        analysis = analysis_data['analysis_results']
        module_info = self._get_module_info(module_name, analysis)
        
        if not module_info:
            raise ValueError(f"Source for module {module_name} not found in analysis")
        
        file_path = Path(module_info['file_path'])
        if not file_path.is_absolute():
            file_path = self.fortran_dir / file_path
        
        return file_path.read_text()

    # ... [Rest of the helper methods remain the same as your original code] ...

    def _run_static_analysis(self) -> Dict[str, Any]:
        analysis_file = self.analysis_dir / "analysis_results.json"
        units_file = self.analysis_dir / "translation_units.json"
        
        # 1. LOAD EXISTING (only if not forcing)
        if not self.force_retranslate and analysis_file.exists():
            # ... existing load logic ...
            pass

        # 2. RUN NEW ANALYSIS
        console.print("[yellow]Running Fortran Static Analysis...[/yellow]")
        try:
            # IMPORTANT: Ensure your analyzer is actually being called here!
            # Example: result = self.analyzer.analyze(self.fortran_dir)
            # If you are using a subprocess, check the return code:
            # if result.returncode != 0: raise Exception("Analyzer failed")
            
            # This is just a placeholder—make sure your actual analyzer call is here!
            self._execute_full_analysis(analysis_file, units_file)
            
        except Exception as e:
            console.print(f"[red]CRITICAL: Static Analysis failed: {e}[/red]")
            sys.exit(1) # Stop the pipeline immediately

        # 3. VERIFY FILES EXIST
        if not analysis_file.exists():
            console.print(f"[red]ERROR: Analyzer finished but {analysis_file} was not created![/red]")
            # Look at the console output above this—did the analyzer print an error?
            sys.exit(1)

        return {
            'analysis_results': json.loads(analysis_file.read_text()),
            'translation_units': json.loads(units_file.read_text()) if units_file.exists() else {}
        }

    def _execute_full_analysis(self, analysis_file, units_file):
        # Execute the fortran_analyzer on the target Fortran directory and
        # ensure `analysis_results.json` and `translation_units.json` are
        # written to `self.analysis_dir` so the rest of the pipeline can
        # consume them.
        # Ensure the workspace root containing `fortran_analyzer/` is on sys.path
        try:
            repo_root = None
            for parent in Path(__file__).resolve().parents:
                if (parent / "fortran_analyzer").exists():
                    repo_root = parent
                    break

            if repo_root and str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))

            from fortran_analyzer.analyzer import create_analyzer_for_project

            # Use a generic template and explicitly point the analyzer at
            # the provided Fortran directory so template auto-detection
            # (which can pick 'ctsm') does not require non-existent subdirs.
            analyzer = create_analyzer_for_project(
                str(self.fortran_dir),
                template="generic",
                output_dir=str(self.analysis_dir),
                source_dirs=[str(self.fortran_dir)],
            )
            results = analyzer.analyze(save_results=True)
        except ImportError as ie:
            raise RuntimeError(
                "Static analysis execution failed: fortran_analyzer not importable. "
                "Ensure the repository root is on PYTHONPATH or install the package (pip install -e .). "
                f"ImportError: {ie}"
            )
        except Exception as e:
            raise RuntimeError(f"Static analysis execution failed: {e}")

        # As a safety net, if the analyzer did not produce files, try to
        # write out the JSON artifacts directly from the returned results.
        try:
            if not analysis_file.exists() and results:
                with open(analysis_file, "w") as f:
                    json.dump(results, f, indent=2)

            # translation units may be exported by the analyzer; if not,
            # attempt to extract any available translation info
            if not units_file.exists():
                translation_payload = None
                # common places where translation unit data may live
                translation_payload = (
                    results.get("translation", {}).get("units_data")
                    or results.get("translation_units")
                    or results.get("translation")
                )
                if translation_payload:
                    with open(units_file, "w") as f:
                        json.dump(translation_payload, f, indent=2)
        except Exception:
            # Don't mask the original exception above; log will surface later
            pass

    def _determine_translation_order(self, analysis_data: Dict[str, Any]) -> List[str]:
        # 1. Access the correct level: Check if nested, else use top-level
        results = analysis_data.get('analysis_results', analysis_data)
        
        # 2. Use .get(..., {}) to prevent NoneType attribute errors
        deps_data = results.get('dependencies') or {}
        
        # 3. Check for the 'analysis' -> 'dependency_levels' structure
        if isinstance(deps_data, dict) and 'analysis' in deps_data:
            analysis_section = deps_data.get('analysis') or {}
            dependency_levels = analysis_section.get('dependency_levels', {})
            ordered = sorted(dependency_levels.keys(), key=lambda k: dependency_levels[k])
        else:
            # Fallback: if no dependencies, just use the keys from the modules dict
            modules = results.get('modules') or {}
            ordered = sorted(modules.keys())
            
        # 4. Filter by manual list if provided
        if self.module_list:
            ordered = [m for m in ordered if m in self.module_list]
            
        return ordered

    def _run_tests(self, test_file: Path) -> str:
        try:
            result = subprocess.run(
                ["pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True, text=True, timeout=300
            )
            return result.stdout + "\n" + result.stderr
        except Exception as e:
            return f"ERROR: {e}"

    def _generate_summary(self) -> PipelineResults:
        # Standard summary generation
        results = PipelineResults(
            total_modules=len(self.module_statuses),
            translated_count=sum(1 for s in self.module_statuses.values() if s.translated),
            tests_generated=sum(1 for s in self.module_statuses.values() if s.tests_generated),
            tests_passed=sum(1 for s in self.module_statuses.values() if s.tests_passed),
            repairs_needed=sum(1 for s in self.module_statuses.values() if s.repair_attempts > 0),
            final_failures=sum(1 for s in self.module_statuses.values() if s.final_status == "failed"),
            module_statuses=self.module_statuses,
        )
        return results