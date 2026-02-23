"""
Orchestrator Agent - Manages the complete Fortran-to-JAX translation pipeline.

This agent coordinates all other agents and handles the end-to-end workflow:
1. Static analysis (Fortran analyzer)
2. Translation (TranslatorAgent)
3. Test generation (TestAgent)
4. Test execution (pytest)
5. Repair (RepairAgent) - iterative until tests pass or max attempts reached
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
    
    Workflow:
        1. Run Fortran static analyzer (or load existing analysis)
        2. Determine module translation order (dependency-aware)
        3. For each module:
            a. Check if already translated (skip if not --force)
            b. Translate Fortran -> JAX
            c. Generate tests
            d. Run tests
            e. If failures -> Repair (iterate up to max_repair_iterations)
            f. Save results + reports
        4. Generate final summary report
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
        console.print("[bold]Step 1: Static Analysis[/bold]")
        analysis_data = self._run_static_analysis()
        
        # Step 2: Determine translation order
        console.print("\n[bold]Step 2: Determining Translation Order[/bold]")
        modules_to_translate = self._determine_translation_order(analysis_data)
        
        if not modules_to_translate:
            console.print("[yellow]No modules to translate.[/yellow]")
            return PipelineResults(0, 0, 0, 0, 0, 0).to_dict()
        
        console.print(f"[green]Found {len(modules_to_translate)} modules to translate[/green]")
        
        # Initialize status tracking
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
        
        # Step 4: Generate summary
        results = self._generate_summary()
        
        return results.to_dict()
    
    def _run_static_analysis(self) -> Dict[str, Any]:
        """Run Fortran static analyzer or load existing analysis."""
        analysis_file = self.analysis_dir / "analysis_results.json"
        units_file = self.analysis_dir / "translation_units.json"
        
        # Check if analysis already exists and is recent
        if not self.force_retranslate and analysis_file.exists() and units_file.exists():
            console.print("[yellow]Using existing static analysis[/yellow]")
            try:
                analysis_data = json.loads(analysis_file.read_text())
                units_data = json.loads(units_file.read_text())
                return {
                    'analysis_results': analysis_data,
                    'translation_units': units_data,
                }
            except Exception as e:
                console.print(f"[yellow]Failed to load existing analysis: {e}[/yellow]")
        
        # Run analyzer
        console.print("[cyan]Running Fortran static analyzer...[/cyan]")
        
        try:
            # Import and run the analyzer
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fortran_analyzer"))
            from fortran_analyzer.analyzer import FortranAnalyzer
            
            analyzer = FortranAnalyzer(self.fortran_dir)
            analysis_results = analyzer.analyze()
            
            # Save results
            analysis_file.write_text(json.dumps(analysis_results, indent=2))
            
            # Generate translation units
            from fortran_analyzer.analysis.translation_decomposer import TranslationDecomposer
            decomposer = TranslationDecomposer(analysis_results)
            translation_units = decomposer.generate_translation_units()
            units_file.write_text(json.dumps(translation_units, indent=2))
            
            console.print("[green]✓ Static analysis complete[/green]")
            
            return {
                'analysis_results': analysis_results,
                'translation_units': translation_units,
            }
            
        except Exception as e:
            console.print(f"[red]Error during static analysis: {e}[/red]")
            raise
    
    def _determine_translation_order(self, analysis_data: Dict[str, Any]) -> List[str]:
        """
        Determine module translation order based on dependencies.
        
        Strategy:
        - If user specified modules via --modules, use that list
        - Otherwise, translate all modules in dependency order (leaves first)
        - Skip already-translated modules unless --force
        """
        analysis_results = analysis_data.get('analysis_results', {})
        modules = analysis_results.get('modules', {})
        
        # If user specified modules, use that
        if self.module_list:
            available = set(modules.keys())
            requested = set(self.module_list)
            missing = requested - available
            if missing:
                console.print(f"[yellow]Warning: Modules not found in analysis: {', '.join(missing)}[/yellow]")
            return [m for m in self.module_list if m in available]
        
        # Otherwise, sort by dependencies (topological sort)
        # For now, simple approach: sort by dependency count (leaves first)
        module_list = []
        for name, info in modules.items():
            deps = info.get('dependencies', [])
            module_list.append((name, len(deps)))
        
        # Sort by dependency count (ascending)
        module_list.sort(key=lambda x: x[1])
        ordered = [name for name, _ in module_list]
        
        # Filter out already-translated modules (unless --force)
        if not self.force_retranslate:
            to_translate = []
            for name in ordered:
                # Check if module already exists in output
                src_dir = info.get('source_directory', 'clm_src_main')
                module_file = self.src_dir / src_dir / f"{name}.py"
                if not module_file.exists():
                    to_translate.append(name)
                else:
                    console.print(f"[dim]Skipping {name} (already translated)[/dim]")
                    self.completed_modules.add(name)
            return to_translate
        
        return ordered
    
    def _process_module(self, module_name: str, analysis_data: Dict[str, Any]):
        """Process a single module through the complete pipeline."""
        status = self.module_statuses[module_name]
        
        try:
            # Step 1: Translate
            console.print(f"\n[cyan]→ Translating {module_name}[/cyan]")
            
            # Load analysis for this module
            self.translator.analysis_results = analysis_data['analysis_results']
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
            console.print(f"[green]✓ Tests generated: {module_name}[/green]")
            
            # Step 3: Run tests
            console.print(f"[cyan]→ Running tests for {module_name}[/cyan]")
            test_report = self._run_tests(test_result.test_file_path)
            
            if "passed" in test_report.lower() and "failed" not in test_report.lower():
                status.tests_passed = True
                status.final_status = "success"
                console.print(f"[green]✓ All tests passed: {module_name}[/green]")
                return
            
            # Step 4: Repair if tests failed
            if self.skip_repair:
                status.final_status = "failed"
                status.error_message = "Tests failed, repair skipped"
                console.print(f"[yellow]⚠ Tests failed for {module_name}, repair skipped[/yellow]")
                return
            
            console.print(f"[yellow]⚠ Tests failed for {module_name}, starting repair...[/yellow]")
            
            # Read Fortran source for repair
            fortran_code = self._read_fortran_source(module_name, analysis_data)
            
            repair_result = self.repair_agent.repair_translation(
                module_name=module_name,
                fortran_code=fortran_code,
                failed_python_code=translation_result.physics_code,
                test_report=test_report,
                test_file_path=test_result.test_file_path,
                output_dir=self.reports_dir / "repair_logs" / module_name,
            )
            
            status.repair_attempts = repair_result.iterations
            
            if repair_result.all_tests_passed:
                status.tests_passed = True
                status.final_status = "success"
                console.print(f"[green]✓ Repair successful: {module_name}[/green]")
                
                # Save corrected code
                corrected_file = saved_files['physics']
                corrected_file.write_text(repair_result.corrected_python_code)
            else:
                status.final_status = "failed"
                status.error_message = f"Failed after {repair_result.iterations} repair attempts"
                console.print(f"[red]✗ Repair failed: {module_name}[/red]")
                
        except Exception as e:
            status.final_status = "failed"
            status.error_message = str(e)
            console.print(f"[red]✗ Error processing {module_name}: {e}[/red]")
            if self.verbose:
                console.print_exception()
    
    def _run_tests(self, test_file: Path) -> str:
        """Run pytest on a test file and capture output."""
        try:
            result = subprocess.run(
                ["pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per test file
            )
            return result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired:
            return "ERROR: Test execution timed out after 5 minutes"
        except Exception as e:
            return f"ERROR: Failed to run tests: {e}"
    
    def _read_fortran_source(self, module_name: str, analysis_data: Dict[str, Any]) -> str:
        """Read original Fortran source for a module."""
        analysis = analysis_data['analysis_results']
        module_info = analysis['modules'].get(module_name)
        
        if not module_info:
            raise ValueError(f"Module {module_name} not found in analysis")
        
        file_path = Path(module_info['file_path'])
        if not file_path.is_absolute():
            file_path = self.fortran_dir / file_path
        
        return file_path.read_text()
    
    def _generate_summary(self) -> PipelineResults:
        """Generate final summary of pipeline execution."""
        results = PipelineResults(
            total_modules=len(self.module_statuses),
            translated_count=sum(1 for s in self.module_statuses.values() if s.translated),
            tests_generated=sum(1 for s in self.module_statuses.values() if s.tests_generated),
            tests_passed=sum(1 for s in self.module_statuses.values() if s.tests_passed),
            repairs_needed=sum(1 for s in self.module_statuses.values() if s.repair_attempts > 0),
            final_failures=sum(1 for s in self.module_statuses.values() if s.final_status == "failed"),
            module_statuses=self.module_statuses,
        )
        
        # Create summary table
        table = Table(title="Module Translation Summary", show_header=True, header_style="bold cyan")
        table.add_column("Module", style="white")
        table.add_column("Status", justify="center")
        table.add_column("Tests", justify="center")
        table.add_column("Repairs", justify="center")
        table.add_column("Result", justify="center")
        
        for name, status in self.module_statuses.items():
            status_icon = "✓" if status.translated else "✗"
            tests_icon = "✓" if status.tests_passed else ("⚠" if status.tests_generated else "—")
            repairs_str = str(status.repair_attempts) if status.repair_attempts > 0 else "—"
            
            if status.final_status == "success":
                result_text = "[green]SUCCESS[/green]"
            elif status.final_status == "failed":
                result_text = "[red]FAILED[/red]"
            else:
                result_text = "[yellow]PENDING[/yellow]"
            
            table.add_row(name, status_icon, tests_icon, repairs_str, result_text)
        
        console.print("\n")
        console.print(table)
        
        # Save summary report
        summary_file = self.reports_dir / "translation_summary.json"
        summary_data = {
            'total_modules': results.total_modules,
            'translated_count': results.translated_count,
            'tests_generated': results.tests_generated,
            'tests_passed': results.tests_passed,
            'repairs_needed': results.repairs_needed,
            'final_failures': results.final_failures,
            'modules': {
                name: {
                    'translated': s.translated,
                    'tests_generated': s.tests_generated,
                    'tests_passed': s.tests_passed,
                    'repair_attempts': s.repair_attempts,
                    'final_status': s.final_status,
                    'error_message': s.error_message,
                }
                for name, s in self.module_statuses.items()
            }
        }
        summary_file.write_text(json.dumps(summary_data, indent=2))
        console.print(f"\n[dim]Summary saved to {summary_file}[/dim]")
        
        return results
