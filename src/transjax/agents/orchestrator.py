"""
Orchestrator Agent - Manages the complete Fortran-to-JAX translation pipeline.

Updated with robust, case-insensitive module lookup logic to prevent
'Module not found' errors during Step 3.
"""

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from transjax.agents.repair_agent import RepairAgent
from transjax.agents.test_agent import TestAgent
from transjax.agents.translator import TranslatorAgent
from transjax.agents.utils.config_loader import get_llm_config
from transjax.agents.utils.translation_state import TranslationStateManager
from transjax.analyzer.analyzer import create_analyzer_for_project

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
        gcm_model_name: Optional[str] = None,
        verbose: bool = False,
        analysis_dir: Optional[Path] = None,
    ):
        self.fortran_dir = fortran_dir
        self.output_dir = output_dir
        self.max_repair_iterations = max_repair_iterations
        self.skip_tests = skip_tests
        self.skip_repair = skip_repair
        self.force_retranslate = force_retranslate
        self.module_list = module_list  # preserved exactly as supplied by the user
        self.gcm_model_name = gcm_model_name or "unspecified"
        self.verbose = verbose

        # Create output directories
        self.src_dir = output_dir / "src"
        self.tests_dir = output_dir / "tests"
        self.docs_dir = output_dir / "docs"
        self.reports_dir = output_dir / "reports"
        # Internal analysis dir (used when we run analysis ourselves)
        self._internal_analysis_dir = output_dir / "static_analysis"

        for d in [self.src_dir, self.tests_dir, self.docs_dir, self.reports_dir, self._internal_analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Resolve analysis directory:
        # Priority: explicit --analysis-dir  →  standard `transjax analyze` output
        #           →  internal output/static_analysis/
        if analysis_dir is not None:
            self.analysis_dir = analysis_dir
        else:
            std_analyze_dir = fortran_dir / "transjax_analysis"
            if std_analyze_dir.exists() and (std_analyze_dir / "analysis_results.json").exists():
                self.analysis_dir = std_analyze_dir
                console.print(
                    f"[dim]Found existing analysis at {self.analysis_dir} "
                    f"(from 'transjax analyze'). Will reuse unless --force is set.[/dim]"
                )
            else:
                self.analysis_dir = self._internal_analysis_dir

        # Initialize agents
        llm_config = get_llm_config()
        self.translator = TranslatorAgent(
            model=model or llm_config.get("model"),
            temperature=temperature if temperature is not None else llm_config.get("temperature"),
            fortran_root=fortran_dir,
            gcm_model_name=self.gcm_model_name,
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

        # In-session state (ModuleStatus objects populated during this run)
        self.completed_modules: Set[str] = set()
        self.module_statuses: Dict[str, ModuleStatus] = {}

        # Persistent state manager — loaded lazily in run() once analysis_dir is resolved
        self.state_manager: Optional[TranslationStateManager] = None

    def run(self) -> Dict[str, Any]:
        """Execute the complete translation pipeline."""
        console.print("\n[bold cyan]Starting Fortran-to-JAX Translation Pipeline[/bold cyan]\n")

        # Step 1: Static Analysis
        analysis_data = self._run_static_analysis()

        # Step 2: Load / initialise persistent state
        console.print("\n[bold]Step 2: Loading Translation State[/bold]")
        self._init_state_manager(analysis_data)

        # Step 3: Determine which modules to translate this run
        console.print("\n[bold]Step 3: Determining Translation Order[/bold]")
        modules_to_translate = self._determine_translation_order(analysis_data)

        if not modules_to_translate:
            console.print("[yellow]No modules to translate — all done or no modules found.[/yellow]")
            if self.state_manager:
                self.state_manager.save()
            return {**PipelineResults(0, 0, 0, 0, 0, 0).to_dict(),
                    "state_file": str(self.state_manager.state_path) if self.state_manager else ""}

        console.print(f"[green]{len(modules_to_translate)} module(s) selected for this run[/green]")

        for module_name in modules_to_translate:
            self.module_statuses[module_name] = ModuleStatus(name=module_name)
            # If force-retranslating, reset state so we don't skip it
            if self.force_retranslate:
                self.state_manager.reset_module(module_name)

        # Step 4: Translate modules
        console.print("\n[bold]Step 4: Translating Modules[/bold]")
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
                # Persist progress after every module so partial runs are recoverable
                self.state_manager.save()
                progress.advance(task)

        summary = self._generate_summary()
        self.state_manager.save()

        return {
            **summary.to_dict(),
            "state_file": str(self.state_manager.state_path),
            "state_summary": self.state_manager.get_summary(),
        }

    def _init_state_manager(self, analysis_data: Dict[str, Any]) -> None:
        """
        Load or create the persistent TranslationStateManager.

        Attempts to initialise the ordered module list from:
          1. ``translation_order.json`` written by ``transjax analyze``
          2. Inline ``file_translation_order`` inside ``analysis_results.json``
          3. Fallback: ``dependency_levels`` dict from ``analysis_results.json``
        """
        self.state_manager = TranslationStateManager(
            output_dir=self.output_dir,
            fortran_dir=self.fortran_dir,
            analysis_dir=self.analysis_dir,
        )
        self.state_manager.load()

        added = 0

        # 1. Try translation_order.json (written by transjax analyze / our new code)
        order_file = self.analysis_dir / "translation_order.json"
        if not order_file.exists():
            # Also check internal analysis dir in case convert ran its own analysis
            order_file = self._internal_analysis_dir / "translation_order.json"

        if order_file.exists():
            try:
                order_data = json.loads(order_file.read_text())
                file_order = order_data.get("files", [])
                added = self.state_manager.initialize_from_order(file_order)
                if added:
                    console.print(
                        f"[dim]Initialised state from {order_file} "
                        f"({added} modules)[/dim]"
                    )
            except Exception as exc:
                logger.warning("Could not load translation_order.json: %s", exc)

        # 2. Inline file_translation_order in analysis_results
        if not added:
            results = analysis_data.get("analysis_results", {})
            file_order = (
                results.get("translation", {}).get("file_translation_order", [])
            )
            if file_order:
                added = self.state_manager.initialize_from_order(file_order)
                if added:
                    console.print(
                        f"[dim]Initialised state from inline file_translation_order "
                        f"({added} modules)[/dim]"
                    )

        # 3. Fallback: dependency_levels
        if not added:
            results = analysis_data.get("analysis_results", {})
            dep_levels = (
                results.get("dependencies", {})
                .get("analysis", {})
                .get("dependency_levels", {})
            )
            all_modules = list(
                (results.get("modules") or
                 results.get("parsing", {}).get("modules", {})).keys()
            )
            if all_modules:
                added = self.state_manager.initialize_from_module_list(
                    all_modules, dependency_levels=dep_levels
                )
                if added:
                    console.print(
                        f"[dim]Initialised state from dependency_levels "
                        f"({added} modules)[/dim]"
                    )

        # 4. Infer already-done modules from existing output files (sync)
        synced = self.state_manager.sync_from_output(self.src_dir)
        if synced:
            console.print(
                f"[dim]Detected {synced} already-translated module(s) "
                "from output directory.[/dim]"
            )

        state_summary = self.state_manager.get_summary()
        console.print(
            f"[cyan]State:[/cyan] "
            f"{state_summary['translated']}/{state_summary['total']} translated "
            f"({state_summary['percent_done']}%), "
            f"{state_summary['pending']} pending, "
            f"{state_summary['failed']} failed"
        )
        if state_summary["next_suggested"]:
            console.print(
                f"[cyan]Next suggested:[/cyan] "
                f"[bold]{state_summary['next_suggested']}[/bold] "
                f"(depth {state_summary['next_depth']})"
            )
        self.state_manager.save()

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
            console.print(f"\n[cyan]-> Translating {module_name}[/cyan]")

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
            translation_result.save_structured(self.output_dir)
            status.translated = True
            console.print(f"[green]Translation complete: {module_name}[/green]")

            if self.skip_tests:
                status.final_status = "success"
                return

            # Step 2: Generate tests
            console.print(f"[cyan]-> Generating tests for {module_name}[/cyan]")
            test_result = self.test_agent.generate_tests(
                module_name=module_name,
                python_code=translation_result.physics_code,
                source_directory=translation_result.source_directory,  # Optional[str]
                output_dir=self.tests_dir,
            )
            status.tests_generated = True

            # Step 3: Run tests
            # TestGenerationResult.save() writes to <tests_dir>/test_<module>.py
            test_file_path = self.tests_dir / f"test_{module_name}.py"
            test_report = self._run_tests(test_file_path)

            if "passed" in test_report.lower() and "failed" not in test_report.lower():
                status.tests_passed = True
                status.final_status = "success"
                console.print(f"[green]All tests passed: {module_name}[/green]")
                return

            # Step 4: Repair
            if not self.skip_repair:
                try:
                    fortran_code = self._read_fortran_source(module_name, analysis_data)
                except Exception:
                    fortran_code = "(Fortran source not available)"

                repair_result = self.repair_agent.repair_translation(
                    module_name=module_name,
                    fortran_code=fortran_code,
                    failed_python_code=translation_result.physics_code,
                    test_report=test_report,
                    test_file_path=test_file_path,
                    output_dir=self.reports_dir,
                )
                status.repair_attempts = repair_result.iterations

                if repair_result.all_tests_passed:
                    # Overwrite the saved JAX file with the repaired code
                    source_subdir = translation_result.source_directory
                    corrected_path = (
                        self.src_dir / source_subdir / f"{module_name}.py"
                        if source_subdir else self.src_dir / f"{module_name}.py"
                    )
                    corrected_path.write_text(repair_result.corrected_python_code)
                    status.tests_passed = True
                    status.final_status = "success"
                    console.print(f"[green]Repair successful: {module_name}[/green]")
                else:
                    status.final_status = "failed"
                    status.error_message = (
                        f"Repair did not converge after {repair_result.iterations} iteration(s)"
                    )
                    console.print(f"[red]Repair failed after {repair_result.iterations} iteration(s): {module_name}[/red]")
            else:
                status.final_status = "failed"

        except Exception as e:
            status.final_status = "failed"
            status.error_message = str(e)
            console.print(f"[red]Error processing {module_name}: {e}[/red]")

        finally:
            # Always persist module outcome to the state manager
            if self.state_manager:
                self.state_manager.mark_module(
                    module_name,
                    status="success" if status.final_status == "success" else "failed",
                    tests_passed=status.tests_passed,
                    repair_attempts=status.repair_attempts,
                    error_message=status.error_message,
                )

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

    def _run_static_analysis(self) -> Dict[str, Any]:
        analysis_file = self.analysis_dir / "analysis_results.json"
        units_file = self.analysis_dir / "translation_units.json"

        # 1. LOAD EXISTING (skip re-analysis unless --force)
        if not self.force_retranslate and analysis_file.exists():
            console.print(f"[dim]Reusing existing analysis: {analysis_file}[/dim]")
            return {
                'analysis_results': json.loads(analysis_file.read_text()),
                'translation_units': json.loads(units_file.read_text()) if units_file.exists() else {},
            }

        # 2. RUN NEW ANALYSIS — always save into the internal dir so we don't
        #    overwrite a directory the user manually pointed to.
        run_dir = self._internal_analysis_dir
        run_analysis_file = run_dir / "analysis_results.json"
        run_units_file = run_dir / "translation_units.json"

        console.print("\n[bold]Step 1: Running Fortran Static Analysis[/bold]")
        console.print(f"[dim]Saving analysis to {run_dir}[/dim]")
        try:
            self._execute_full_analysis(run_analysis_file, run_units_file)
        except Exception as e:
            console.print(f"[red]CRITICAL: Static Analysis failed: {e}[/red]")
            sys.exit(1)

        # 3. VERIFY FILES EXIST
        if not run_analysis_file.exists():
            console.print(f"[red]ERROR: Analyzer finished but {run_analysis_file} was not created![/red]")
            sys.exit(1)

        # Update self.analysis_dir to point at where we actually saved
        self.analysis_dir = run_dir

        return {
            'analysis_results': json.loads(run_analysis_file.read_text()),
            'translation_units': json.loads(run_units_file.read_text()) if run_units_file.exists() else {},
        }

    def _execute_full_analysis(self, analysis_file: Path, units_file: Path) -> None:
        """Execute the Fortran analyzer and write results to disk."""
        output_dir = str(analysis_file.parent)
        try:
            analyzer = create_analyzer_for_project(
                str(self.fortran_dir),
                template="auto",
                output_dir=output_dir,
            )
            results = analyzer.analyze(save_results=True)
        except Exception as e:
            raise RuntimeError(f"Static analysis execution failed: {e}")

        # Safety net: write results directly if the analyzer didn't produce the files.
        if not analysis_file.exists() and results:
            with open(analysis_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

        if not units_file.exists():
            # The decomposer saves translation_units.json alongside analysis_results.json.
            # If it's missing, try to reconstruct a minimal payload from in-memory results.
            units_data = results.get("translation_units")
            if not units_data:
                # Older analyzer shape: units count lives at results["translation"]["units"]
                # but the actual list is written by the decomposer separately.
                pass  # Nothing useful to write; leave file absent (handled upstream)
            if units_data:
                with open(units_file, "w") as f:
                    json.dump(units_data, f, indent=2, default=str)

    def _determine_translation_order(self, analysis_data: Dict[str, Any]) -> List[str]:
        """
        Determine the ordered list of modules to translate in this run.

        Priority:
          1. If ``--modules`` was given explicitly, honour that list exactly
             (filtered through the dependency order for correctness).
          2. Otherwise, use the persistent state manager's pending module list
             (which is already in dependency order).
          3. Unless ``--force``, skip modules already marked ``success``.
        """
        # --- Case 1: explicit module list from the user ---
        if self.module_list:
            # Preserve dependency order for the requested subset
            if self.state_manager and self.state_manager.get_ordered_modules():
                state_order = [
                    e["module"]
                    for e in self.state_manager.get_ordered_modules()
                    if e["module"] in self.module_list
                ]
                # Append any explicitly requested modules not in state (edge case)
                in_state = set(state_order)
                extras = [m for m in self.module_list if m not in in_state]
                ordered = state_order + extras
            else:
                ordered = list(self.module_list)

            if not self.force_retranslate and self.state_manager:
                skipped = [m for m in ordered if self.state_manager.is_translated(m)]
                if skipped:
                    console.print(
                        f"[dim]Skipping {len(skipped)} already-translated module(s) "
                        f"(use --force to re-translate): "
                        f"{', '.join(skipped[:5])}"
                        f"{'…' if len(skipped) > 5 else ''}[/dim]"
                    )
                ordered = [m for m in ordered if not self.state_manager.is_translated(m)]

            return ordered

        # --- Case 2: state-driven automatic order ---
        if self.state_manager and self.state_manager.get_ordered_modules():
            if self.force_retranslate:
                # Re-translate everything in order
                return [e["module"] for e in self.state_manager.get_ordered_modules()]
            else:
                pending = self.state_manager.get_pending_modules()
                failed = [
                    e["module"]
                    for e in self.state_manager.get_ordered_modules()
                    if e["status"] == "failed"
                ]
                # Pending first (in order), then failed (retry), then done = skip
                return pending + [m for m in failed if m not in pending]

        # --- Case 3: no state, fall back to raw analysis ---
        results = analysis_data.get('analysis_results', analysis_data)
        deps_data = results.get('dependencies') or {}
        if isinstance(deps_data, dict) and 'analysis' in deps_data:
            dep_levels = (deps_data.get('analysis') or {}).get('dependency_levels', {})
            return sorted(dep_levels.keys(), key=lambda k: dep_levels[k])
        modules = results.get('modules') or {}
        return sorted(modules.keys())

    def _run_tests(self, test_file: Path) -> str:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True, text=True, timeout=300
            )
            return result.stdout + "\n" + result.stderr
        except Exception as e:
            return f"ERROR: {e}"

    def _generate_summary(self) -> PipelineResults:
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
