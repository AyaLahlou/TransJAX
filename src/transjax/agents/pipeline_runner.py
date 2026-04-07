"""
Full-pipeline runner — end-to-end Fortran → JAX translation with parity gating.

Iterates over every module in dependency order (from translation_order.json)
and runs four steps for each:

  Step 1  FTest     Generate Fortran functional-test drivers (FtestAgent)
  Step 2  Golden    Capture golden I/O reference data (GoldenAgent)
  Step 3  Translate Translate Fortran → JAX/Python (TranslatorAgent)
  Step 4  Parity    Run numerical parity tests; auto-repair on failure
                    (ParityRepairAgent → wraps ParityAgent + repair loop)

State is persisted in <output_dir>/translation_state.json via
TranslationStateManager so the pipeline can be interrupted and resumed.
Each step is also guarded by file-existence checks so previously completed
steps are skipped automatically (override with --force-* flags).

Output directory layout
-----------------------
<output_dir>/
├── translation_state.json          # per-module status (TranslationStateManager)
├── pipeline_state.json             # per-module per-step details
├── ftest/<module>/                 # FtestAgent output
│   ├── ftest_report.json
│   ├── drivers/
│   └── tests/golden/<module>_<sub>.json
├── jax/src/<module>.py             # TranslatorAgent output
├── jax/src/<module>_params.py      #   (optional params module)
└── parity/<module>/                # ParityRepairAgent output
    ├── parity_report.json
    └── docs/<module>_numerical_parity_repair_PASS|FAIL.md
"""

import json
import logging
import re
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from transjax.agents.ftest_agent import FtestAgent
from transjax.agents.golden_agent import GoldenAgent
from transjax.agents.parity_repair_agent import ParityRepairAgent
from transjax.agents.translator import TranslatorAgent
from transjax.agents.utils.config_loader import get_llm_config
from transjax.agents.utils.git_coordinator import GitCoordinator
from transjax.agents.utils.translation_state import TranslationStateManager

logger = logging.getLogger(__name__)
console = Console()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Per-module, per-step state
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    status: str = "pending"     # pending | skipped | success | failed
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    detail: Optional[str] = None

    def start(self) -> None:
        self.started_at = _now_iso()
        self.status = "running"

    def succeed(self, detail: str = "") -> None:
        self.finished_at = _now_iso()
        self.status = "success"
        self.detail = detail or self.detail

    def skip(self, reason: str = "") -> None:
        self.status = "skipped"
        self.detail = reason

    def fail(self, error: str) -> None:
        self.finished_at = _now_iso()
        self.status = "failed"
        self.error = error


@dataclass
class ModulePipelineState:
    """Full pipeline record for one module."""
    module: str
    fortran_file: str
    rank: int
    depth: int
    ftest:     StepResult = field(default_factory=StepResult)
    golden:    StepResult = field(default_factory=StepResult)
    translate: StepResult = field(default_factory=StepResult)
    parity:    StepResult = field(default_factory=StepResult)

    @property
    def overall_status(self) -> str:
        # Overall status is driven by the two steps that determine JAX correctness:
        # translate and parity.  FTest/golden failures are infrastructure failures
        # (e.g. no compiler) and do not invalidate a successful translation.
        if self.translate.status == "failed":
            return "failed"
        if self.parity.status == "failed":
            return "failed"
        if self.parity.status == "success":
            return "success"
        return "in_progress"


# ---------------------------------------------------------------------------
# Pipeline state persistence
# ---------------------------------------------------------------------------

class PipelineStateStore:
    """
    Persists per-module, per-step state to pipeline_state.json.

    Complement to TranslationStateManager: that file tracks translation
    status; this one tracks ftest/golden/parity steps.
    """

    _FILE = "pipeline_state.json"

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / self._FILE
        self._modules: Dict[str, ModulePipelineState] = {}

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            for entry in raw.get("modules", []):
                m = entry["module"]
                self._modules[m] = ModulePipelineState(
                    module=m,
                    fortran_file=entry.get("fortran_file", ""),
                    rank=entry.get("rank", 0),
                    depth=entry.get("depth", 0),
                    ftest=StepResult(**entry.get("ftest", {})),
                    golden=StepResult(**entry.get("golden", {})),
                    translate=StepResult(**entry.get("translate", {})),
                    parity=StepResult(**entry.get("parity", {})),
                )
        except Exception as exc:
            logger.warning("Could not load pipeline_state.json: %s", exc)

    def save(self) -> None:
        data = {
            "last_updated": _now_iso(),
            "modules": [
                {
                    "module":       ms.module,
                    "fortran_file": ms.fortran_file,
                    "rank":         ms.rank,
                    "depth":        ms.depth,
                    "overall":      ms.overall_status,
                    "ftest":        asdict(ms.ftest),
                    "golden":       asdict(ms.golden),
                    "translate":    asdict(ms.translate),
                    "parity":       asdict(ms.parity),
                }
                for ms in self._modules.values()
            ],
        }
        self.path.write_text(json.dumps(data, indent=2))

    def get_or_create(self, module: str, fortran_file: str,
                      rank: int, depth: int) -> ModulePipelineState:
        if module not in self._modules:
            self._modules[module] = ModulePipelineState(
                module=module,
                fortran_file=fortran_file,
                rank=rank,
                depth=depth,
            )
        return self._modules[module]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """
    Runs the full end-to-end pipeline for every module in translation order.

    Usage
    -----
    runner = PipelineRunner(
        fortran_dir   = Path("/path/to/fortran"),
        analysis_dir  = Path("/path/to/analysis"),  # has translation_order.json
        output_dir    = Path("/path/to/pipeline_out"),
        gcm_model_name= "CTSM",
    )
    runner.run()
    """

    def __init__(
        self,
        fortran_dir: Path,
        analysis_dir: Path,
        output_dir: Path,
        gcm_model_name: str = "generic ESM",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        ftest_build_dir: Optional[Path] = None,
        ftest_compiler: str = "nvfortran",
        ftest_netcdf_inc: Optional[str] = None,
        ftest_netcdf_lib: Optional[str] = None,
        golden_n_cases: int = 5,
        parity_rtol: float = 1e-10,
        parity_atol: float = 1e-12,
        max_repair_iterations: int = 5,
        module_filter: Optional[List[str]] = None,
        force: bool = False,
        force_ftest: bool = False,
        force_golden: bool = False,
        force_translate: bool = False,
        force_parity: bool = False,
        skip_ftest: bool = False,
        skip_golden: bool = False,
        run_integrate: bool = False,
        max_integration_repair_iterations: int = 5,
        use_tmux: bool = False,
        auto_git_commit: bool = False,
        verbose: bool = False,
    ) -> None:
        self.fortran_dir    = Path(fortran_dir).resolve()
        self.analysis_dir   = Path(analysis_dir).resolve()
        self.output_dir     = Path(output_dir).resolve()
        self.gcm_model_name = gcm_model_name

        # Sub-directories
        self.ftest_base_dir  = self.output_dir / "ftest"
        self.jax_src_dir     = self.output_dir / "jax" / "src"
        self.parity_base_dir = self.output_dir / "parity"

        # Ftest / golden settings
        self.ftest_build_dir   = Path(ftest_build_dir).resolve() if ftest_build_dir else None
        self.ftest_compiler    = ftest_compiler
        self.ftest_netcdf_inc  = ftest_netcdf_inc
        self.ftest_netcdf_lib  = ftest_netcdf_lib
        self.golden_n_cases    = golden_n_cases

        # Parity settings
        self.parity_rtol          = parity_rtol
        self.parity_atol          = parity_atol
        self.max_repair_iterations = max_repair_iterations

        # Filtering
        self.module_filter = [m.lower() for m in module_filter] if module_filter else None

        # Force / skip flags
        self.force           = force
        self.force_ftest     = force or force_ftest
        self.force_golden    = force or force_golden
        self.force_translate = force or force_translate
        self.force_parity    = force or force_parity
        self.skip_ftest      = skip_ftest
        self.skip_golden     = skip_golden
        self.run_integrate   = run_integrate
        self.max_integration_repair_iterations = max_integration_repair_iterations
        self.use_tmux        = use_tmux
        self.verbose         = verbose

        # Git coordination — auto-commit after each successful module
        self.git_coordinator = GitCoordinator(
            repo_root=Path(".").resolve(),
            output_dir=self.output_dir,
            auto_commit=auto_git_commit,
        )

        # LLM config
        llm_cfg = get_llm_config()
        self.model       = model or llm_cfg.get("model", "claude-sonnet-4-6")
        self.temperature = temperature if temperature is not None else llm_cfg.get("temperature", 0.0)

        # State managers — initialised in run()
        self.state_mgr: Optional[TranslationStateManager] = None
        self.pipeline_store: Optional[PipelineStateStore] = None

    # ---------------------------------------------------------------------- #
    # Public entry point                                                       #
    # ---------------------------------------------------------------------- #

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline.  Returns a summary dict."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jax_src_dir.mkdir(parents=True, exist_ok=True)

        # ── Load translation order ─────────────────────────────────────────
        order = self._load_translation_order()
        if not order:
            console.print("[red]No translation_order.json found in analysis_dir.[/red]")
            console.print(
                f"  Run: transjax analyze {self.fortran_dir} "
                f"-o {self.analysis_dir}"
            )
            return {"error": "translation_order.json not found"}

        console.print(
            Panel.fit(
                f"[white]Fortran dir:   [/white] {self.fortran_dir}\n"
                f"[white]Analysis dir: [/white]  {self.analysis_dir}\n"
                f"[white]Output dir:   [/white]  {self.output_dir}\n"
                f"[white]ESM model:    [/white]  {self.gcm_model_name}\n"
                f"[white]Modules:      [/white]  {len(order)} in translation order\n"
                f"[white]Tolerances:   [/white]  rtol={self.parity_rtol}, atol={self.parity_atol}",
                title="[bold cyan]TransJAX Full Pipeline[/bold cyan]",
            )
        )

        # ── State managers ─────────────────────────────────────────────────
        self.state_mgr = TranslationStateManager(
            output_dir=self.output_dir,
            fortran_dir=self.fortran_dir,
            analysis_dir=self.analysis_dir,
        )
        self.state_mgr.load()
        self.state_mgr.initialize_from_order(order)

        self.pipeline_store = PipelineStateStore(self.output_dir)
        self.pipeline_store.load()

        # ── Iterate modules ────────────────────────────────────────────────
        total = passed = failed = 0

        for file_entry in order:
            for module_name in file_entry.get("modules", []):
                if self.module_filter and module_name.lower() not in self.module_filter:
                    continue

                total += 1
                ok = self._run_module(
                    module_name=module_name,
                    fortran_file=Path(file_entry["file"]),
                    rank=file_entry.get("rank", 0),
                    depth=file_entry.get("depth", 0),
                )
                if ok:
                    passed += 1
                else:
                    failed += 1

                # Save state after every module
                self.state_mgr.save()
                self.pipeline_store.save()

        self._print_final_summary()

        summary: Dict[str, Any] = {
            "total":  total,
            "passed": passed,
            "failed": failed,
            "pipeline_state": str(self.pipeline_store.path),
            "translation_state": str(self.state_mgr.state_path),
        }

        # ── Optional integration step ──────────────────────────────────────
        if self.run_integrate:
            integration_result = self._step_integrate()
            summary["integration_status"] = integration_result.final_status
            summary["integration_dir"] = str(integration_result.integration_dir)

        return summary

    # ---------------------------------------------------------------------- #
    # Per-module pipeline                                                      #
    # ---------------------------------------------------------------------- #

    def _run_module(
        self,
        module_name: str,
        fortran_file: Path,
        rank: int,
        depth: int,
    ) -> bool:
        """Run all four pipeline steps for one module.  Returns True on success."""
        ms = self.pipeline_store.get_or_create(
            module_name, str(fortran_file), rank, depth
        )

        console.print(
            f"\n[bold cyan]{'━'*60}[/bold cyan]"
            f"\n[bold cyan]Module:[/bold cyan] [white]{module_name}[/white]"
            f"  [dim](rank {rank}, depth {depth})[/dim]"
            f"\n[bold cyan]{'━'*60}[/bold cyan]"
        )
        console.print(f"[dim]Fortran: {fortran_file}[/dim]\n")

        if not fortran_file.exists():
            console.print(f"  [yellow]⚠ Fortran file not found — skipping module[/yellow]")
            return False

        # Module-scoped output dirs
        module_ftest_dir  = self.ftest_base_dir / module_name
        module_golden_dir = module_ftest_dir / "tests" / "golden"
        module_ftest_report = module_ftest_dir / "ftest_report.json"
        jax_python_file   = self.jax_src_dir / f"{module_name}.py"
        module_parity_dir = self.parity_base_dir / module_name

        # ── Step 1: FTest ─────────────────────────────────────────────────
        ftest_ok = self._step_ftest(
            ms=ms,
            module_name=module_name,
            module_ftest_dir=module_ftest_dir,
            module_ftest_report=module_ftest_report,
        )

        # ── Step 2: Golden ────────────────────────────────────────────────
        golden_ok = self._step_golden(
            ms=ms,
            module_name=module_name,
            module_ftest_dir=module_ftest_dir,
            module_golden_dir=module_golden_dir,
            ftest_ok=ftest_ok,
        )

        # ── Step 3: Translate ─────────────────────────────────────────────
        translate_ok = self._step_translate(
            ms=ms,
            module_name=module_name,
            fortran_file=fortran_file,
            jax_python_file=jax_python_file,
        )

        # ── Step 4: Parity + Repair ───────────────────────────────────────
        parity_ok = self._step_parity(
            ms=ms,
            module_name=module_name,
            fortran_file=fortran_file,
            jax_python_file=jax_python_file,
            module_golden_dir=module_golden_dir,
            module_ftest_report=module_ftest_report,
            module_parity_dir=module_parity_dir,
            translate_ok=translate_ok,
        )

        # ── Update translation state ──────────────────────────────────────
        overall_ok = translate_ok and parity_ok
        status_str = "success" if overall_ok else "failed"
        self.state_mgr.mark_module(
            module_name,
            status=status_str,
            tests_passed=parity_ok,
            repair_attempts=_count_repair_attempts(ms),
            error_message=_collect_errors(ms) or None,
        )

        # ── Git coordination ──────────────────────────────────────────────
        if overall_ok:
            parity_str = "PASS" if parity_ok else "FAIL"
            self.git_coordinator.append_changelog(
                module_name, status_str,
                notes=f"parity={parity_str}",
            )
            self.git_coordinator.commit_module(
                module_name,
                status=status_str,
                parity_result=parity_str,
                jax_src_dir=self.jax_src_dir,
                reports_dir=self.output_dir / "reports",
            )

        # ── Tmux cleanup ──────────────────────────────────────────────────
        if self.use_tmux:
            _close_tmux_session(module_name)

        icon = "[green]✓[/green]" if overall_ok else "[red]✗[/red]"
        console.print(
            f"\n  {icon} [bold]{module_name}[/bold] — "
            + ("pipeline complete" if overall_ok else "pipeline finished with errors")
        )
        return overall_ok

    # ---------------------------------------------------------------------- #
    # Step implementations                                                     #
    # ---------------------------------------------------------------------- #

    def _step_ftest(
        self,
        ms: ModulePipelineState,
        module_name: str,
        module_ftest_dir: Path,
        module_ftest_report: Path,
    ) -> bool:
        console.print("  [bold][1/4][/bold] FTest — generate Fortran test drivers")

        if self.skip_ftest:
            ms.ftest.skip("--skip-ftest")
            console.print("        [dim]skipped (--skip-ftest)[/dim]")
            return module_ftest_report.exists()

        already_done = (
            module_ftest_report.exists()
            and ms.ftest.status in ("success", "skipped")
        )
        if already_done and not self.force_ftest:
            ms.ftest.skip("already done")
            console.print("        [dim]✓ already done — skipped[/dim]")
            return True

        ms.ftest.start()
        try:
            agent = FtestAgent(model=self.model, temperature=self.temperature, use_tmux=self.use_tmux)
            _attach_tmux(agent, module_name, self.output_dir, self.use_tmux)
            agent.run(
                fortran_dir=self.fortran_dir,
                output_dir=module_ftest_dir,
                build_dir=self.ftest_build_dir,
                compiler=self.ftest_compiler,
                netcdf_inc=self.ftest_netcdf_inc,
                netcdf_lib=self.ftest_netcdf_lib,
                module_filter=[module_name],
                verbose=self.verbose,
            )
            ms.ftest.succeed(f"report: {module_ftest_report}")
            console.print(f"        [green]✓ done → {module_ftest_report}[/green]")
            return True
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            ms.ftest.fail(err)
            console.print(f"        [red]✗ FTest failed: {err}[/red]")
            if self.verbose:
                console.print(traceback.format_exc())
            return False

    def _step_golden(
        self,
        ms: ModulePipelineState,
        module_name: str,
        module_ftest_dir: Path,
        module_golden_dir: Path,
        ftest_ok: bool,
    ) -> bool:
        console.print("  [bold][2/4][/bold] Golden — capture reference I/O data")

        if self.skip_golden:
            ms.golden.skip("--skip-golden")
            console.print("        [dim]skipped (--skip-golden)[/dim]")
            return any(module_golden_dir.glob("*.json")) if module_golden_dir.exists() else False

        existing_golden = list(module_golden_dir.glob("*.json")) if module_golden_dir.exists() else []
        already_done = (
            bool(existing_golden)
            and ms.golden.status in ("success", "skipped")
        )
        if already_done and not self.force_golden:
            ms.golden.skip(f"already done ({len(existing_golden)} file(s))")
            console.print(f"        [dim]✓ {len(existing_golden)} golden file(s) — skipped[/dim]")
            return True

        if not ftest_ok:
            ms.golden.skip("ftest step failed or skipped")
            console.print("        [yellow]⚠ skipped — FTest step did not succeed[/yellow]")
            # Still OK if golden files already exist from a previous run
            return bool(existing_golden)

        report_path = module_ftest_dir / "ftest_report.json"
        if not report_path.exists():
            ms.golden.skip("ftest_report.json not found")
            console.print("        [yellow]⚠ skipped — ftest_report.json not found[/yellow]")
            return bool(existing_golden)

        ms.golden.start()
        try:
            agent = GoldenAgent(model=self.model, temperature=self.temperature, use_tmux=self.use_tmux)
            _attach_tmux(agent, module_name, self.output_dir, self.use_tmux)
            result = agent.run(
                ftest_output_dir=module_ftest_dir,
                n_cases=self.golden_n_cases,
                gcm_model_name=self.gcm_model_name,
                verbose=self.verbose,
            )
            n_written = result.golden_written
            ms.golden.succeed(f"{n_written} subroutine(s), dir: {module_golden_dir}")
            console.print(f"        [green]✓ {n_written} subroutine(s) → {module_golden_dir}[/green]")
            return True
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            ms.golden.fail(err)
            console.print(f"        [red]✗ Golden failed: {err}[/red]")
            if self.verbose:
                console.print(traceback.format_exc())
            return bool(existing_golden)   # graceful: existing golden still usable

    def _step_translate(
        self,
        ms: ModulePipelineState,
        module_name: str,
        fortran_file: Path,
        jax_python_file: Path,
    ) -> bool:
        console.print("  [bold][3/4][/bold] Translate — Fortran → JAX/Python")

        already_done = (
            jax_python_file.exists()
            and self.state_mgr is not None
            and self.state_mgr.is_translated(module_name)
        )
        if already_done and not self.force_translate:
            ms.translate.skip(f"already at {jax_python_file}")
            console.print(f"        [dim]✓ already translated — skipped[/dim]")
            return True

        ms.translate.start()
        try:
            analysis_path = self.analysis_dir / "analysis_results.json"
            agent = TranslatorAgent(
                fortran_root=self.fortran_dir,
                gcm_model_name=self.gcm_model_name,
                model=self.model,
                temperature=self.temperature,
                use_tmux=self.use_tmux,
            )
            if analysis_path.exists():
                agent.load_analysis(analysis_path)
            _attach_tmux(agent, module_name, self.output_dir, self.use_tmux)
            result = agent.translate_module(
                module_name=module_name,
                fortran_file=fortran_file,
                output_dir=self.jax_src_dir,
            )
            ms.translate.succeed(f"→ {jax_python_file}")
            console.print(f"        [green]✓ translated → {jax_python_file}[/green]")
            return True
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            ms.translate.fail(err)
            console.print(f"        [red]✗ Translation failed: {err}[/red]")
            if self.verbose:
                console.print(traceback.format_exc())
            return False

    def _step_parity(
        self,
        ms: ModulePipelineState,
        module_name: str,
        fortran_file: Path,
        jax_python_file: Path,
        module_golden_dir: Path,
        module_ftest_report: Path,
        module_parity_dir: Path,
        translate_ok: bool,
    ) -> bool:
        console.print("  [bold][4/4][/bold] Parity — numerical parity test + repair")

        if not translate_ok:
            ms.parity.skip("translation step failed")
            console.print("        [yellow]⚠ skipped — translation did not succeed[/yellow]")
            return False

        if not jax_python_file.exists():
            ms.parity.skip("JAX module file not found")
            console.print(f"        [yellow]⚠ skipped — {jax_python_file} not found[/yellow]")
            return False

        # If golden data is absent, we can still run parity tests in LLM mode.
        # Warn the user but don't block.
        golden_files = list(module_golden_dir.glob("*.json")) if module_golden_dir.exists() else []
        if not golden_files:
            ms.parity.skip("no golden data available")
            console.print(
                "        [yellow]⚠ skipped — no golden JSON files found. "
                "Run the golden step first (requires compiled Fortran drivers).[/yellow]"
            )
            return False

        # Skip if previously passed and not forcing
        parity_report = module_parity_dir / "parity_report.json"
        if not self.force_parity and parity_report.exists():
            try:
                rep = json.loads(parity_report.read_text())
                if rep.get("summary", {}).get("subroutines_failed", 1) == 0:
                    ms.parity.skip("already passed")
                    console.print("        [dim]✓ already passed — skipped[/dim]")
                    return True
            except Exception:
                pass  # corrupt report — re-run

        ms.parity.start()
        try:
            agent = ParityRepairAgent(model=self.model, temperature=self.temperature, use_tmux=self.use_tmux)
            _attach_tmux(agent, module_name, self.output_dir, self.use_tmux)
            result = agent.run(
                python_file=jax_python_file,
                fortran_file=fortran_file,
                golden_dir=module_golden_dir,
                output_dir=module_parity_dir,
                ftest_report_path=module_ftest_report if module_ftest_report.exists() else None,
                rtol=self.parity_rtol,
                atol=self.parity_atol,
                max_iterations=self.max_repair_iterations,
                verbose=self.verbose,
            )

            n_iter  = result.total_iterations
            status  = result.final_status

            if status == "PASS":
                detail = (
                    f"PASS after {n_iter} repair iteration(s)"
                    if n_iter else "PASS — no repair needed"
                )
                ms.parity.succeed(detail)
                console.print(f"        [green]✓ {detail}[/green]")
                return True
            else:
                ms.parity.fail(f"FAIL after {n_iter} repair iteration(s)")
                console.print(
                    f"        [red]✗ Parity FAIL after {n_iter} iteration(s) "
                    f"— see {result.report_path}[/red]"
                )
                return False

        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            ms.parity.fail(err)
            console.print(f"        [red]✗ Parity step error: {err}[/red]")
            if self.verbose:
                console.print(traceback.format_exc())
            return False

    # ---------------------------------------------------------------------- #
    # Integration step                                                          #
    # ---------------------------------------------------------------------- #

    def _step_integrate(self):
        """Run IntegratorAgent after all modules complete."""
        from transjax.agents.integrator_agent import IntegrationResult, IntegratorAgent

        console.print(
            f"\n[bold cyan]{'━'*60}[/bold cyan]"
            f"\n[bold cyan]System Integration[/bold cyan]"
            f"\n[bold cyan]{'━'*60}[/bold cyan]"
        )

        agent = IntegratorAgent(model=self.model, temperature=self.temperature)
        result = agent.run(
            fortran_dir=self.fortran_dir,
            jax_src_dir=self.jax_src_dir,
            output_dir=self.output_dir,
            gcm_model_name=self.gcm_model_name,
            max_repair_iterations=self.max_integration_repair_iterations,
            verbose=self.verbose,
        )

        if result.passed:
            console.print(
                f"\n  [bold green]✓ System integration PASSED[/bold green]\n"
                f"  [dim]integration dir → {result.integration_dir}[/dim]"
            )
        else:
            console.print(
                f"\n  [bold red]✗ System integration {result.final_status}[/bold red]"
            )
            if result.error:
                console.print(f"  [red]{result.error}[/red]")

        return result

    # ---------------------------------------------------------------------- #
    # Summary                                                                  #
    # ---------------------------------------------------------------------- #

    def _print_final_summary(self) -> None:
        if self.pipeline_store is None:
            return

        table = Table(title="Pipeline Summary", show_lines=True)
        table.add_column("Module",     style="cyan", no_wrap=True)
        table.add_column("FTest",      justify="center")
        table.add_column("Golden",     justify="center")
        table.add_column("Translate",  justify="center")
        table.add_column("Parity",     justify="center")
        table.add_column("Overall",    justify="center")

        _icon = {
            "success": "[green]✓[/green]",
            "skipped": "[dim]—[/dim]",
            "pending": "[dim]?[/dim]",
            "failed":  "[red]✗[/red]",
            "running": "[yellow]…[/yellow]",
        }

        totals = {"success": 0, "failed": 0, "other": 0}
        for ms in self.pipeline_store._modules.values():
            table.add_row(
                ms.module,
                _icon.get(ms.ftest.status,     ms.ftest.status),
                _icon.get(ms.golden.status,    ms.golden.status),
                _icon.get(ms.translate.status, ms.translate.status),
                _icon.get(ms.parity.status,    ms.parity.status),
                _icon.get(ms.overall_status,   ms.overall_status),
            )
            if ms.overall_status == "success":
                totals["success"] += 1
            elif ms.overall_status == "failed":
                totals["failed"] += 1
            else:
                totals["other"] += 1

        console.print()
        console.print(table)
        console.print(
            Panel.fit(
                f"[green]{totals['success']} passed[/green]  "
                f"[red]{totals['failed']} failed[/red]  "
                f"[dim]{totals['other']} in-progress/skipped[/dim]\n"
                f"Pipeline state: {self.pipeline_store.path}",
                title="[bold]Final Summary[/bold]",
            )
        )

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    def _load_translation_order(self) -> List[Dict[str, Any]]:
        """Load the file-level translation order from analysis_dir."""
        order_file = self.analysis_dir / "translation_order.json"
        if not order_file.exists():
            return []
        try:
            data = json.loads(order_file.read_text())
            return data.get("files", [])
        except Exception as exc:
            logger.error("Could not parse translation_order.json: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _count_repair_attempts(ms: ModulePipelineState) -> int:
    """Estimate repair attempts from parity step detail string."""
    detail = ms.parity.detail or ""
    m = re.search(r"(\d+)\s+repair", detail)
    return int(m.group(1)) if m else 0


def _collect_errors(ms: ModulePipelineState) -> str:
    errors = []
    for step_name in ("ftest", "golden", "translate", "parity"):
        step: StepResult = getattr(ms, step_name)
        if step.error:
            errors.append(f"{step_name}: {step.error}")
    return "; ".join(errors)


def _attach_tmux(agent: Any, module_name: str, output_dir: Path, use_tmux: bool) -> None:
    """Set the tmux session on an agent if tmux mode is enabled."""
    if not use_tmux:
        return
    session_name = f"transjax-{module_name}"
    work_dir = output_dir / "tmux_work" / module_name
    work_dir.mkdir(parents=True, exist_ok=True)
    agent.set_tmux_session(session_name=session_name, work_dir=work_dir)


def _close_tmux_session(module_name: str) -> None:
    """Kill the tmux session for a module (best-effort, no error on failure)."""
    import subprocess
    session_name = f"transjax-{module_name}"
    subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        capture_output=True,
    )
