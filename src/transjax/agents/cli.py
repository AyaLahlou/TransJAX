"""
Command-line interface for TransJAX.

Single entry point for the full Fortran-to-JAX workflow.

Usage:
    transjax convert /path/to/fortran -o ./jax_output
    transjax analyze /path/to/fortran
    transjax show-config
    transjax init
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="transjax")
def cli():
    """TransJAX — translate Fortran scientific code to JAX.

    TransJAX uses Claude to convert Fortran source code into differentiable
    JAX/Python, preserving physics and applying JAX best practices (pure
    functions, lax primitives, NamedTuples).

    \b
    Recommended workflow:
      1. transjax init                               # set up authentication
      2. transjax analyze /path/to/fortran           # build dependency order
      3. transjax convert /path/to/fortran -o ./out  # translate (resumes automatically)
      4. transjax convert /path/to/fortran --next    # translate next pending module
      5. transjax status ./out                       # view progress dashboard
      6. transjax ftest /path/to/fortran -o ./ftest  # functional test framework
      7. transjax golden ./ftest                     # capture golden reference data
      8. transjax test-parity ./out/src/mod.py \\
             --golden-dir ./ftest/tests/golden       # verify JAX ↔ Fortran parity
      9. transjax parity-repair ./out/src/mod.py \\
             --fortran-file ./src/mod.F90 \\
             --golden-dir ./ftest/tests/golden       # repair failures automatically

    \b
    OR run the full end-to-end pipeline in one command:
     10. transjax pipeline /path/to/fortran \\
             --analysis-dir ./analysis -o ./pipeline  # steps 6-9 for every module
     11. transjax integrate ./jax/src \\
             --fortran-dir /path/to/fortran \\
             -o ./pipeline  # build & test Python system integration

    \b
    Quick reference:
      transjax analyze --help    # all analysis options
      transjax convert --help    # all conversion options (incl. --next, --yes)
      transjax status --help     # translation progress dashboard
      transjax ftest --help      # Fortran functional test framework
      transjax golden --help     # golden (trusted-run) data generation
      transjax show-config       # view/verify your config
    """


# ---------------------------------------------------------------------------
# transjax convert
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("fortran_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default="./jax_output", show_default=True,
              help="Directory where translated JAX files, tests, and reports are written.")
@click.option("--model", default=None, metavar="MODEL",
              help="Claude model to use for translation, e.g. claude-sonnet-4-6. "
                   "Defaults to the value in config.yaml.")
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY", metavar="KEY",
              help="Anthropic API key.  Reads $ANTHROPIC_API_KEY when omitted.  "
                   "Not needed if you authenticated with `claude login`.")
@click.option("--max-repair-iterations", default=5, show_default=True, metavar="N",
              help="Maximum number of times TransJAX will ask Claude to fix a module "
                   "whose tests fail.  Set to 0 to disable the repair loop entirely.")
@click.option("--skip-tests", is_flag=True,
              help="Do not generate or run pytest tests after translation.")
@click.option("--skip-repair", is_flag=True,
              help="Translate every module once and stop, even if tests fail.")
@click.option("--force", is_flag=True,
              help="Re-translate modules already marked successful in the state file.")
@click.option("--modules", default=None, metavar="MOD1,MOD2,…",
              help="Comma-separated list of Fortran module names to translate.  "
                   "When omitted, TransJAX translates all pending modules in "
                   "dependency order (lowest depth first).")
@click.option("--next", "translate_next", is_flag=True,
              help="Translate only the single next pending module suggested by the "
                   "state tracker.  Useful for one-module-at-a-time workflows.")
@click.option("--yes", "-y", is_flag=True,
              help="Skip the confirmation prompt when --next auto-selects a module.")
@click.option("--temperature", default=None, type=float, metavar="T",
              help="Sampling temperature for the LLM (0.0–1.0).")
@click.option("--analysis-dir", default=None, metavar="DIR",
              help="Path to an existing analysis directory produced by 'transjax analyze'.")
@click.option("--gcm-model", default=None, metavar="NAME",
              help="Name of the GCM/ESM being translated (e.g. CTSM, CESM, MOM6).")
@click.option("--mode", default="units", show_default=True,
              type=click.Choice(["units", "whole"], case_sensitive=False),
              help="Translation strategy.  'units' translates each subroutine separately "
                   "then assembles (better for large modules, more granular repair). "
                   "'whole' sends the full Fortran source in one call (faster, lower "
                   "overhead, better for small-to-medium modules).")
@click.option("--verbose", "-v", is_flag=True,
              help="Print DEBUG-level logs.")
def convert(
    fortran_directory: str,
    output: str,
    model: Optional[str],
    api_key: Optional[str],
    max_repair_iterations: int,
    skip_tests: bool,
    skip_repair: bool,
    force: bool,
    modules: Optional[str],
    translate_next: bool,
    yes: bool,
    temperature: Optional[float],
    analysis_dir: Optional[str],
    gcm_model: Optional[str],
    mode: str,
    verbose: bool,
):
    """Translate a Fortran codebase to JAX (full pipeline).

    FORTRAN_DIRECTORY is the root of the Fortran source tree to translate.

    TransJAX tracks translation progress in <output>/translation_state.json.
    On every run it skips already-successful modules and resumes from where it
    left off.  Use --force to re-translate successful modules.

    \b
    Automatic next-module suggestion (no --modules needed):
      transjax convert ./fortran              # translate all pending, lowest-dep first
      transjax convert ./fortran --next       # translate only the next pending module
      transjax convert ./fortran --next --yes # same, skip confirmation prompt

    \b
    Manual module selection:
      transjax convert ./fortran --modules CanopyFluxes,SoilWater

    \b
    Pipeline stages:
      1. Analyze   — parse Fortran, build dependency graph (reused if cached)
      2. State     — load/init translation_state.json, detect already-done modules
      3. Order     — pick pending modules in dependency order (lowest depth first)
      4. Translate — call Claude to convert each module to JAX
      5. Test      — generate and run pytest suites (skippable)
      6. Repair    — ask Claude to fix failing tests (skippable)

    \b
    Output structure:
      <output>/src/                     translated JAX modules
      <output>/tests/                   generated pytest files
      <output>/reports/                 per-module logs
      <output>/translation_state.json   persistent progress tracker

    \b
    See also:
      transjax status ./jax_output      view progress without translating
      transjax analyze ./fortran        run static analysis separately

    \b
    Examples:
      transjax convert ./fortran -o ./jax_output
      transjax convert ./fortran --next --yes
      transjax convert ./fortran --modules CanopyFluxes,SoilWater
      transjax convert ./fortran --skip-tests --force
      transjax convert ./fortran --model claude-opus-4-6 --temperature 0.2
      transjax convert ./fortran --mode whole          # single-call translation
    """
    from transjax.agents.orchestrator import OrchestratorAgent
    from transjax.agents.utils.translation_state import TranslationStateManager

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        auth_display = "[green]Claude subscription (claude login)[/green]"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        auth_display = "[green]API key[/green]"
    else:
        auth_display = "[yellow]none detected — run `claude login` or set ANTHROPIC_API_KEY[/yellow]"

    fortran_path = Path(fortran_directory).resolve()
    output_path = Path(output).resolve()

    # ------------------------------------------------------------------ #
    # --next  logic: peek at state, suggest a module, ask for confirmation #
    # ------------------------------------------------------------------ #
    module_list: Optional[list] = (
        [m.strip() for m in modules.split(",")] if modules else None
    )

    if translate_next and not module_list:
        state_file = output_path / "translation_state.json"
        if state_file.exists():
            import json as _json
            try:
                state_data = _json.loads(state_file.read_text())
                next_mod = None
                next_depth = None
                next_file = ""
                for entry in state_data.get("ordered_modules", []):
                    if entry.get("status") == "pending":
                        next_mod = entry["module"]
                        next_depth = entry["depth"]
                        next_file = entry.get("file", "")
                        break
            except Exception:
                next_mod = None

            if next_mod is None:
                console.print(
                    "[green]All modules already translated — nothing left to do.[/green]"
                )
                sys.exit(0)

            console.print(Panel.fit(
                f"[white]Next pending module:[/white] [bold cyan]{next_mod}[/bold cyan]\n"
                f"[white]Dependency depth:[/white]    {next_depth} "
                f"[dim](0 = no internal deps)[/dim]\n"
                f"[white]Source file:[/white]         {next_file}",
                title="[bold cyan]Suggested next module[/bold cyan]",
                border_style="cyan",
            ))

            if not yes:
                confirmed = click.confirm(
                    f"Translate '{next_mod}' now?", default=True
                )
                if not confirmed:
                    console.print("[yellow]Aborted.[/yellow]")
                    sys.exit(0)

            module_list = [next_mod]
        else:
            console.print(
                "[yellow]No translation_state.json found in output directory.\n"
                "Run without --next first to initialise the state, "
                "or run 'transjax analyze' first.[/yellow]"
            )
            # Fall through — let the orchestrator initialise state from scratch

    console.print(Panel.fit(
        f"[white]Fortran:[/white]           {fortran_path}\n"
        f"[white]Output:[/white]            {output_path}\n"
        f"[white]LLM model:[/white]         {model or 'from config'}\n"
        f"[white]GCM model:[/white]         {gcm_model or 'unspecified'}\n"
        f"[white]Auth:[/white]              {auth_display}\n"
        f"[white]Modules this run:[/white]  "
        f"{', '.join(module_list) if module_list else 'all pending (auto)'}\n"
        f"[white]Max repairs:[/white]       {max_repair_iterations}\n"
        f"[white]Skip tests:[/white]        {skip_tests}\n"
        f"[white]Force retranslate:[/white] {force}",
        title="[bold cyan]TransJAX — convert[/bold cyan]",
        border_style="cyan",
    ))

    try:
        orchestrator = OrchestratorAgent(
            fortran_dir=fortran_path,
            output_dir=output_path,
            model=model,
            temperature=temperature,
            max_repair_iterations=max_repair_iterations,
            skip_tests=skip_tests,
            skip_repair=skip_repair,
            force_retranslate=force,
            module_list=module_list,
            gcm_model_name=gcm_model,
            verbose=verbose,
            analysis_dir=Path(analysis_dir).resolve() if analysis_dir else None,
            translation_mode=mode,
        )
        results = orchestrator.run()

        state_summary = results.get("state_summary", {})
        pct = state_summary.get("percent_done", 0)
        total = state_summary.get("total", 0)
        done = state_summary.get("translated", 0)
        pending = state_summary.get("pending", 0)
        failed_count = state_summary.get("failed", 0)
        next_up = state_summary.get("next_suggested")

        progress_bar = _render_progress_bar(done, total)

        done_lines = (
            f"[white]Translated this run:[/white]  {results.get('translated_count', 0)}\n"
            f"[white]Tests passed:[/white]         {results.get('tests_passed', 0)}\n"
            f"[white]Failures this run:[/white]    {results.get('final_failures', 0)}\n"
        )
        overall_lines = (
            f"\n[white]Overall progress:[/white]    {progress_bar} "
            f"{done}/{total} ({pct}%)\n"
            f"[white]Pending:[/white]             {pending}\n"
            f"[white]Failed (cumulative):[/white]  {failed_count}\n"
        )
        next_line = (
            f"[white]Next suggested:[/white]      [cyan]{next_up}[/cyan]"
            if next_up else
            "[white]Status:[/white]              [green]All modules translated![/green]"
        )
        state_line = f"\n[white]State file:[/white]          {results.get('state_file', '')}"

        console.print(Panel.fit(
            done_lines + overall_lines + next_line + state_line,
            title="[bold green]Done[/bold green]",
            border_style="green",
        ))

        if next_up:
            console.print(
                f"\n[dim]Run [bold]transjax convert {fortran_directory} --next[/bold] "
                f"to translate '[cyan]{next_up}[/cyan]' next.[/dim]"
            )

        if results.get("final_failures", 0) > 0:
            console.print(
                f"\n[yellow]{results['final_failures']} module(s) failed — "
                "see output/reports/ for details.[/yellow]"
            )
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]Error: {exc}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _render_progress_bar(done: int, total: int, width: int = 20) -> str:
    """Return a plain-text progress bar string."""
    if total == 0:
        return "[" + " " * width + "]"
    filled = int(width * done / total)
    return "[green]" + "█" * filled + "[/green]" + "░" * (width - filled)


# ---------------------------------------------------------------------------
# transjax analyze
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("fortran_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default=None, metavar="DIR",
              help="Directory for analysis reports.  "
                   "Defaults to <cwd>/transjax_analysis.")
@click.option("--template", "-t", default="auto", show_default=True,
              metavar="TEMPLATE",
              help="Project type template (see command help for choices).")
@click.option("--no-graphs", is_flag=True,
              help="Skip generation of GraphML / JSON dependency graph files.  "
                   "Speeds up analysis on very large codebases.")
@click.option("--verbose", "-v", is_flag=True,
              help="Print DEBUG-level logs, including per-file parse details.")
def analyze(
    fortran_directory: str,
    output: Optional[str],
    template: str,
    no_graphs: bool,
    verbose: bool,
):
    """Analyse a Fortran codebase without translating it.

    FORTRAN_DIRECTORY is the root of the Fortran source tree to inspect.

    Parses all Fortran files, builds a module dependency graph, estimates
    translation complexity, and writes results to the output directory.
    Run this before `transjax convert` to understand the project structure
    and spot potential issues (circular deps, large modules, etc.).

    \b
    Output files (written to --output, default <cwd>/transjax_analysis):
      analysis_results.json   full structured results (modules, deps, stats)
      analysis_summary.txt    human-readable report
      translation_units.json  per-function breakdown with complexity scores
      graphs/                 dependency graph in GraphML and JSON formats

    \b
    Key metrics shown after analysis:
      Files              total Fortran source files found
      Lines              total lines of code
      Modules            Fortran modules (each becomes one JAX file)
      Translation units  individual functions/subroutines to translate
      Dependencies       inter-module USE relationships
      Circular deps      dependency cycles (must resolve before translating)

    \b
    --template choices:
      auto                 auto-detect from directory structure (default)
      generic              search the entire project root recursively
      scientific_computing HPC / MPI projects
      numerical_library    standalone Fortran libraries
      climate_model        projects with src/physics + src/dynamics layout
      ctsm                 Community Terrestrial Systems Model

    \b
    Examples:
      transjax analyze ./fortran
      transjax analyze ./fortran -o ./my_analysis
      transjax analyze ./fortran --template generic
      transjax analyze ./fortran --template scientific_computing --no-graphs
      transjax analyze ./fortran --verbose
    """
    import logging

    from transjax.analyzer.analyzer import create_analyzer_for_project

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    fortran_path = Path(fortran_directory).resolve()
    output_dir = output or str(Path.cwd() / "transjax_analysis")

    console.print(f"[cyan]Analysing:[/cyan] {fortran_path}")
    console.print(f"[cyan]Template:[/cyan]  {template}")

    try:
        analyzer = create_analyzer_for_project(
            str(fortran_path),
            template=template,
            output_dir=output_dir,
            generate_graphs=not no_graphs,
        )
        analyzer.analyze(save_results=True)
        stats = analyzer.get_summary_statistics()
        file_order = analyzer.get_file_translation_order()

        max_depth = max((e["depth"] for e in file_order), default=0)
        n_circular = sum(1 for e in file_order if e.get("circular_dep_involved"))
        order_file = Path(output_dir) / "translation_order.json"

        console.print(Panel.fit(
            f"[white]Files:[/white]              {stats['files']}\n"
            f"[white]Lines:[/white]              {stats['lines']:,}\n"
            f"[white]Modules:[/white]            {stats['modules']}\n"
            f"[white]Translation units:[/white]  {stats['translation_units']}\n"
            f"[white]Dependencies:[/white]       {stats['dependencies']}\n"
            f"[white]Circular deps:[/white]      {stats['circular_dependencies']}\n"
            f"[white]Translation depth:[/white]  0 – {max_depth}\n"
            f"[white]Circular dep files:[/white] {n_circular}\n"
            f"[white]Results saved to:[/white]   {output_dir}\n"
            f"[white]Translation order:[/white]  {order_file}",
            title="[bold green]Analysis complete[/bold green]",
            border_style="green",
        ))

        if file_order:
            # Print a rich table: one row per depth group, then all files
            console.print()
            console.print("[bold cyan]File Translation Order[/bold cyan] "
                          "[dim](depth 0 = translate first)[/dim]")

            # Depth-group summary
            depth_table = Table(show_header=True, header_style="bold", box=None,
                                pad_edge=False)
            depth_table.add_column("Depth", style="cyan", width=7)
            depth_table.add_column("Files", width=7)
            depth_table.add_column("Meaning", style="dim")
            for d in range(max_depth + 1):
                group = [e for e in file_order if e["depth"] == d]
                if d == 0:
                    meaning = "no internal dependencies — translate first"
                elif d == max_depth:
                    meaning = "most transitive dependencies — translate last"
                else:
                    meaning = f"depends on depth-{d-1} files"
                depth_table.add_row(str(d), str(len(group)), meaning)
            console.print(depth_table)

            # Per-file table (cap at 50 to avoid flooding the terminal)
            console.print()
            file_table = Table(show_header=True, header_style="bold", box=None,
                               pad_edge=False)
            file_table.add_column("Rank", width=5, style="dim")
            file_table.add_column("Dep", width=4, style="cyan",
                                  header="Dep", no_wrap=True)
            file_table.add_column("Deps→", width=5, header="Deps→")
            file_table.add_column("←Used", width=5, header="←Used")
            file_table.add_column("Subs", width=5)
            file_table.add_column("Lines", width=6)
            file_table.add_column("⚠", width=2, header="⚠")
            file_table.add_column("File")

            display = file_order if len(file_order) <= 50 else file_order[:50]
            for e in display:
                warn = "[red]●[/red]" if e["circular_dep_involved"] else ""
                # Make path relative to fortran_path for readability
                try:
                    rel = str(Path(e["file"]).relative_to(fortran_path))
                except ValueError:
                    rel = e["file"]
                file_table.add_row(
                    str(e["rank"]),
                    str(e["depth"]),
                    str(e["n_internal_deps"]),
                    str(e["n_dependents"]),
                    str(e["n_subroutines"]),
                    str(e["line_count"]),
                    warn,
                    rel,
                )

            console.print(file_table)

            if len(file_order) > 50:
                console.print(
                    f"[dim]… {len(file_order) - 50} more files — "
                    f"see {order_file}[/dim]"
                )

    except Exception as exc:
        console.print(f"[red]Analysis failed: {exc}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


# ---------------------------------------------------------------------------
# transjax ftest
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("fortran_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default="./ftest_output", show_default=True,
              help="Directory where drivers/, tests/, Makefile, and report are written.")
@click.option("--build-dir", default=None, metavar="DIR",
              help="Existing model build directory containing compiled .o files that "
                   "test drivers will link against.  When omitted, drivers are compiled "
                   "stand-alone (works for subroutines with no external dependencies).")
@click.option("--compiler", default="nvfortran", show_default=True, metavar="CMD",
              help="Fortran compiler command used in the generated Makefile.")
@click.option("--netcdf-inc", default=None, metavar="DIR",
              help="NetCDF include directory (-I flag in the generated Makefile).")
@click.option("--netcdf-lib", default=None, metavar="DIR",
              help="NetCDF library directory (-L flag in the generated Makefile).")
@click.option("--modules", default=None, metavar="MOD1,MOD2,…",
              help="Comma-separated Fortran module names to include.  "
                   "When omitted, all modules in FORTRAN_DIRECTORY are processed.")
@click.option("--model", default=None, metavar="MODEL",
              help="Claude model to use.  Defaults to the value in config.yaml.")
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY", metavar="KEY",
              help="Anthropic API key.  Reads $ANTHROPIC_API_KEY when omitted.")
@click.option("--verbose", "-v", is_flag=True,
              help="Print DEBUG-level logs, including full prompts and API responses.")
def ftest(
    fortran_directory: str,
    output: str,
    build_dir: Optional[str],
    compiler: str,
    netcdf_inc: Optional[str],
    netcdf_lib: Optional[str],
    modules: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    verbose: bool,
):
    """Build a functional testing framework for a Fortran codebase (Ftest).

    FORTRAN_DIRECTORY is the root of the Fortran source tree to instrument.

    For each subroutine found, TransJAX generates:

    \b
      1. A thin Fortran "test driver" (drivers/test_<name>.f90)
         Reads inputs from a Fortran NAMELIST on stdin.
         Calls the subroutine.
         Writes outputs as KEY=VALUE lines to stdout.

      2. A Python pytest file (tests/test_<name>.py)
         Compiles and invokes the driver as a subprocess.
         Feeds multiple input sets.
         Asserts on outputs with physical-range checks.

      3. A Makefile that compiles all drivers against the existing model build.

      4. A conftest.py with the run_driver() helper shared by all tests.

    \b
    Typical HPC workflow:
      transjax ftest ./src -o ./ftest_output \\
          --build-dir ./model_build \\
          --compiler nvfortran \\
          --netcdf-inc /opt/netcdf/include \\
          --netcdf-lib /opt/netcdf/lib
      cd ftest_output && make all
      python -m pytest tests/ -v

    \b
    Quick local run (no model build required):
      transjax ftest ./src -o ./ftest_output --compiler gfortran

    \b
    Filter to specific modules:
      transjax ftest ./src --modules CanopyFluxes,SoilWater -o ./ftest_output

    \b
    Output structure written to --output:
      <output>/drivers/           Fortran test-driver source files
      <output>/drivers/bin/       Compiled driver binaries (after `make all`)
      <output>/tests/conftest.py  run_driver() pytest helper
      <output>/tests/             Python pytest files (one per subroutine)
      <output>/Makefile           Compiles drivers against the model build
      <output>/ftest_report.json  Summary of generated tests
    """
    from transjax.agents.ftest_agent import FtestAgent

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        auth_display = "[green]Claude subscription (claude login)[/green]"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        auth_display = "[green]API key[/green]"
    else:
        auth_display = "[yellow]none detected — run `claude login` or set ANTHROPIC_API_KEY[/yellow]"

    fortran_path = Path(fortran_directory).resolve()
    output_path = Path(output).resolve()
    build_path = Path(build_dir).resolve() if build_dir else None
    module_filter = [m.strip() for m in modules.split(",")] if modules else None

    console.print(Panel.fit(
        f"[white]Fortran:[/white]     {fortran_path}\n"
        f"[white]Output:[/white]      {output_path}\n"
        f"[white]Build dir:[/white]   {build_path or 'none (stand-alone)'}\n"
        f"[white]Compiler:[/white]    {compiler}\n"
        f"[white]NetCDF inc:[/white]  {netcdf_inc or 'not set'}\n"
        f"[white]NetCDF lib:[/white]  {netcdf_lib or 'not set'}\n"
        f"[white]Modules:[/white]     {modules or 'all'}\n"
        f"[white]LLM model:[/white]   {model or 'from config'}\n"
        f"[white]Auth:[/white]        {auth_display}",
        title="[bold cyan]TransJAX — ftest[/bold cyan]",
        border_style="cyan",
    ))

    try:
        agent = FtestAgent(model=model)
        result = agent.run(
            fortran_dir=fortran_path,
            output_dir=output_path,
            build_dir=build_path,
            compiler=compiler,
            netcdf_inc=netcdf_inc,
            netcdf_lib=netcdf_lib,
            module_filter=module_filter,
            verbose=verbose,
        )

        status_color = "green" if result.errors == 0 else "yellow"
        console.print(Panel.fit(
            f"[white]Subroutines found:[/white]   {result.subroutines_found}\n"
            f"[white]Tests generated:[/white]     {result.tests_generated}\n"
            f"[white]Skipped:[/white]             {result.skipped}\n"
            f"[white]Errors:[/white]              {result.errors}\n"
            f"[white]Output:[/white]              {output_path}\n"
            f"\n"
            f"[white]Next steps:[/white]\n"
            f"  cd {output_path} && make all\n"
            f"  python -m pytest tests/ -v",
            title=f"[bold {status_color}]Ftest complete[/bold {status_color}]",
            border_style=status_color,
        ))

        if result.errors > 0:
            console.print(
                f"\n[yellow]{result.errors} subroutine(s) failed — "
                "check ftest_report.json for details.[/yellow]"
            )
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]Error: {exc}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


# ---------------------------------------------------------------------------
# transjax status
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("output_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--show-all", is_flag=True,
              help="Show all modules including already-translated ones.  "
                   "By default only pending and failed modules are listed.")
def status(output_directory: str, show_all: bool):
    """Show translation progress for an output directory.

    OUTPUT_DIRECTORY is the --output directory used with 'transjax convert'.
    It must contain a translation_state.json file.

    \b
    Examples:
      transjax status ./jax_output
      transjax status ./jax_output --show-all
    """
    import json as _json

    output_path = Path(output_directory).resolve()
    state_file = output_path / "translation_state.json"

    if not state_file.exists():
        console.print(
            f"[yellow]No translation_state.json found in {output_path}.[/yellow]\n"
            "Run 'transjax convert' to initialise the state, "
            "or 'transjax analyze' first to build the dependency order."
        )
        sys.exit(1)

    try:
        state_data = _json.loads(state_file.read_text())
    except Exception as exc:
        console.print(f"[red]Could not read state file: {exc}[/red]")
        sys.exit(1)

    ordered = state_data.get("ordered_modules", [])
    if not ordered:
        console.print("[yellow]State file exists but contains no modules.[/yellow]")
        sys.exit(0)

    # Summary counts
    by_status = {"pending": 0, "success": 0, "failed": 0}
    for e in ordered:
        s = e.get("status", "pending")
        by_status[s] = by_status.get(s, 0) + 1

    total = len(ordered)
    done = by_status["success"]
    pct = round(100 * done / total, 1) if total else 0.0
    progress_bar = _render_progress_bar(done, total)

    # Next suggested
    next_mod = next(
        (e["module"] for e in ordered if e.get("status") == "pending"), None
    )
    next_entry = next((e for e in ordered if e["module"] == next_mod), None) if next_mod else None

    console.print(Panel.fit(
        f"[white]Total modules:[/white]    {total}\n"
        f"[white]Progress:[/white]         {progress_bar} {done}/{total} ({pct}%)\n"
        f"[white]Translated:[/white]       [green]{done}[/green]\n"
        f"[white]Failed:[/white]           [red]{by_status['failed']}[/red]\n"
        f"[white]Pending:[/white]          {by_status['pending']}\n"
        + (
            f"[white]Next suggested:[/white]   [bold cyan]{next_mod}[/bold cyan]  "
            f"[dim]depth {next_entry['depth']}  {next_entry.get('file', '')}[/dim]"
            if next_mod else
            "[white]Status:[/white]           [bold green]All modules translated![/bold green]"
        ),
        title=f"[bold cyan]Translation Status — {output_path.name}[/bold cyan]",
        border_style="cyan",
    ))

    # Per-module table
    console.print()
    tbl = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    tbl.add_column("Rank", width=5, style="dim")
    tbl.add_column("Dep", width=4, style="cyan", header="Dep")
    tbl.add_column("Status", width=10)
    tbl.add_column("Tests", width=6)
    tbl.add_column("Repairs", width=8)
    tbl.add_column("Module")
    tbl.add_column("File", style="dim")

    _STATUS_STYLE = {
        "success": "[green]success[/green]",
        "failed":  "[red]failed[/red]",
        "pending": "[dim]pending[/dim]",
    }

    for entry in ordered:
        s = entry.get("status", "pending")
        if not show_all and s == "success":
            continue  # hide done modules unless --show-all

        # Shorten file path
        fp = entry.get("file", "")
        try:
            fp = str(Path(fp).relative_to(output_path.parent))
        except ValueError:
            pass

        tbl.add_row(
            str(entry.get("rank", "")),
            str(entry.get("depth", "")),
            _STATUS_STYLE.get(s, s),
            "[green]✓[/green]" if entry.get("tests_passed") else
            ("[red]✗[/red]" if s != "pending" else ""),
            str(entry.get("repair_attempts", 0)) if entry.get("repair_attempts") else "",
            entry["module"],
            fp,
        )

    console.print(tbl)

    if not show_all and by_status["success"] > 0:
        console.print(
            f"[dim]{by_status['success']} translated module(s) hidden — "
            "use --show-all to display them.[/dim]"
        )

    if next_mod:
        console.print(
            f"\n[dim]Next: [bold]transjax convert <fortran_dir> "
            f"--next[/bold] → will translate '[cyan]{next_mod}[/cyan]'[/dim]"
        )


# ---------------------------------------------------------------------------
# transjax golden
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("ftest_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--n-cases", default=5, show_default=True, metavar="N",
              help="Number of golden input/output cases to capture per subroutine.")
@click.option("--gcm-model", default="generic ESM", show_default=True, metavar="NAME",
              help="Name of the ESM being tested (e.g. CTSM, MOM6, CESM).  "
                   "Injected into prompts so Claude applies model-specific physical knowledge.")
@click.option("--model", default=None, metavar="MODEL",
              help="Claude model to use.  Defaults to the value in config.yaml.")
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY", metavar="KEY",
              help="Anthropic API key.  Reads $ANTHROPIC_API_KEY when omitted.")
@click.option("--verbose", "-v", is_flag=True,
              help="Print DEBUG-level logs.")
def golden(
    ftest_directory: str,
    n_cases: int,
    gcm_model: str,
    model: Optional[str],
    api_key: Optional[str],
    verbose: bool,
):
    """Generate golden (trusted-run) data for all Ftest subroutine drivers.

    FTEST_DIRECTORY is the output directory produced by `transjax ftest`.
    It must contain ftest_report.json and compiled binaries in drivers/bin/.

    For each compiled test executable, the agent:

    \b
      1. Reads the subroutine interface from ftest_report.json.
      2. Asks Claude (as an ESM domain expert) to propose --n-cases physically
         representative input scenarios spanning typical, extreme, and edge-case
         regimes appropriate for the subroutine's physical process.
      3. Runs each scenario through the compiled Fortran driver binary and
         captures the real KEY=VALUE output from the trusted build.
      4. Writes a JSON golden file to tests/golden/<module>_<subroutine>.json.

    \b
    Golden files can then be loaded by pytest tests to perform regression
    checks against future builds:

    \b
      import json
      GOLDEN = json.loads(Path("tests/golden/CanopyFluxesMod_CanopyFluxes.json").read_text())
      for case in GOLDEN["cases"]:
          out = run_driver(DRIVER, case["inputs"])
          for key, expected in case["outputs"].items():
              assert abs(out[key] - expected) < 1e-4 * abs(expected) + 1e-10

    \b
    Prerequisites:
      transjax ftest ./src -o ./ftest_output   # generate drivers
      cd ftest_output && make all              # compile binaries
      transjax golden ./ftest_output           # capture golden data

    \b
    Examples:
      transjax golden ./ftest_output
      transjax golden ./ftest_output --n-cases 8 --gcm-model CTSM
      transjax golden ./ftest_output --gcm-model MOM6 --model claude-opus-4-6

    \b
    Output (written inside FTEST_DIRECTORY):
      tests/golden/<module>_<subroutine>.json   one file per subroutine
    """
    from transjax.agents.golden_agent import GoldenAgent

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        auth_display = "[green]Claude subscription (claude login)[/green]"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        auth_display = "[green]API key[/green]"
    else:
        auth_display = "[yellow]none detected — run `claude login` or set ANTHROPIC_API_KEY[/yellow]"

    ftest_path = Path(ftest_directory).resolve()

    console.print(Panel.fit(
        f"[white]Ftest directory:[/white]  {ftest_path}\n"
        f"[white]Cases per sub:[/white]   {n_cases}\n"
        f"[white]ESM context:[/white]     {gcm_model}\n"
        f"[white]LLM model:[/white]       {model or 'from config'}\n"
        f"[white]Auth:[/white]            {auth_display}",
        title="[bold cyan]TransJAX — golden[/bold cyan]",
        border_style="cyan",
    ))

    try:
        agent = GoldenAgent(model=model)
        result = agent.run(
            ftest_output_dir=ftest_path,
            n_cases=n_cases,
            gcm_model_name=gcm_model,
            verbose=verbose,
        )

        status_color = "green" if result.errors == 0 else "yellow"
        golden_dir = ftest_path / "tests" / "golden"
        console.print(Panel.fit(
            f"[white]Subroutines processed:[/white]  {result.subroutines_attempted}\n"
            f"[white]Golden files written:[/white]   {result.golden_written}\n"
            f"[white]Skipped (no binary):[/white]    {result.skipped}\n"
            f"[white]Errors:[/white]                 {result.errors}\n"
            f"[white]Output:[/white]                 {golden_dir}",
            title=f"[bold {status_color}]Golden data complete[/bold {status_color}]",
            border_style=status_color,
        ))

        if result.errors > 0:
            console.print(
                f"\n[yellow]{result.errors} subroutine(s) failed — "
                "check logs for details.[/yellow]"
            )
            sys.exit(1)

    except FileNotFoundError as exc:
        console.print(f"\n[red]Error: {exc}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]Error: {exc}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


# ---------------------------------------------------------------------------
# transjax test-parity
# ---------------------------------------------------------------------------

@cli.command("test-parity")
@click.argument("python_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--golden-dir", required=True, metavar="DIR",
              help="Directory containing golden JSON files produced by 'transjax golden'.")
@click.option("--ftest-report", default=None, metavar="FILE",
              help="Path to ftest_report.json from 'transjax ftest'.  "
                   "RECOMMENDED: when provided, parity tests are generated programmatically "
                   "from the exact Fortran interface — no LLM inference, guaranteed dtype and "
                   "argument-order accuracy.  When omitted, Claude infers the interface from "
                   "the JAX source (slower, less reliable).")
@click.option("--output", "-o", default="./parity_tests", show_default=True, metavar="DIR",
              help="Directory where generated pytest files and the parity report are written.")
@click.option("--rtol", default=1e-10, show_default=True, type=float, metavar="F",
              help="Relative tolerance passed to jnp.allclose for output comparisons.")
@click.option("--atol", default=1e-12, show_default=True, type=float, metavar="F",
              help="Absolute tolerance passed to jnp.allclose for output comparisons.")
@click.option("--model", default=None, metavar="MODEL",
              help="Claude model to use (fallback path only). Defaults to config.yaml.")
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY", metavar="KEY",
              help="Anthropic API key. Reads $ANTHROPIC_API_KEY when omitted.")
@click.option("--verbose", "-v", is_flag=True,
              help="Print DEBUG logs and full pytest output on failures.")
def test_parity(
    python_file: str,
    golden_dir: str,
    ftest_report: Optional[str],
    output: str,
    rtol: float,
    atol: float,
    model: Optional[str],
    api_key: Optional[str],
    verbose: bool,
):
    """Generate and run numerical parity tests for a JAX-translated module.

    PYTHON_FILE is the translated JAX/Python module to test
    (e.g. jax_output/src/physics_mod.py).

    For each golden JSON file in GOLDEN_DIR that belongs to PYTHON_FILE's
    module, the agent:

    \b
    Step 1 — Generate a pytest file.
      With --ftest-report (RECOMMENDED):
        Built programmatically from the Ftest interface definition — exact
        argument names, Fortran types, and array dimensions.  No LLM call.
      Without --ftest-report (fallback):
        Claude infers the interface from the JAX source code.  Slower and
        less reliable; may mis-identify dtypes or argument order.

    \b
    Step 2 — Run the generated pytest and report per-case pass/fail.

    \b
    Golden files are matched to the module by name prefix:
      physics_mod_compute_flux.json   → matched to physics_mod.py

    \b
    Output structure:
      <output>/test_parity_<module>_<subroutine>.py   generated pytest
      <output>/parity_report.json                     aggregated results

    \b
    Examples:
      # Recommended — with ftest interface (programmatic, no LLM)
      transjax test-parity ./jax_output/src/physics_mod.py \\
          --golden-dir ./ftest/tests/golden \\
          --ftest-report ./ftest/ftest_report.json

      # Fallback — without ftest report
      transjax test-parity ./jax_output/src/CanopyFluxesMod.py \\
          --golden-dir ./ftest/tests/golden \\
          --rtol 1e-8 --atol 1e-10 \\
          --output ./parity_results
    """
    import logging

    from transjax.agents.parity_agent import ParityAgent

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    python_path      = Path(python_file).resolve()
    golden_path      = Path(golden_dir).resolve()
    output_path      = Path(output).resolve()
    ftest_report_path = Path(ftest_report).resolve() if ftest_report else None

    try:
        agent = ParityAgent(model=model)
        result = agent.run(
            python_file=python_path,
            golden_dir=golden_path,
            output_dir=output_path,
            ftest_report_path=ftest_report_path,
            rtol=rtol,
            atol=atol,
            verbose=verbose,
        )

        # Exit with non-zero if any subroutine failed
        if result.subroutines_failed > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]Error: {exc}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


# ---------------------------------------------------------------------------
# transjax parity-repair
# ---------------------------------------------------------------------------

@cli.command("parity-repair")
@click.argument("python_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--fortran-file", required=True, metavar="FILE",
              help="Original Fortran source file (read-only reference for Claude).")
@click.option("--golden-dir", required=True, metavar="DIR",
              help="Directory containing golden JSON files produced by 'transjax golden'.")
@click.option("--ftest-report", default=None, metavar="FILE",
              help="Path to ftest_report.json from 'transjax ftest'. "
                   "RECOMMENDED: enables programmatic parity-test generation from the "
                   "exact Fortran interface — no LLM inference for test construction.")
@click.option("--output", "-o", default="./parity_repair", show_default=True, metavar="DIR",
              help="Directory where generated test files, the parity report, and the "
                   "repair-summary markdown are written.")
@click.option("--rtol", default=1e-10, show_default=True, type=float, metavar="F",
              help="Relative tolerance passed to jnp.allclose for output comparisons.")
@click.option("--atol", default=1e-12, show_default=True, type=float, metavar="F",
              help="Absolute tolerance passed to jnp.allclose for output comparisons.")
@click.option("--max-iterations", default=5, show_default=True, type=int, metavar="N",
              help="Maximum number of repair iterations before giving up.")
@click.option("--model", default=None, metavar="MODEL",
              help="Claude model to use. Defaults to config.yaml.")
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY", metavar="KEY",
              help="Anthropic API key. Reads $ANTHROPIC_API_KEY when omitted.")
@click.option("--verbose", "-v", is_flag=True,
              help="Print DEBUG logs and full pytest output.")
def parity_repair(
    python_file: str,
    fortran_file: str,
    golden_dir: str,
    ftest_report: Optional[str],
    output: str,
    rtol: float,
    atol: float,
    max_iterations: int,
    model: Optional[str],
    api_key: Optional[str],
    verbose: bool,
):
    """Run parity tests and iteratively repair the JAX module until they pass.

    PYTHON_FILE is the translated JAX/Python module to test and (if needed)
    repair (e.g. jax_output/src/physics_mod.py).

    \b
    Workflow:
      1. Run parity tests against golden Fortran reference data.
      2. PASS → write a green-light markdown report and exit 0.
      3. FAIL → enter a repair loop:
           a. Ask Claude to diagnose the numerical discrepancy.
           b. Apply the fix to PYTHON_FILE only.
           c. Re-run parity tests.
           d. Repeat up to --max-iterations times.
      4. Write a repair-summary markdown (PASS or FAIL) to <output>/docs/.

    \b
    Hard constraints enforced by the repair agent:
      • Only PYTHON_FILE may be modified — never golden data, test files,
        or the Fortran source.
      • Function signatures and NamedTuple return types are preserved.
      • JAX JIT compatibility and float64 precision are maintained.

    \b
    Output structure:
      <output>/test_parity_<module>_<sub>.py          generated pytest
      <output>/parity_report.json                     parity test results
      <output>/docs/<module>_numerical_parity_repair_PASS.md
      <output>/docs/<module>_numerical_parity_repair_FAIL.md

    \b
    Examples:
      # With ftest interface (recommended)
      transjax parity-repair ./jax_output/src/physics_mod.py \\
          --fortran-file ./src/physics_mod.F90 \\
          --golden-dir ./ftest/tests/golden \\
          --ftest-report ./ftest/ftest_report.json

      # Without ftest report (LLM fallback for test generation)
      transjax parity-repair ./jax_output/src/CanopyFluxesMod.py \\
          --fortran-file ./src/CanopyFluxesMod.F90 \\
          --golden-dir ./ftest/tests/golden \\
          --max-iterations 3 --rtol 1e-8 --atol 1e-10
    """
    import logging

    from transjax.agents.parity_repair_agent import ParityRepairAgent

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    python_path       = Path(python_file).resolve()
    fortran_path      = Path(fortran_file).resolve()
    golden_path       = Path(golden_dir).resolve()
    output_path       = Path(output).resolve()
    ftest_report_path = Path(ftest_report).resolve() if ftest_report else None

    if not fortran_path.exists():
        console.print(f"[red]Fortran file not found: {fortran_path}[/red]")
        sys.exit(1)

    try:
        agent = ParityRepairAgent(model=model)
        result = agent.run(
            python_file=python_path,
            fortran_file=fortran_path,
            golden_dir=golden_path,
            output_dir=output_path,
            ftest_report_path=ftest_report_path,
            rtol=rtol,
            atol=atol,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        if result.final_status == "FAIL":
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]Error: {exc}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


# ---------------------------------------------------------------------------
# transjax show-config
# ---------------------------------------------------------------------------

@cli.command("show-config")
@click.argument("config_file", type=click.Path(exists=True), required=False)
def show_config(config_file: Optional[str]):
    """Print the active configuration (YAML).

    CONFIG_FILE: optional path to a custom config.yaml.  When omitted the
    package-bundled defaults are shown.
    """
    import yaml

    from transjax.agents.utils.config_loader import load_config

    config = load_config(Path(config_file) if config_file else None)
    console.print("\n[bold cyan]TransJAX configuration[/bold cyan]\n")
    console.print(yaml.dump(config, default_flow_style=False, sort_keys=False))


# ---------------------------------------------------------------------------
# transjax init
# ---------------------------------------------------------------------------

@cli.command()
def init():
    """Create a .env.template file and show authentication options."""
    env_template = (
        "# ── Authentication ────────────────────────────────────────────────────────\n"
        "#\n"
        "# Option 1 (recommended for Claude Pro/Max subscribers):\n"
        "#   Run `claude login` once — no further configuration needed.\n"
        "#   TransJAX will automatically pick up CLAUDE_CODE_OAUTH_TOKEN.\n"
        "#\n"
        "# Option 2 (pay-per-use API key):\n"
        "#   Get your key at https://console.anthropic.com/ and paste it below.\n"
        "#\n"
        "# ANTHROPIC_API_KEY=your_api_key_here\n"
    )
    target = Path.cwd() / ".env.template"
    if target.exists():
        console.print("[yellow].env.template already exists — skipping.[/yellow]")
    else:
        target.write_text(env_template)
        console.print("[green]Created .env.template[/green]")

    console.print("\n[bold]Authentication options:[/bold]")
    console.print()
    console.print("  [cyan]Option 1 — Claude subscription (Pro/Max):[/cyan]")
    console.print("    claude login")
    console.print("    transjax convert /path/to/fortran -o ./jax_output")
    console.print()
    console.print("  [cyan]Option 2 — API key:[/cyan]")
    console.print("    cp .env.template .env   # then fill in ANTHROPIC_API_KEY")
    console.print("    transjax convert /path/to/fortran -o ./jax_output")


# ---------------------------------------------------------------------------
# transjax pipeline
# ---------------------------------------------------------------------------

@cli.command("pipeline")
@click.argument("fortran_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--analysis-dir", required=True, metavar="DIR",
              help="Directory produced by 'transjax analyze'. Must contain "
                   "translation_order.json.")
@click.option("--output", "-o", default="./pipeline_output", show_default=True, metavar="DIR",
              help="Root output directory. Receives ftest/, jax/, parity/ sub-trees, "
                   "translation_state.json, and pipeline_state.json.")
@click.option("--gcm-model-name", default="generic ESM", show_default=True, metavar="NAME",
              help="Earth system model name (e.g. CTSM, MOM6). Injected into prompts "
                   "so Claude can apply model-specific physical domain knowledge.")
@click.option("--mode", default="units", show_default=True,
              type=click.Choice(["units", "whole"]),
              help="Translation mode. 'units' translates subroutine-by-subroutine then "
                   "assembles; 'whole' sends the full Fortran module in one LLM call.")
@click.option("--ftest-build-dir", default=None, metavar="DIR",
              help="Fortran model build directory containing .o / .mod files that "
                   "Ftest drivers link against (required for Ftest compilation).")
@click.option("--ftest-compiler", default="nvfortran", show_default=True, metavar="CMD",
              help="Fortran compiler command used to build test drivers.")
@click.option("--ftest-netcdf-inc", default=None, metavar="DIR",
              help="NetCDF include path (passed as -I to the Fortran compiler).")
@click.option("--ftest-netcdf-lib", default=None, metavar="DIR",
              help="NetCDF library path (passed as -L to the linker).")
@click.option("--golden-cases", default=5, show_default=True, type=int, metavar="N",
              help="Number of golden I/O cases to generate per subroutine.")
@click.option("--rtol", default=1e-10, show_default=True, type=float, metavar="F",
              help="Relative tolerance for numerical parity comparisons (jnp.allclose).")
@click.option("--atol", default=1e-12, show_default=True, type=float, metavar="F",
              help="Absolute tolerance for numerical parity comparisons (jnp.allclose).")
@click.option("--max-repair-iterations", default=5, show_default=True, type=int, metavar="N",
              help="Maximum repair iterations per module if parity tests fail.")
@click.option("--module", "module_filter", multiple=True, metavar="NAME",
              help="Restrict pipeline to these module(s). Repeatable: --module A --module B.")
@click.option("--skip-ftest", is_flag=True,
              help="Skip FTest step entirely (use existing ftest_report.json if present).")
@click.option("--skip-golden", is_flag=True,
              help="Skip golden I/O generation (use existing golden JSON files if present).")
@click.option("--force", is_flag=True,
              help="Re-run all pipeline steps for every module, even if already done.")
@click.option("--force-ftest", is_flag=True,
              help="Re-run FTest step even if ftest_report.json already exists.")
@click.option("--force-golden", is_flag=True,
              help="Re-run golden I/O generation even if golden JSON files already exist.")
@click.option("--force-translate", is_flag=True,
              help="Re-translate even if the JAX module file already exists.")
@click.option("--force-parity", is_flag=True,
              help="Re-run parity tests even if they previously passed.")
@click.option("--model", default=None, metavar="MODEL",
              help="Claude model to use. Defaults to config.yaml.")
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY", metavar="KEY",
              help="Anthropic API key. Reads $ANTHROPIC_API_KEY when omitted.")
@click.option("--integrate", "run_integrate", is_flag=True,
              help="After all modules complete, run IntegratorAgent to build and test "
                   "the Python system integration (model_run.py + test_integration.py).")
@click.option("--max-integration-repair-iterations", default=5, show_default=True,
              type=int, metavar="N",
              help="Maximum IntegrationRepairAgent iterations if the initial "
                   "integration test fails (only relevant with --integrate).")
@click.option("--verbose", "-v", is_flag=True,
              help="Print DEBUG logs and full subprocess/pytest output.")
def pipeline(
    fortran_dir: str,
    analysis_dir: str,
    output: str,
    gcm_model_name: str,
    mode: str,
    ftest_build_dir: Optional[str],
    ftest_compiler: str,
    ftest_netcdf_inc: Optional[str],
    ftest_netcdf_lib: Optional[str],
    golden_cases: int,
    rtol: float,
    atol: float,
    max_repair_iterations: int,
    module_filter: tuple,
    skip_ftest: bool,
    skip_golden: bool,
    force: bool,
    force_ftest: bool,
    force_golden: bool,
    force_translate: bool,
    force_parity: bool,
    run_integrate: bool,
    max_integration_repair_iterations: int,
    model: Optional[str],
    api_key: Optional[str],
    verbose: bool,
):
    """Run the full end-to-end pipeline for every module in translation order.

    FORTRAN_DIR is the root of the Fortran source tree (same directory you
    passed to 'transjax analyze').

    \b
    Per-module steps (in dependency order from translation_order.json):
      [1/4] FTest     — generate Fortran functional-test drivers & compile
      [2/4] Golden    — run drivers to capture golden I/O reference data
      [3/4] Translate — translate Fortran module to JAX/Python with Claude
      [4/4] Parity    — run numerical parity tests; auto-repair if they fail

    \b
    State is persisted after each module so the pipeline can be safely
    interrupted and resumed:
      <output>/translation_state.json   module translation status
      <output>/pipeline_state.json      per-module per-step detail

    \b
    Steps are skipped automatically if their output files already exist.
    Use --force-* flags to re-run specific steps, or --force to re-run all.

    \b
    Output layout:
      <output>/ftest/<module>/              FTest drivers & golden JSON
      <output>/jax/src/<module>.py          Translated JAX module
      <output>/parity/<module>/             Parity test files & report
      <output>/parity/<module>/docs/        Repair summary markdown

    \b
    Prerequisites:
      1. Run 'transjax analyze FORTRAN_DIR -o ANALYSIS_DIR' first.
      2. For FTest compilation you need nvfortran (or another compiler)
         and the model build directory (--ftest-build-dir).
      3. Golden I/O capture requires the compiled driver binaries.
         If compilation fails (e.g. on a laptop), use --skip-ftest
         --skip-golden to run translate + parity only.

    \b
    Examples:
      # Full pipeline (requires compiled Fortran drivers)
      transjax pipeline ./src/clm \\
          --analysis-dir ./analysis \\
          --output ./pipeline_out \\
          --gcm-model-name CTSM \\
          --ftest-build-dir ./build

      # Translate + parity only (skip Fortran compilation)
      transjax pipeline ./src/clm \\
          --analysis-dir ./analysis \\
          --output ./pipeline_out \\
          --skip-ftest --skip-golden

      # Single module, force re-run of all steps
      transjax pipeline ./src/clm \\
          --analysis-dir ./analysis \\
          --output ./pipeline_out \\
          --module CanopyFluxesMod \\
          --force
    """
    import logging

    from transjax.agents.pipeline_runner import PipelineRunner

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    try:
        runner = PipelineRunner(
            fortran_dir=Path(fortran_dir).resolve(),
            analysis_dir=Path(analysis_dir).resolve(),
            output_dir=Path(output).resolve(),
            gcm_model_name=gcm_model_name,
            model=model,
            translation_mode=mode,
            ftest_build_dir=Path(ftest_build_dir).resolve() if ftest_build_dir else None,
            ftest_compiler=ftest_compiler,
            ftest_netcdf_inc=ftest_netcdf_inc,
            ftest_netcdf_lib=ftest_netcdf_lib,
            golden_n_cases=golden_cases,
            parity_rtol=rtol,
            parity_atol=atol,
            max_repair_iterations=max_repair_iterations,
            module_filter=list(module_filter) if module_filter else None,
            force=force,
            force_ftest=force_ftest,
            force_golden=force_golden,
            force_translate=force_translate,
            force_parity=force_parity,
            skip_ftest=skip_ftest,
            skip_golden=skip_golden,
            run_integrate=run_integrate,
            max_integration_repair_iterations=max_integration_repair_iterations,
            verbose=verbose,
        )
        summary = runner.run()

        if summary.get("failed", 0) > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted. State has been saved — resume with the same command.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]Pipeline error: {exc}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


# ---------------------------------------------------------------------------
# transjax integrate
# ---------------------------------------------------------------------------

@cli.command("integrate")
@click.argument("jax_src_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--fortran-dir", required=True, metavar="DIR",
              type=click.Path(exists=True, file_okay=False),
              help="Root of the Fortran source tree (read-only; used to understand "
                   "the model's call structure and produce accurate documentation).")
@click.option("--output", "-o", default="./pipeline_output", show_default=True, metavar="DIR",
              help="Root output directory.  Integration files are written to "
                   "<output>/integration/.")
@click.option("--gcm-model-name", default="generic ESM", show_default=True, metavar="NAME",
              help="Earth system model name (e.g. CTSM, MOM6).  Injected into prompts "
                   "and the generated README.")
@click.option("--max-repair-iterations", default=5, show_default=True, type=int, metavar="N",
              help="Maximum IntegrationRepairAgent iterations if the initial test fails.")
@click.option("--model", default=None, metavar="MODEL",
              help="Claude model to use. Defaults to config.yaml.")
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY", metavar="KEY",
              help="Anthropic API key.  Reads $ANTHROPIC_API_KEY when omitted.")
@click.option("--verbose", "-v", is_flag=True,
              help="Print DEBUG logs and full pytest output.")
def integrate(
    jax_src_dir: str,
    fortran_dir: str,
    output: str,
    gcm_model_name: str,
    max_repair_iterations: int,
    model: Optional[str],
    api_key: Optional[str],
    verbose: bool,
):
    """Build and test the Python system integration for a translated ESM.

    JAX_SRC_DIR is the directory of translated JAX/Python modules (e.g.
    <pipeline_output>/jax/src/).

    \b
    The agent:
      1. Reads the Fortran source structure to understand the model's call
         sequence (init → timestep loop → finalize).
      2. Reads public function signatures from all translated JAX modules.
      3. Asks Claude to generate:
           model_run.py              — Python integration driver
           test_integration.py       — pytest wrapper with assertions
           System_integration_README.md
      4. Runs test_integration.py via pytest.
      5. If tests fail, invokes IntegrationRepairAgent iteratively until
         the tests pass or the repair limit is reached.

    \b
    Output layout:
      <output>/integration/model_run.py
      <output>/integration/test_integration.py
      <output>/integration/docs/System_integration_README.md
      <output>/integration/docs/integration_repair_PASS|FAIL.md  (if repair needed)

    \b
    Examples:
      # After a full pipeline run
      transjax integrate ./pipeline_out/jax/src \\
          --fortran-dir ./src/clm \\
          --output ./pipeline_out \\
          --gcm-model-name CTSM

      # Standalone (jax modules live in ./translated/)
      transjax integrate ./translated \\
          --fortran-dir ./fortran_src \\
          --output ./integration_out
    """
    import logging

    from transjax.agents.integrator_agent import IntegratorAgent

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    try:
        agent = IntegratorAgent(model=model)
        result = agent.run(
            fortran_dir=Path(fortran_dir).resolve(),
            jax_src_dir=Path(jax_src_dir).resolve(),
            output_dir=Path(output).resolve(),
            gcm_model_name=gcm_model_name,
            max_repair_iterations=max_repair_iterations,
            verbose=verbose,
        )

        if result.passed:
            console.print(
                f"\n[bold green]✓ Integration PASSED[/bold green]\n"
                f"  model_run.py → {result.model_run_path}\n"
                f"  test file   → {result.test_path}\n"
                f"  README      → {result.readme_path}"
            )
        else:
            console.print(
                f"\n[bold red]✗ Integration {result.final_status}[/bold red]"
            )
            if result.error:
                console.print(f"  Error: {result.error}")
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]Integration error: {exc}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Package entry point."""
    cli()


if __name__ == "__main__":
    main()
