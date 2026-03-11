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
      1. transjax init                           # set up authentication
      2. transjax analyze /path/to/fortran       # inspect the project first
      3. transjax convert /path/to/fortran -o ./out  # translate + test + repair

    \b
    Quick reference:
      transjax analyze --help    # all analysis options
      transjax convert --help    # all conversion options
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
              help="Do not generate or run pytest tests after translation.  "
                   "Useful for a quick first pass or when the test infrastructure "
                   "is not yet set up.")
@click.option("--skip-repair", is_flag=True,
              help="Translate every module once and stop, even if tests fail.  "
                   "Equivalent to --max-repair-iterations 0 but leaves test "
                   "infrastructure intact.")
@click.option("--force", is_flag=True,
              help="Re-translate modules that were already translated in a previous "
                   "run.  Without this flag, existing output files are skipped.")
@click.option("--modules", default=None, metavar="MOD1,MOD2,…",
              help="Comma-separated list of Fortran module names to translate.  "
                   "All other modules are skipped.  "
                   "Example: --modules CanopyFluxes,SoilWater")
@click.option("--temperature", default=None, type=float, metavar="T",
              help="Sampling temperature for the LLM (0.0–1.0).  "
                   "Lower values produce more deterministic output.  "
                   "Default: 0.0 (fully deterministic).")
@click.option("--analysis-dir", default=None, metavar="DIR",
              help="Path to an existing analysis directory produced by 'transjax analyze'. "
                   "When provided, skips re-running analysis. "
                   "By default TransJAX looks for <fortran_dir>/transjax_analysis/ "
                   "automatically before running a fresh analysis.")
@click.option("--verbose", "-v", is_flag=True,
              help="Print DEBUG-level logs, including full prompts and API responses.")
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
    temperature: Optional[float],
    analysis_dir: Optional[str],
    verbose: bool,
):
    """Translate a Fortran codebase to JAX (full pipeline).

    FORTRAN_DIRECTORY is the root of the Fortran source tree to translate.

    TransJAX runs four stages automatically:

    \b
      1. Analyze   — parse all Fortran files, build dependency graph
      2. Translate — call Claude to convert each module to JAX
      3. Test      — generate and run pytest suites (skippable)
      4. Repair    — ask Claude to fix failing tests (skippable)

    Output structure written to --output:

    \b
      <output>/src/          translated JAX modules
      <output>/tests/        generated pytest files
      <output>/reports/      per-module status and repair logs

    \b
    Authentication (one of):
      claude login                        # Claude Pro/Max subscription
      export ANTHROPIC_API_KEY=sk-ant-…  # pay-per-use API key

    \b
    Examples:
      transjax convert ./fortran -o ./jax_output
      transjax convert ./fortran --modules CanopyFluxes,SoilWater
      transjax convert ./fortran --skip-tests
      transjax convert ./fortran --force
      transjax convert ./fortran --model claude-opus-4-6 --temperature 0.2
      transjax convert ./fortran --skip-repair
      transjax convert ./fortran --max-repair-iterations 10 --verbose
    """
    from transjax.agents.orchestrator import OrchestratorAgent

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    # Show which authentication method will be used.
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        auth_display = "[green]Claude subscription (claude login)[/green]"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        auth_display = "[green]API key[/green]"
    else:
        auth_display = "[yellow]none detected — run `claude login` or set ANTHROPIC_API_KEY[/yellow]"

    fortran_path = Path(fortran_directory).resolve()
    output_path = Path(output).resolve()
    module_list = [m.strip() for m in modules.split(",")] if modules else None

    console.print(Panel.fit(
        f"[white]Fortran:[/white]        {fortran_path}\n"
        f"[white]Output:[/white]         {output_path}\n"
        f"[white]Model:[/white]          {model or 'from config'}\n"
        f"[white]Auth:[/white]           {auth_display}\n"
        f"[white]Modules:[/white]        {modules or 'all'}\n"
        f"[white]Max repairs:[/white]    {max_repair_iterations}\n"
        f"[white]Skip tests:[/white]     {skip_tests}\n"
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
            verbose=verbose,
            analysis_dir=Path(analysis_dir).resolve() if analysis_dir else None,
        )
        results = orchestrator.run()

        console.print(Panel.fit(
            f"[white]Translated:[/white]  {results['translated_count']}\n"
            f"[white]Tests passed:[/white] {results['tests_passed']}\n"
            f"[white]Failures:[/white]    {results['final_failures']}\n"
            f"[white]Output:[/white]      {output_path}",
            title="[bold green]Done[/bold green]",
            border_style="green",
        ))

        if results["final_failures"] > 0:
            console.print(f"\n[yellow]{results['final_failures']} module(s) failed — "
                          "see output/reports/ for details.[/yellow]")
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
# transjax analyze
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("fortran_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default=None, metavar="DIR",
              help="Directory for analysis reports.  "
                   "Defaults to <fortran_dir>/transjax_analysis.")
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
    Output files (written to --output, default <fortran_dir>/transjax_analysis):
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
    output_dir = output or str(fortran_path / "transjax_analysis")

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

        console.print(Panel.fit(
            f"[white]Files:[/white]              {stats['files']}\n"
            f"[white]Lines:[/white]              {stats['lines']:,}\n"
            f"[white]Modules:[/white]            {stats['modules']}\n"
            f"[white]Translation units:[/white]  {stats['translation_units']}\n"
            f"[white]Dependencies:[/white]       {stats['dependencies']}\n"
            f"[white]Circular deps:[/white]      {stats['circular_dependencies']}\n"
            f"[white]Results saved to:[/white]   {output_dir}",
            title="[bold green]Analysis complete[/bold green]",
            border_style="green",
        ))

    except Exception as exc:
        console.print(f"[red]Analysis failed: {exc}[/red]")
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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Package entry point."""
    cli()


if __name__ == "__main__":
    main()
