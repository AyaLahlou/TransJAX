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

    \b
    Common workflow:
      transjax analyze /path/to/fortran        # inspect the codebase first
      transjax convert /path/to/fortran -o ./out  # translate + test + repair
    """


# ---------------------------------------------------------------------------
# transjax convert
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("fortran_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default="./jax_output", show_default=True,
              help="Output directory.")
@click.option("--model", default=None, help="Claude model (default: from config).")
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY",
              help="Anthropic API key (default: $ANTHROPIC_API_KEY).")
@click.option("--max-repair-iterations", default=5, show_default=True,
              help="Max repair attempts per module.")
@click.option("--skip-tests", is_flag=True, help="Skip test generation.")
@click.option("--skip-repair", is_flag=True, help="Skip the repair loop.")
@click.option("--force", is_flag=True, help="Re-translate already-translated files.")
@click.option("--modules", default=None,
              help="Comma-separated module filter (default: all).")
@click.option("--temperature", default=None, type=float,
              help="LLM temperature (default: 0.0).")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging.")
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
    verbose: bool,
):
    """Translate a Fortran codebase to JAX (full pipeline).

    Runs static analysis, translates each module with Claude, generates pytest
    tests, and iteratively repairs failures — all in one command.

    \b
    Examples:
      transjax convert ./fortran -o ./jax_output
      transjax convert ./fortran --modules clm_varctl,SoilStateType
      transjax convert ./fortran --skip-tests --force
    """
    from transjax.agents.orchestrator import OrchestratorAgent

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    fortran_path = Path(fortran_directory).resolve()
    output_path = Path(output).resolve()
    module_list = [m.strip() for m in modules.split(",")] if modules else None

    console.print(Panel.fit(
        f"[white]Fortran:[/white]        {fortran_path}\n"
        f"[white]Output:[/white]         {output_path}\n"
        f"[white]Model:[/white]          {model or 'from config'}\n"
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
@click.option("--output", "-o", default=None,
              help="Output directory for reports (default: <fortran_dir>/transjax_analysis).")
@click.option("--template", "-t", default="auto", show_default=True,
              help="Project template: auto, ctsm, scientific_computing, generic, …")
@click.option("--no-graphs", is_flag=True, help="Skip graph visualisation.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging.")
def analyze(
    fortran_directory: str,
    output: Optional[str],
    template: str,
    no_graphs: bool,
    verbose: bool,
):
    """Analyse a Fortran codebase without translating it.

    Parses source files, builds dependency graphs, and writes a JSON report and
    summary to the output directory.  Useful for inspecting a project before
    running `transjax convert`.

    \b
    Examples:
      transjax analyze ./fortran
      transjax analyze ./fortran -o ./analysis --template scientific_computing
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
    """Create a .env.template file in the current directory."""
    env_template = (
        "# Anthropic API key — get yours at https://console.anthropic.com/\n"
        "ANTHROPIC_API_KEY=your_api_key_here\n"
    )
    target = Path.cwd() / ".env.template"
    if target.exists():
        console.print("[yellow].env.template already exists — skipping.[/yellow]")
    else:
        target.write_text(env_template)
        console.print("[green]Created .env.template[/green]")

    console.print("\nNext steps:")
    console.print("  cp .env.template .env   # add your real key")
    console.print("  transjax convert /path/to/fortran -o ./jax_output")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Package entry point."""
    cli()


if __name__ == "__main__":
    main()
