"""
Command-line interface for Fortran-to-JAX translation.

Usage:
    transjax convert /path/to/fortran --output /path/to/output
    transjax convert /path/to/fortran -o ./output --model claude-opus-4-5
    transjax convert /path/to/fortran --skip-tests --force
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

from transjax.agents.orchestrator import OrchestratorAgent

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="transjax")
def cli():
    """TransJAX: Automatic translation of Fortran code to JAX."""
    pass


@cli.command()
@click.argument('fortran_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    '--output', '-o',
    type=click.Path(),
    default='./jax_output',
    help='Output directory for translated code (default: ./jax_output)'
)
@click.option(
    '--model',
    type=str,
    default=None,
    help='Claude model to use (default: from config or claude-sonnet-4-5)'
)
@click.option(
    '--api-key',
    type=str,
    default=None,
    envvar='ANTHROPIC_API_KEY',
    help='Anthropic API key (default: from environment or .env file)'
)
@click.option(
    '--max-repair-iterations',
    type=int,
    default=5,
    help='Maximum repair attempts per module (default: 5)'
)
@click.option(
    '--skip-tests',
    is_flag=True,
    help='Skip test generation and execution'
)
@click.option(
    '--skip-repair',
    is_flag=True,
    help='Skip automatic repair on test failures'
)
@click.option(
    '--force',
    is_flag=True,
    help='Re-translate existing files (ignore incremental state)'
)
@click.option(
    '--modules',
    type=str,
    default=None,
    help='Comma-separated list of modules to translate (default: all modules)'
)
@click.option(
    '--temperature',
    type=float,
    default=None,
    help='LLM temperature (default: 0.0 for deterministic output)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
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
    """
    Convert Fortran code to JAX automatically.

    FORTRAN_DIRECTORY: Path to directory containing Fortran source files

    Example:
        transjax convert ./my_fortran_code -o ./jax_output
        transjax convert ./fortran --modules clm_varctl,SoilStateType
    """
    fortran_path = Path(fortran_directory).resolve()
    output_path = Path(output).resolve()

    # Validate inputs
    if not fortran_path.exists():
        console.print(f"[red]Error: Fortran directory not found: {fortran_path}[/red]")
        sys.exit(1)

    # Parse module list
    module_list = None
    if modules:
        module_list = [m.strip() for m in modules.split(',')]

    # Display configuration
    console.print(Panel.fit(
        f"[bold cyan]Fortran-to-JAX Translation Pipeline[/bold cyan]\n\n"
        f"[white]Fortran directory:[/white] {fortran_path}\n"
        f"[white]Output directory:[/white]  {output_path}\n"
        f"[white]Model:[/white]             {model or 'from config'}\n"
        f"[white]Max repair iter:[/white]   {max_repair_iterations}\n"
        f"[white]Skip tests:[/white]        {skip_tests}\n"
        f"[white]Skip repair:[/white]       {skip_repair}\n"
        f"[white]Force retranslate:[/white] {force}\n"
        f"[white]Modules:[/white]           {modules or 'all'}",
        title="Configuration",
        border_style="cyan"
    ))

    # Set API key if provided via CLI
    if api_key:
        import os
        os.environ['ANTHROPIC_API_KEY'] = api_key

    try:
        # Create orchestrator
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

        # Run the pipeline
        results = orchestrator.run()

        # Display summary
        console.print("\n")
        console.print(Panel.fit(
            f"[bold green]Translation Complete[/bold green]\n\n"
            f"[white]Modules translated:[/white] {results['translated_count']}\n"
            f"[white]Tests generated:[/white]    {results['tests_generated']}\n"
            f"[white]Tests passed:[/white]       {results['tests_passed']}\n"
            f"[white]Repairs needed:[/white]     {results['repairs_needed']}\n"
            f"[white]Final failures:[/white]     {results['final_failures']}\n\n"
            f"[white]Output saved to:[/white] {output_path}",
            title="Summary",
            border_style="green"
        ))

        # Exit code based on results
        if results['final_failures'] > 0:
            console.print(
                f"\n[yellow]Warning: {results['final_failures']} module(s) failed after max repair attempts.[/yellow]"
            )
            console.print("[yellow]Check repair_logs/ for detailed failure reports.[/yellow]")
            sys.exit(1)
        else:
            console.print("\n[bold green]All modules translated successfully![/bold green]")
            sys.exit(0)

    except KeyboardInterrupt:
        console.print("\n[yellow]Translation interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error during translation: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True), required=False)
def show_config(config_file: Optional[str]):
    """
    Display current configuration.

    CONFIG_FILE: Optional path to config.yaml (default: package default)
    """
    import yaml

    from transjax.agents.utils.config_loader import load_config

    if config_file:
        config = load_config(Path(config_file))
    else:
        config = load_config()
    console.print("\n[bold cyan]Current Configuration:[/bold cyan]\n")
    console.print(yaml.dump(config, default_flow_style=False, sort_keys=False))


@cli.command()
def init():
    """
    Initialize a new project with default configuration files.

    Creates:
        - .env.template (API key template)
    """
    cwd = Path.cwd()

    env_template = """# Anthropic API Key
# Get your key from: https://console.anthropic.com/
ANTHROPIC_API_KEY=your_api_key_here
"""

    files = {
        '.env.template': env_template,
    }

    for filename, content in files.items():
        filepath = cwd / filename
        if filepath.exists():
            console.print(f"[yellow]{filename} already exists, skipping[/yellow]")
        else:
            filepath.write_text(content)
            console.print(f"[green]Created {filename}[/green]")

    console.print("\n[bold cyan]Project initialized![/bold cyan]")
    console.print("\nNext steps:")
    console.print("1. Copy .env.template to .env and add your API key")
    console.print("2. Run: transjax convert /path/to/fortran")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
