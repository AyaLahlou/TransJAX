#!/usr/bin/env python3
"""
Cleaned Test Generator for CLM-ML-JAX.
Optimized for: jax-agents/translated_modules/ directory structure.
"""

import argparse
import sys
import traceback
from pathlib import Path

# --- DIRECTORY SETUP ---
# Script is at: ~/clm-ml-jax-1/jax-agents/examples/generate_tests.py
script_dir = Path(__file__).resolve().parent
# Project root is: ~/clm-ml-jax-1/jax-agents/
PROJECT_ROOT = script_dir.parent 

# Add the 'src' directory so 'jax_agents' package is importable
# Path: ~/clm-ml-jax-1/jax-agents/src/
src_path = PROJECT_ROOT / "src"
sys.path.insert(0, str(src_path))

try:
    from jax_agents import TestAgent
    from rich.console import Console
except ImportError:
    print(f"Error: Could not find 'jax_agents' in {src_path}")
    print("Please run: pip install rich")
    sys.exit(1)

console = Console()

def get_source_dir(python_file: Path) -> str:
    """Matches the file path to a CLM source category."""
    valid_dirs = {'clm_src_main', 'clm_src_biogeophys', 'clm_src_utils', 'clm_src_cpl'}
    for part in python_file.parts:
        if part in valid_dirs:
            return part
    return "clm_src_main"

def generate_tests_for_module(module_name: str, num_cases: int, manual_python: str = None):
    """Core logic to generate tests using the TestAgent."""
    
    # 1. Resolve Python file path
    if manual_python:
        python_file = Path(manual_python).resolve()
    else:
        # Matches your ls: jax-agents/translated_modules/clm_varctl/clm_varctl.py
        python_file = PROJECT_ROOT / "translated_modules" / module_name / f"{module_name}.py"

    if not python_file.exists():
        console.print(f"[red]Error: File not found at {python_file}[/red]")
        translated_dir = PROJECT_ROOT / "translated_modules"
        if translated_dir.exists():
            console.print(f"[yellow]Available folders in {translated_dir}:[/yellow]")
            for d in translated_dir.iterdir():
                if d.is_dir(): console.print(f"  • {d.name}")
        return

    # 2. Setup output directory (the 'tests' folder inside the module folder)
    output_dir = python_file.parent / "tests"
    output_dir.mkdir(exist_ok=True)

    console.print(f"\n[bold cyan]🚀 Generating {num_cases} tests for {module_name}[/bold cyan]")
    
    test_agent = TestAgent()
    source_dir = get_source_dir(python_file)

    try:
        test_agent.generate_tests(
            module_name=module_name,
            python_code=python_file.read_text(),
            num_test_cases=num_cases,
            include_edge_cases=True,
            source_directory=source_dir,
            output_dir=output_dir,
        )
        console.print(f"\n[bold green]✓ Success! Files saved to: {output_dir}[/bold green]")
        
        cost = test_agent.get_cost_estimate()
        console.print(f"[dim]Estimated Cost: ${cost['total_cost_usd']:.4f}[/dim]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Cleaned JAX Test Generator")
    parser.add_argument("module_name", nargs="?", help="Name of the module (e.g., clm_varctl)")
    parser.add_argument("--num-cases", type=int, default=10, help="Number of test cases")
    parser.add_argument("--python", help="Manual path to a specific Python file")

    args = parser.parse_args()

    if not args.module_name and not args.python:
        parser.print_help()
        return

    generate_tests_for_module(args.module_name, args.num_cases, args.python)

if __name__ == "__main__":
    main()