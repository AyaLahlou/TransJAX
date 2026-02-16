#!/usr/bin/env python3
"""
Example usage of the Repair Agent.
Demonstrates iterative fixing of JAX code using failure analysis.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure jax_agents is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from jax_agents.repair_agent import RepairAgent
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    print("Error: Missing dependencies. Ensure 'rich' is installed and 'jax_agents' is in path.")
    sys.exit(1)

console = Console()

def display_summary(result, cost_info):
    """Helper to display formatted results in the console."""
    table = Table(title="Repair Process Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    
    table.add_row("Module", result.module_name)
    table.add_row("Iterations", str(result.iterations))
    table.add_row("Success", "[green]YES[/]" if result.all_tests_passed else "[red]NO[/]")
    table.add_row("Total Cost", f"${cost_info['total_cost_usd']:.4f}")
    
    console.print(table)
    
    # Show snippet of Root Cause Analysis
    console.print(Panel(result.root_cause_analysis[:500] + "...", title="Root Cause Analysis Summary", border_style="cyan"))

def repair_failed_translation(
    module_name: str,
    fortran_code: str,
    failed_python_code: str,
    test_report: str,
    test_file_path: Optional[Path] = None,
    output_dir: Path = Path("repair_outputs"),
    max_iterations: int = 5,
):
    """Repair a failed translation iteratively."""
    repair_agent = RepairAgent(max_repair_iterations=max_iterations)
    
    console.print(f"[bold yellow]Starting repair for:[/] [bold cyan]{module_name}[/]")
    console.print("-" * 40)
    
    result = repair_agent.repair_translation(
        module_name=module_name,
        fortran_code=fortran_code,
        failed_python_code=failed_python_code,
        test_report=test_report,
        test_file_path=test_file_path,
        output_dir=output_dir,
    )
    
    display_summary(result, repair_agent.get_cost_estimate())
    return result

def run_example_repair():
    """Runs the built-in SoilTemperatureMod bug example."""
    module_name = "SoilTemperatureMod"
    
    fortran_code = """
subroutine calculate_temperature(temp_in, temp_out, n)
    do i = 1, n
        temp_out(i) = temp_in(i) + 273.15
    end do
end subroutine
"""
    
    failed_python_code = """
import jax.numpy as jnp
def calculate_temperature(temp_in):
    n = temp_in.shape[0]
    temp_out = jnp.zeros(n)
    for i in range(1, n):  # BUG: Starts from 1 instead of 0
        temp_out = temp_out.at[i].set(temp_in[i] + 273.15)
    return temp_out
"""
    
    test_report = "AssertionError: Expected [273.15, ...], Got [0.0, ...]"
    
    return repair_failed_translation(
        module_name=module_name,
        fortran_code=fortran_code,
        failed_python_code=failed_python_code,
        test_report=test_report
    )

def main():
    parser = argparse.ArgumentParser(description="Repair failed JAX translations automatically")
    parser.add_argument("--example", action="store_true", help="Run with built-in example")
    parser.add_argument("--module", help="Module name")
    parser.add_argument("--fortran", help="Path to Fortran file")
    parser.add_argument("--python", help="Path to failed Python file")
    parser.add_argument("--test-report", help="Path to test report txt")
    parser.add_argument("--max-iterations", type=int, default=5)
    # Added back the -o flag
    parser.add_argument("-o", "--output", type=str, default="repair_outputs", help="Output directory")
    
    args = parser.parse_args()

    try:
        if args.example:
            run_example_repair()
        elif all([args.module, args.fortran, args.python, args.test_report]):
            repair_failed_translation(
                module_name=args.module,
                fortran_code=Path(args.fortran).read_text(),
                failed_python_code=Path(args.python).read_text(),
                test_report=Path(args.test_report).read_text(),
                output_dir=Path(args.output),
                max_iterations=args.max_iterations
            )
        else:
            parser.print_help()
    except Exception as e:
        console.print(f"[bold red]Critical Error:[/] {e}")

if __name__ == "__main__":
    main()
