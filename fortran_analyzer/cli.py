"""
Command-line interface for the Fortran Analyzer.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from analyzer import FortranAnalyzer, create_analyzer_for_project
from config.project_config import ConfigurationManager, create_default_config


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_config_command(args) -> None:
    """Create a configuration file."""
    manager = ConfigurationManager()

    if args.template == "list":
        print("Available templates:")
        for template in manager.list_templates():
            info = manager.get_template_info(template)
            print(f"  {template}: {info.get('project_name', 'Generic template')}")
        return

    if args.template == "auto":
        detected = manager.auto_detect_project_type(args.project_root)
        print(f"Auto-detected project type: {detected}")
        template = detected
    else:
        template = args.template

    config = create_default_config(args.project_root, template)

    # Apply command line overrides
    if args.project_name:
        config.project_name = args.project_name
    if args.output_dir:
        config.output_dir = args.output_dir

    # Save configuration
    config_file = (
        Path(args.output)
        if args.output
        else Path(args.project_root) / "fortran_analyzer_config.yaml"
    )
    config.to_yaml(config_file)

    print(f"Configuration saved to: {config_file}")
    print(f"Project type: {template}")
    print(f"Source directories: {config.source_dirs}")


def analyze_command(args) -> None:
    """Run analysis command."""
    if args.config:
        # Load from configuration file
        from .config.project_config import load_config

        config = load_config(args.config)
        analyzer = FortranAnalyzer(config)
    else:
        # Create from command line arguments
        config_overrides = {}

        if args.output_dir:
            config_overrides["output_dir"] = args.output_dir
        if args.project_name:
            config_overrides["project_name"] = args.project_name
        if args.max_unit_lines:
            config_overrides["max_translation_unit_lines"] = args.max_unit_lines
        if not args.graphs:
            config_overrides["generate_graphs"] = False

        analyzer = create_analyzer_for_project(
            args.project_root, args.template, **config_overrides
        )

    # Run analysis
    print(f"Analyzing Fortran project: {analyzer.config.project_name}")
    print(f"Project root: {analyzer.config.project_root}")

    try:
        results = analyzer.analyze(save_results=True)

        # Print summary
        summary = analyzer.get_summary_statistics()
        print("\nAnalysis Complete!")
        print(f"  Files analyzed: {summary['files']}")
        print(f"  Total lines: {summary['lines']:,}")
        print(f"  Modules found: {summary['modules']}")
        print(f"  Translation units: {summary['translation_units']}")
        print(f"  Dependencies: {summary['dependencies']}")

        if summary["circular_dependencies"] > 0:
            print(f"  ⚠️  Circular dependencies: {summary['circular_dependencies']}")

        if summary["external_dependencies"] > 0:
            print(f"  External dependencies: {summary['external_dependencies']}")

        print(f"\nResults saved to: {analyzer.output_dir}")

        if analyzer.config.generate_graphs:
            viz_files = results.get("visualizations", {})
            if viz_files:
                print(f"Visualizations generated: {len(viz_files)}")

    except Exception as e:
        print(f"Analysis failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def info_command(args) -> None:
    """Show project information."""
    manager = ConfigurationManager()

    if args.detect:
        project_type = manager.auto_detect_project_type(args.project_root)
        print(f"Detected project type: {project_type}")

        template_info = manager.get_template_info(project_type)
        print(f"Recommended settings:")
        for key, value in template_info.items():
            print(f"  {key}: {value}")

    if args.list_files:
        config = create_default_config(args.project_root, "generic")
        from .parser.fortran_parser import FortranParser

        parser = FortranParser(config)

        files = parser.find_fortran_files()
        print(f"\nFound {len(files)} Fortran files:")
        for file_path in files:
            relative_path = file_path.relative_to(config.project_root)
            print(f"  {relative_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fortran Codebase Analyzer - Analyze and prepare Fortran code for translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Create configuration file
  fortran-analyzer config /path/to/project --template scientific_computing
  
  # Quick analysis with auto-detection
  fortran-analyzer analyze /path/to/project --template auto
  
  # Show project information
  fortran-analyzer info /path/to/project --detect --list-files
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Config command
    config_parser = subparsers.add_parser("config", help="Create configuration file")
    config_parser.add_argument("project_root", help="Path to Fortran project root")
    config_parser.add_argument(
        "--template",
        "-t",
        default="auto",
        help='Configuration template (use "list" to see all)',
    )
    config_parser.add_argument("--output", "-o", help="Output configuration file path")
    config_parser.add_argument("--project-name", help="Project name override")
    config_parser.add_argument("--output-dir", help="Output directory override")
    config_parser.set_defaults(func=create_config_command)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze Fortran project")
    analyze_parser.add_argument(
        "project_root", nargs="?", help="Path to Fortran project root"
    )
    analyze_parser.add_argument("--config", "-c", help="Configuration file path")
    analyze_parser.add_argument(
        "--template", "-t", default="auto", help="Configuration template"
    )
    analyze_parser.add_argument("--output-dir", "-o", help="Output directory")
    analyze_parser.add_argument("--project-name", help="Project name")
    analyze_parser.add_argument(
        "--max-unit-lines", type=int, help="Maximum lines per translation unit"
    )
    analyze_parser.add_argument(
        "--no-graphs",
        dest="graphs",
        action="store_false",
        help="Disable graph generation",
    )
    analyze_parser.set_defaults(func=analyze_command)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show project information")
    info_parser.add_argument("project_root", help="Path to Fortran project root")
    info_parser.add_argument(
        "--detect", action="store_true", help="Auto-detect project type"
    )
    info_parser.add_argument(
        "--list-files", action="store_true", help="List Fortran files in project"
    )
    info_parser.set_defaults(func=info_command)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    setup_logging(args.verbose)

    # Validate arguments
    if (
        args.command in ["analyze", "info"]
        and not args.project_root
        and not getattr(args, "config", None)
    ):
        print("Error: project_root is required when not using --config")
        sys.exit(1)

    if args.command == "analyze" and not args.project_root and not args.config:
        print("Error: either project_root or --config must be specified")
        sys.exit(1)

    # Run command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
