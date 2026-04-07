"""Entry point for ``python -m transjax.analyzer`` and ``python analyzer.py``."""
import argparse
import logging
from pathlib import Path

from transjax.analyzer.analyzer import create_analyzer_for_project


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m transjax.analyzer",
        description="Analyse a Fortran codebase and produce analysis_results.json, "
                    "DESIGN.md, and translation_order.md.",
    )
    parser.add_argument("fortran_dir", help="Root of the Fortran source tree")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (default: <fortran_dir>/transjax_analysis)")
    parser.add_argument("-t", "--template", default="auto",
                        help="Project type template (default: auto)")
    parser.add_argument("--no-graphs", action="store_true",
                        help="Skip GraphML/JSON dependency graph generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = args.output or str(Path(args.fortran_dir) / "transjax_analysis")
    analyzer = create_analyzer_for_project(
        args.fortran_dir,
        template=args.template,
        output_dir=output_dir,
        generate_graphs=not args.no_graphs,
    )
    analyzer.analyze()
    print(f"\nDone. Results written to: {output_dir}")


if __name__ == "__main__":
    main()
