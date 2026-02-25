"""
Main Fortran Analyzer framework.
Orchestrates the parsing, analysis, and visualization of Fortran codebases.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json
import time

from .config.project_config import FortranProjectConfig, ConfigurationManager
from .parser.fortran_parser import FortranParser
from .analysis.call_graph_builder import CallGraphBuilder
from .analysis.translation_decomposer import TranslationUnitDecomposer
from .visualization.visualizer import FortranVisualizer

logger = logging.getLogger(__name__)


class FortranAnalyzer:
    """
    Main analyzer class that orchestrates Fortran codebase analysis.

    This class provides a complete framework for analyzing Fortran projects, including:
    - Parsing Fortran source files to extract modules, subroutines, functions, and types
    - Building dependency graphs to understand module relationships
    - Decomposing code into translation units for easier porting
    - Generating visualizations and reports
    - Providing translation recommendations

    Example:
        Basic usage with auto-detection:

        >>> from fortran_analyzer import create_analyzer_for_project
        >>> analyzer = create_analyzer_for_project('/path/to/fortran/project', template='auto')
        >>> results = analyzer.analyze()
        >>> print(f"Found {len(results['parsing']['modules'])} modules")

        Custom configuration:

        >>> from fortran_analyzer.config.project_config import FortranProjectConfig
        >>> config = FortranProjectConfig(
        ...     project_name="My Project",
        ...     project_root="/path/to/project",
        ...     source_dirs=["src", "lib"],
        ...     max_translation_unit_lines=100
        ... )
        >>> analyzer = FortranAnalyzer(config)
        >>> results = analyzer.analyze()

    Attributes:
        config (FortranProjectConfig): Configuration for the analysis
        results (Dict[str, Any]): Analysis results (populated after analyze() is called)
        parser (FortranParser): Parser for Fortran source files
        call_graph_builder (CallGraphBuilder): Builder for dependency graphs
        decomposer (TranslationUnitDecomposer): Decomposer for translation units
        visualizer (FortranVisualizer): Visualizer for graphs and charts
    """

    def __init__(self, config: FortranProjectConfig):
        """
        Initialize the FortranAnalyzer with a configuration.

        Args:
            config (FortranProjectConfig): Configuration object defining project settings

        Raises:
            ValueError: If configuration is invalid (e.g., project root doesn't exist)
        """
        self.config = config
        self.results: Dict[str, Any] = {}

        # Validate configuration first
        if not config.validate():
            raise ValueError(
                f"Invalid configuration: {config.project_root} does not exist or is not accessible"
            )

        # Initialize components
        self.parser = FortranParser(config)
        self.call_graph_builder = CallGraphBuilder(config)
        self.decomposer = TranslationUnitDecomposer(config)
        self.visualizer = FortranVisualizer(config) if config.generate_graphs else None

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized FortranAnalyzer for project: {config.project_name}")

    def analyze(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Perform complete analysis of the Fortran codebase.

        This method executes the full analysis pipeline:
        1. Parse all Fortran source files
        2. Build module dependency and call graphs
        3. Decompose code into translation units
        4. Generate visualizations (if enabled)
        5. Generate recommendations
        6. Save results to files (if save_results=True)

        Args:
            save_results (bool, optional): Whether to save analysis results to files.
                Defaults to True. When True, creates:
                - analysis_results.json: Complete results in JSON format
                - analysis_summary.txt: Human-readable summary report
                - graphs/: Dependency graphs in various formats
                - translation_units.json: Translation unit decomposition

        Returns:
            Dict[str, Any]: Analysis results containing:
                - config: Analysis configuration and metadata
                - parsing: Parsed modules and statistics
                - dependencies: Dependency graphs and analysis
                - translation: Translation units and statistics
                - recommendations: Actionable insights and suggestions

        Raises:
            Exception: If analysis fails (e.g., unable to parse files)

        Example:
            >>> analyzer = create_analyzer_for_project('/path/to/project')
            >>> results = analyzer.analyze(save_results=True)
            >>> print(f"Analyzed {results['parsing']['statistics']['total_files']} files")
        """
        logger.info("Starting Fortran codebase analysis")
        start_time = time.time()

        try:
            # Step 1: Parse the codebase
            logger.info("Step 1: Parsing Fortran source files")
            parsing_results = self.parser.parse_project()
            modules = parsing_results.get("modules", {})

            if not modules:
                logger.warning("No modules found in the codebase")
                # Return proper structure even for empty projects
                return {
                    "config": {
                        "project_name": self.config.project_name,
                        "project_root": self.config.project_root,
                        "source_dirs": self.config.source_dirs,
                        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    "parsing": {
                        "modules": {},
                        "statistics": {
                            "total_files": 0,
                            "total_modules": 0,
                            "total_subroutines": 0,
                            "total_functions": 0,
                            "total_types": 0,
                            "lines_of_code": 0,
                        },
                    },
                    "dependencies": {
                        "module_graph_summary": {"nodes": 0, "edges": 0},
                        "analysis": {
                            "external_dependencies": [],
                            "hub_modules": [],
                            "leaf_modules": [],
                            "dependency_chains": [],
                        },
                    },
                    "translation": {
                        "units": 0,
                        "statistics": {
                            "total_units": 0,
                            "units_by_type": {},
                            "average_unit_size": 0,
                        },
                    },
                    "recommendations": {
                        "translation_strategy": "No modules found for translation",
                        "dependency_issues": [],
                        "optimization_opportunities": [],
                        "risks": [],
                    },
                }

            # Step 2: Build call graphs and dependency analysis
            logger.info("Step 2: Building call graphs and analyzing dependencies")
            module_graph = self.call_graph_builder.build_module_dependency_graph(
                modules
            )
            entity_graph = self.call_graph_builder.build_entity_call_graph(modules)

            dependency_analysis = self.call_graph_builder.analyze_dependencies()
            graph_metrics = self.call_graph_builder.calculate_metrics()

            # Step 3: Decompose into translation units
            logger.info("Step 3: Decomposing into translation units")
            translation_units = self.decomposer.decompose_modules(modules)
            translation_stats = self.decomposer.get_statistics()

            # Step 4: Generate visualizations
            visualization_files = {}
            if self.visualizer and self.config.generate_graphs:
                logger.info("Step 4: Generating visualizations")
                analysis_results = {
                    "call_graph_builder": self.call_graph_builder,
                    "translation_units": translation_units,
                    "statistics": parsing_results.get("statistics", {}),
                    "dependency_analysis": dependency_analysis,
                }
                visualization_files = self.visualizer.generate_all_visualizations(
                    analysis_results
                )

            # Compile results
            # FINAL ALIGNMENT FOR ORCHESTRATOR
            self.results = {
                "config": {
                    "project_name": self.config.project_name,
                    "project_root": self.config.project_root,
                    "analysis_timestamp": time.time(),
                    "analysis_duration": time.time() - start_time,
                },
                # The Orchestrator MUST find this key at the top level
                "modules": parsing_results.get("modules", {}), 
                
                # These are kept for reports but ignored by the translator
                "dependencies": {
                    "module_graph_summary": {
                        "nodes": module_graph.number_of_nodes() if module_graph else 0,
                        "edges": module_graph.number_of_edges() if module_graph else 0,
                    },
                    "analysis": dependency_analysis,
                    "metrics": graph_metrics,
                },
                "translation": {
                    "units": len(translation_units),
                    "statistics": translation_stats,
                    "translation_order": self.call_graph_builder.get_translation_order(),
                }
            }

            # Save results
            if save_results:
                self._save_analysis_results()
                self._export_graphs()
                self._export_translation_units(translation_units)

            elapsed_time = time.time() - start_time
            logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")

            return self.results

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    def _generate_recommendations(
        self, dependency_analysis: Dict, translation_stats: Dict, modules: Dict
    ) -> Dict[str, Any]:
        """Generate recommendations based on analysis results."""
        recommendations: Dict[str, List[str]] = {
            "translation_strategy": [],
            "dependency_issues": [],
            "optimization_opportunities": [],
            "risks": [],
        }

        # Translation strategy recommendations
        if translation_stats.get("total_units", 0) > 50:
            recommendations["translation_strategy"].append(
                "Large project detected. Consider incremental translation approach."
            )

        high_effort_units = translation_stats.get("units_by_effort", {}).get("high", 0)
        total_units = translation_stats.get("total_units", 1)

        if high_effort_units / total_units > 0.3:
            recommendations["translation_strategy"].append(
                "High percentage of complex units. Plan for extended timeline."
            )

        # Dependency issue recommendations
        circular_deps = dependency_analysis.get("circular_dependencies", [])
        if circular_deps:
            recommendations["dependency_issues"].append(
                f"Found {len(circular_deps)} circular dependencies. These must be resolved before translation."
            )

        external_deps = dependency_analysis.get("external_dependencies", [])
        if external_deps:
            recommendations["dependency_issues"].append(
                f"Project depends on {len(external_deps)} external libraries. Verify availability in target language."
            )

        # Optimization opportunities
        orphaned_modules = dependency_analysis.get("orphaned_modules", [])
        if orphaned_modules:
            recommendations["optimization_opportunities"].append(
                f"Found {len(orphaned_modules)} orphaned modules. Consider removing unused code."
            )

        hub_modules = dependency_analysis.get("hub_modules", [])
        if hub_modules:
            recommendations["optimization_opportunities"].append(
                f"Modules {hub_modules[:3]} are heavily used. Prioritize their translation."
            )

        # Risk assessment
        avg_lines = translation_stats.get("average_lines_per_unit", 0)
        if avg_lines > self.config.max_translation_unit_lines * 0.8:
            recommendations["risks"].append(
                "High average lines per unit. Consider reducing translation unit size."
            )

        if not dependency_analysis.get("is_dag", True):
            recommendations["risks"].append(
                "Module dependency graph contains cycles. This complicates translation."
            )

        return recommendations

    def _save_analysis_results(self) -> None:
        """Save analysis results to files."""
        # Save main results as JSON
        results_file = self.output_dir / "analysis_results.json"

        # Create a serializable copy
        serializable_results = self._make_serializable(self.results)

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Analysis results saved to {results_file}")

        # Save summary report
        self._save_summary_report()

    def _save_summary_report(self) -> None:
        """Save a human-readable summary report."""
        report_file = self.output_dir / "analysis_summary.txt"

        with open(report_file, "w") as f:
            f.write(f"Fortran Codebase Analysis Report\n")
            f.write(f"{'=' * 40}\n\n")

            f.write(f"Project: {self.config.project_name}\n")
            f.write(
                f"Analysis Date: {time.ctime(self.results['config']['analysis_timestamp'])}\n"
            )
            f.write(
                f"Duration: {self.results['config']['analysis_duration']:.2f} seconds\n\n"
            )

            # Parsing summary
            parsing = self.results.get("parsing", {})
            stats = parsing.get("statistics", {})

            f.write("Parsing Summary:\n")
            f.write(f"  Total Files: {stats.get('total_files', 0)}\n")
            f.write(f"  Total Lines: {stats.get('total_lines', 0):,}\n")
            f.write(f"  Modules: {len(parsing.get('modules', {}))}\n")
            f.write(f"  Subroutines: {stats.get('total_subroutines', 0)}\n")
            f.write(f"  Functions: {stats.get('total_functions', 0)}\n")
            f.write(f"  Types: {stats.get('total_types', 0)}\n\n")

            # Translation summary
            translation = self.results.get("translation", {})

            f.write("Translation Analysis:\n")
            f.write(f"  Translation Units: {translation.get('units', 0)}\n")

            t_stats = translation.get("statistics", {})
            units_by_effort = t_stats.get("units_by_effort", {})
            f.write(f"  Low Effort Units: {units_by_effort.get('low', 0)}\n")
            f.write(f"  Medium Effort Units: {units_by_effort.get('medium', 0)}\n")
            f.write(f"  High Effort Units: {units_by_effort.get('high', 0)}\n\n")

            # Dependency summary
            dependencies = self.results.get("dependencies", {})
            analysis = dependencies.get("analysis", {})

            f.write("Dependency Analysis:\n")
            f.write(
                f"  Module Dependencies: {dependencies.get('module_graph_summary', {}).get('edges', 0)}\n"
            )
            f.write(
                f"  Circular Dependencies: {len(analysis.get('circular_dependencies', []))}\n"
            )
            f.write(
                f"  External Dependencies: {len(analysis.get('external_dependencies', []))}\n"
            )
            f.write(f"  Hub Modules: {len(analysis.get('hub_modules', []))}\n\n")

            # Recommendations
            recommendations = self.results.get("recommendations", {})

            if recommendations:
                f.write("Recommendations:\n")

                for category, items in recommendations.items():
                    if items:
                        f.write(f"  {category.replace('_', ' ').title()}:\n")
                        for item in items:
                            f.write(f"    - {item}\n")
                        f.write("\n")

        logger.info(f"Summary report saved to {report_file}")

    def _export_graphs(self) -> None:
        """Export graph data in various formats."""
        if not hasattr(self, "call_graph_builder"):
            return

        graphs_dir = self.output_dir / "graphs"
        graphs_dir.mkdir(exist_ok=True)

        try:
            exported_files = self.call_graph_builder.export_graphs(
                graphs_dir, formats=["graphml", "json"]
            )
            logger.info(f"Exported {len(exported_files)} graph files")
        except Exception as e:
            logger.error(f"Failed to export graphs: {e}")

    def _export_translation_units(self, translation_units: List) -> None:
        """Export translation units data."""
        if not translation_units:
            return

        units_file = self.output_dir / "translation_units.json"

        try:
            self.decomposer.export_units(units_file, format="json")
            logger.info(f"Translation units exported to {units_file}")
        except Exception as e:
            logger.error(f"Failed to export translation units: {e}")

    def _make_serializable(self, obj) -> Union[Dict, List, str, int, float, bool, None]:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            try:
                json.dumps(obj)  # Test if serializable
                return obj
            except (TypeError, ValueError):
                return str(obj)

    def get_results(self) -> Dict[str, Any]:
        """
        Get the complete analysis results.

        Returns:
            Dict[str, Any]: Full analysis results dictionary

        Note:
            Results are only available after calling analyze()
        """
        return self.results

    def get_translation_order(self) -> List[str]:
        """
        Get the recommended order for translating modules.

        The order is based on dependency analysis - modules with no dependencies
        come first, allowing you to translate the codebase incrementally without
        breaking dependencies.

        Returns:
            List[str]: List of module names in recommended translation order

        Example:
            >>> analyzer = create_analyzer_for_project('/path/to/project')
            >>> results = analyzer.analyze()
            >>> order = analyzer.get_translation_order()
            >>> print(f"Start by translating: {order[0]}")
        """
        return self.call_graph_builder.get_translation_order()

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get high-level summary statistics for the analyzed codebase.

        Returns:
            Dict[str, Any]: Summary containing:
                - files: Total number of files analyzed
                - lines: Total lines of code
                - modules: Number of modules found
                - translation_units: Number of translation units
                - dependencies: Number of module dependencies
                - circular_dependencies: Number of circular dependency chains
                - external_dependencies: Number of external library dependencies

        Example:
            >>> analyzer = create_analyzer_for_project('/path/to/project')
            >>> results = analyzer.analyze()
            >>> stats = analyzer.get_summary_statistics()
            >>> print(f"Project has {stats['modules']} modules and {stats['lines']:,} lines")
        """
        if not self.results:
            return {}

        return {
            "files": self.results.get("parsing", {})
            .get("statistics", {})
            .get("total_files", 0),
            "lines": self.results.get("parsing", {})
            .get("statistics", {})
            .get("total_lines", 0),
            "modules": len(self.results.get("parsing", {}).get("modules", {})),
            "translation_units": self.results.get("translation", {}).get("units", 0),
            "dependencies": self.results.get("dependencies", {})
            .get("module_graph_summary", {})
            .get("edges", 0),
            "circular_dependencies": len(
                self.results.get("dependencies", {})
                .get("analysis", {})
                .get("circular_dependencies", [])
            ),
            "external_dependencies": len(
                self.results.get("dependencies", {})
                .get("analysis", {})
                .get("external_dependencies", [])
            ),
        }


def create_analyzer_from_config_file(config_path: Union[str, Path]) -> FortranAnalyzer:
    """
    Create a FortranAnalyzer from a configuration file.

    Args:
        config_path (Union[str, Path]): Path to YAML or JSON configuration file

    Returns:
        FortranAnalyzer: Configured analyzer instance

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration file format is unsupported

    Example:
        >>> analyzer = create_analyzer_from_config_file('my_config.yaml')
        >>> results = analyzer.analyze()
    """
    from .config.project_config import load_config

    config = load_config(config_path)
    return FortranAnalyzer(config)


def create_analyzer_for_project(
    project_root: str, template: str = "auto", **config_overrides
) -> FortranAnalyzer:
    """
    Create a FortranAnalyzer using a predefined template.

    This is the recommended way to create an analyzer for most use cases.
    The template system provides sensible defaults for different project types.

    Args:
        project_root (str): Path to the Fortran project root directory
        template (str, optional): Template name. Options are:
            - 'auto': Auto-detect project type (recommended)
            - 'ctsm': Community Terrestrial Systems Model
            - 'scientific_computing': General scientific computing projects
            - 'numerical_library': Numerical libraries
            - 'climate_model': Climate and atmospheric models
            - 'generic': Generic Fortran projects
            Defaults to 'auto'.
        **config_overrides: Additional keyword arguments to override template settings.
            Common overrides:
            - source_dirs (List[str]): Source directories to analyze
            - output_dir (str): Output directory for results
            - max_translation_unit_lines (int): Maximum lines per translation unit
            - generate_graphs (bool): Whether to generate visualizations

    Returns:
        FortranAnalyzer: Configured analyzer instance ready for analysis

    Example:
        Basic usage with auto-detection:

        >>> analyzer = create_analyzer_for_project('/path/to/project', template='auto')
        >>> results = analyzer.analyze()

        With custom settings:

        >>> analyzer = create_analyzer_for_project(
        ...     '/path/to/project',
        ...     template='scientific_computing',
        ...     max_translation_unit_lines=100,
        ...     output_dir='custom_output'
        ... )
        >>> results = analyzer.analyze()

        Analyze multiple projects:

        >>> for project_path in ['/path/a', '/path/b', '/path/c']:
        ...     analyzer = create_analyzer_for_project(project_path, template='auto')
        ...     results = analyzer.analyze()
    """
    from .config.project_config import create_default_config
    from pathlib import Path

    config = create_default_config(project_root, template)

    # Apply any overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            # Handle relative paths for output_dir
            if key == "output_dir" and not Path(value).is_absolute():
                value = str(Path(project_root) / value)
            setattr(config, key, value)

    return FortranAnalyzer(config)


def quick_analyze(
    project_root: str, template: str = "auto", output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform a quick analysis of a Fortran project with minimal configuration.

    This is the simplest way to analyze a Fortran codebase. It auto-detects the
    project type, analyzes the code, and returns results.

    Args:
        project_root (str): Path to the Fortran project root directory
        template (str, optional): Template name ('auto', 'ctsm', 'scientific_computing', etc.).
            Defaults to 'auto' which auto-detects the project type.
        output_dir (Optional[str], optional): Directory for output files.
            Defaults to None, which uses 'output/' in the project root.

    Returns:
        Dict[str, Any]: Complete analysis results. Access results with:
            - results['parsing']['modules']: All parsed modules
            - results['parsing']['statistics']: File and code statistics
            - results['dependencies']['analysis']: Dependency analysis
            - results['translation']['units']: Translation units
            - results['recommendations']: Actionable recommendations

    Example:
        Simplest usage:

        >>> from fortran_analyzer import quick_analyze
        >>> results = quick_analyze('/path/to/fortran/project')
        >>> print(f"Found {len(results['parsing']['modules'])} modules")

        With custom output directory:

        >>> results = quick_analyze('/path/to/project', output_dir='my_analysis')

        Analyze multiple projects:

        >>> projects = ['/project1', '/project2', '/project3']
        >>> all_results = {}
        >>> for path in projects:
        ...     all_results[path] = quick_analyze(path)
        >>> print(f"Analyzed {len(all_results)} projects")
    """
    config_overrides = {}
    if output_dir:
        config_overrides["output_dir"] = output_dir

    analyzer = create_analyzer_for_project(project_root, template, **config_overrides)
    return analyzer.analyze()
