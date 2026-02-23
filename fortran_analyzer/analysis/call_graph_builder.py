"""
Generic call graph builder for Fortran projects.
Builds dependency graphs and call relationships between modules and procedures.
"""

import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
import logging

from ..parser.fortran_parser import ModuleInfo, FortranEntity
from ..config.project_config import FortranProjectConfig

logger = logging.getLogger(__name__)


class CallGraphBuilder:
    """Builds call graphs and dependency relationships for Fortran projects."""

    def __init__(self, config: FortranProjectConfig):
        self.config = config
        self.module_graph = nx.DiGraph()
        self.call_graph = nx.DiGraph()
        self.entity_graph = nx.DiGraph()

    def build_module_dependency_graph(
        self, modules: Dict[str, ModuleInfo]
    ) -> nx.DiGraph:
        """Build module-level dependency graph."""
        logger.info("Building module dependency graph")

        self.module_graph.clear()

        # Add all modules as nodes
        for module_name, module_info in modules.items():
            self.module_graph.add_node(
                module_name,
                file_path=module_info.file_path,
                line_count=module_info.line_count,
                subroutines=len(module_info.subroutines),
                functions=len(module_info.functions),
                types=len(module_info.types),
            )

        # Add dependency edges
        for module_name, module_info in modules.items():
            for use in module_info.uses:
                dep_module = use["module"]

                # Ensure dep_module is a string
                if not isinstance(dep_module, str):
                    continue

                # Skip system modules if configured
                if self._is_system_module(dep_module):
                    continue

                # Only add edge if target module exists in our project
                if dep_module in modules:
                    self.module_graph.add_edge(
                        module_name,
                        dep_module,
                        dependency_type="use",
                        only_items=use.get("only", []),
                    )
                else:
                    # External dependency
                    if self.config.track_dependencies and isinstance(dep_module, str):
                        self.module_graph.add_node(
                            dep_module,
                            external=True,
                            library=self._classify_external_library(dep_module),
                        )
                        self.module_graph.add_edge(
                            module_name, dep_module, dependency_type="external_use"
                        )

        logger.info(
            f"Module graph: {self.module_graph.number_of_nodes()} nodes, "
            f"{self.module_graph.number_of_edges()} edges"
        )

        return self.module_graph

    def build_entity_call_graph(self, modules: Dict[str, ModuleInfo]) -> nx.DiGraph:
        """Build entity-level call graph (subroutines, functions)."""
        logger.info("Building entity call graph")

        self.entity_graph.clear()

        # Add all entities as nodes
        for module_name, module_info in modules.items():
            for entity in module_info.entities:
                node_id = f"{module_name}::{entity.name}"
                self.entity_graph.add_node(
                    node_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    module=module_name,
                    file_path=entity.file_path,
                    line_start=entity.line_start,
                    line_end=entity.line_end,
                    line_count=(
                        entity.line_end - entity.line_start + 1
                        if entity.line_end > 0
                        else 0
                    ),
                )

        # Build call relationships (this would require more sophisticated parsing)
        # For now, we'll create a basic structure
        self._extract_call_relationships(modules)

        logger.info(
            f"Entity graph: {self.entity_graph.number_of_nodes()} nodes, "
            f"{self.entity_graph.number_of_edges()} edges"
        )

        return self.entity_graph

    def _extract_call_relationships(self, modules: Dict[str, ModuleInfo]) -> None:
        """Extract call relationships between procedures (simplified implementation)."""
        # This is a simplified implementation. A full implementation would
        # require parsing the body of each subroutine/function to find CALL statements

        for module_name, module_info in modules.items():
            try:
                with open(
                    module_info.file_path, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    content = f.read()

                # Find CALL statements using regex
                import re

                call_pattern = r"call\s+(\w+)(?:\s*\(|$)"
                calls = re.findall(call_pattern, content, re.IGNORECASE)

                # For each entity in this module, check if it makes calls
                for entity in module_info.entities:
                    if entity.entity_type in ["subroutine", "function"]:
                        caller_id = f"{module_name}::{entity.name}"

                        # Get the content of this specific entity
                        if entity.line_start > 0 and entity.line_end > 0:
                            lines = content.splitlines()
                            entity_content = "\n".join(
                                lines[entity.line_start - 1 : entity.line_end]
                            )

                            # Find calls within this entity
                            entity_calls = re.findall(
                                call_pattern, entity_content, re.IGNORECASE
                            )

                            for called_proc in entity_calls:
                                # Try to find the called procedure in our modules
                                for (
                                    target_module_name,
                                    target_module,
                                ) in modules.items():
                                    if (
                                        called_proc in target_module.subroutines
                                        or called_proc in target_module.functions
                                    ):
                                        callee_id = (
                                            f"{target_module_name}::{called_proc}"
                                        )
                                        if self.entity_graph.has_node(callee_id):
                                            self.entity_graph.add_edge(
                                                caller_id,
                                                callee_id,
                                                call_type="procedure_call",
                                            )
                                            break

            except Exception as e:
                logger.warning(
                    f"Failed to extract calls from {module_info.file_path}: {e}"
                )

    def _is_system_module(self, module_name: str) -> bool:
        """Check if a module is a system/intrinsic module."""
        return module_name in self.config.system_modules

    def _classify_external_library(self, module_name: str) -> Optional[str]:
        """Classify which external library a module belongs to."""
        module_lower = module_name.lower()

        for library in self.config.external_libraries:
            if library.lower() in module_lower:
                return library

        # Common patterns
        if "netcdf" in module_lower or "nf90" in module_lower:
            return "netcdf"
        elif "mpi" in module_lower:
            return "mpi"
        elif "hdf" in module_lower:
            return "hdf5"
        elif "lapack" in module_lower or "blas" in module_lower:
            return "linear_algebra"
        elif "esmf" in module_lower:
            return "esmf"

        return "unknown"

    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency patterns and identify issues."""
        analysis: Dict[str, Any] = {
            "circular_dependencies": [],
            "strongly_connected_components": [],
            "dependency_levels": {},
            "external_dependencies": [],
            "orphaned_modules": [],
            "hub_modules": [],
            "leaf_modules": [],
        }

        if not self.module_graph.nodes():
            return analysis

        # Find circular dependencies
        try:
            cycles = list(nx.simple_cycles(self.module_graph))
            analysis["circular_dependencies"] = cycles
        except:
            logger.warning("Could not detect cycles in module graph")

        # Find strongly connected components
        try:
            sccs = list(nx.strongly_connected_components(self.module_graph))
            analysis["strongly_connected_components"] = [
                list(scc) for scc in sccs if len(scc) > 1
            ]
        except:
            logger.warning("Could not find strongly connected components")

        # Calculate dependency levels (topological sort)
        try:
            # Remove external nodes for topological sort
            internal_graph = self.module_graph.subgraph(
                [
                    n
                    for n in self.module_graph.nodes()
                    if not self.module_graph.nodes[n].get("external", False)
                ]
            )

            if nx.is_directed_acyclic_graph(internal_graph):
                topo_order = list(nx.topological_sort(internal_graph))
                levels = {}
                for i, node in enumerate(topo_order):
                    levels[node] = i
                analysis["dependency_levels"] = levels
        except:
            logger.warning("Could not calculate dependency levels")

        # Find external dependencies
        external_deps = [
            n
            for n in self.module_graph.nodes()
            if self.module_graph.nodes[n].get("external", False)
        ]
        analysis["external_dependencies"] = external_deps

        # Find orphaned modules (no dependencies)
        orphaned = [
            n
            for n in self.module_graph.nodes()
            if (
                self.module_graph.in_degree(n) == 0
                and self.module_graph.out_degree(n) == 0
                and not self.module_graph.nodes[n].get("external", False)
            )
        ]
        analysis["orphaned_modules"] = orphaned

        # Find hub modules (high in-degree)
        in_degrees = dict(self.module_graph.in_degree())
        hub_threshold = max(
            2, len(self.module_graph.nodes()) * 0.1
        )  # At least 10% of modules
        hubs = [
            n
            for n, degree in in_degrees.items()
            if degree >= hub_threshold
            and not self.module_graph.nodes[n].get("external", False)
        ]
        analysis["hub_modules"] = sorted(
            hubs, key=lambda x: in_degrees[x], reverse=True
        )

        # Find leaf modules (no outgoing dependencies)
        leaves = [
            n
            for n in self.module_graph.nodes()
            if (
                self.module_graph.out_degree(n) == 0
                and not self.module_graph.nodes[n].get("external", False)
            )
        ]
        analysis["leaf_modules"] = leaves

        return analysis

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate graph metrics."""
        metrics = {}

        if self.module_graph.nodes():
            # Module graph metrics
            metrics["module_graph"] = {
                "nodes": self.module_graph.number_of_nodes(),
                "edges": self.module_graph.number_of_edges(),
                "density": nx.density(self.module_graph),
                "average_degree": sum(dict(self.module_graph.degree()).values())
                / self.module_graph.number_of_nodes(),
                "is_connected": nx.is_weakly_connected(self.module_graph),
                "is_dag": nx.is_directed_acyclic_graph(self.module_graph),
            }

            # Add centrality measures
            try:
                metrics["module_graph"]["in_degree_centrality"] = (
                    nx.in_degree_centrality(self.module_graph)
                )
                metrics["module_graph"]["out_degree_centrality"] = (
                    nx.out_degree_centrality(self.module_graph)
                )
                metrics["module_graph"]["betweenness_centrality"] = (
                    nx.betweenness_centrality(self.module_graph)
                )
            except:
                logger.warning("Could not calculate centrality measures")

        if self.entity_graph.nodes():
            # Entity graph metrics
            metrics["entity_graph"] = {
                "nodes": self.entity_graph.number_of_nodes(),
                "edges": self.entity_graph.number_of_edges(),
                "density": nx.density(self.entity_graph),
                "is_connected": nx.is_weakly_connected(self.entity_graph),
            }

        return metrics

    def get_translation_order(self) -> List[str]:
        """Get suggested translation order based on dependencies."""
        if not self.module_graph.nodes():
            return []

        # Remove external dependencies for ordering
        internal_nodes = [
            n
            for n in self.module_graph.nodes()
            if not self.module_graph.nodes[n].get("external", False)
        ]

        if not internal_nodes:
            return []

        internal_graph = self.module_graph.subgraph(internal_nodes)

        if nx.is_directed_acyclic_graph(internal_graph):
            # Use topological sort for DAG - reverse to get dependencies first
            return list(reversed(list(nx.topological_sort(internal_graph))))
        else:
            # For graphs with cycles, use a heuristic approach
            # Start with nodes that have no incoming edges
            order = []
            remaining = set(internal_nodes)

            while remaining:
                # Find nodes with no incoming edges from remaining nodes
                candidates = [
                    n
                    for n in remaining
                    if all(
                        pred not in remaining for pred in internal_graph.predecessors(n)
                    )
                ]

                if not candidates:
                    # Break cycles by choosing node with lowest in-degree
                    candidates = [
                        min(remaining, key=lambda x: internal_graph.in_degree(x))
                    ]

                # Sort candidates by some priority (e.g., line count, complexity)
                candidates.sort(
                    key=lambda x: self.module_graph.nodes[x].get("line_count", 0)
                )

                chosen = candidates[0]
                order.append(chosen)
                remaining.remove(chosen)

            return order

    def export_graphs(
        self, output_dir: Path, formats: List[str] = ["graphml", "gexf"]
    ) -> Dict[str, str]:
        """Export graphs with environment-safe imports."""
        try:
            import networkx as nx
        except ImportError:
            logger.error("NetworkX not found. Skipping graph export.")
            return {}
        """Export graphs to various formats with reinforced None-type safety."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        def sanitize_graph(G):
            """Convert None to strings and Lists to strings for GraphML compatibility."""
            Safe_G = G.copy()
            for n in Safe_G.nodes():
                attrs = Safe_G.nodes[n]
                for k, v in list(attrs.items()):
                    if v is None: attrs[k] = ""
                    elif isinstance(v, list): attrs[k] = ", ".join(map(str, v))
            
            for u, v, attrs in Safe_G.edges(data=True):
                for key, val in list(attrs.items()):
                    if val is None: attrs[key] = ""
                    elif isinstance(val, list): attrs[key] = ", ".join(map(str, val))
            return Safe_G

        # Global try block to ensure one bad graph doesn't kill the Orchestrator
        try:
            for graph_name, graph in [
                ("module_graph", self.module_graph),
                ("entity_graph", self.entity_graph),
            ]:
                if graph is None or not graph.nodes():
                    logger.warning(f"Skipping {graph_name}: Graph is empty or None")
                    continue

                # Create a sanitized copy for export
                export_graph = sanitize_graph(graph)

                for format_name in formats:
                    filename = f"{graph_name}.{format_name}"
                    filepath = output_dir / filename

                    try:
                        if format_name == "graphml":
                            nx.write_graphml(export_graph, str(filepath))
                        elif format_name == "gexf":
                            nx.write_gexf(export_graph, str(filepath))
                        elif format_name == "json":
                            import json
                            data = nx.node_link_data(export_graph)
                            with open(filepath, "w") as f:
                                json.dump(data, f, indent=2)
                        
                        exported_files[f"{graph_name}_{format_name}"] = str(filepath)
                        logger.info(f"Exported {graph_name} to {filepath}")
                    except Exception as e:
                        # Log the error but keep going so other files/graphs are saved
                        logger.error(f"Failed to export {graph_name} to {format_name}: {e}")
        
        except Exception as e:
            logger.critical(f"Critical failure in export_graphs logic: {e}")

        return exported_files

    def get_module_graph(self) -> nx.DiGraph:
        """Get the module dependency graph."""
        return self.module_graph

    def get_entity_graph(self) -> nx.DiGraph:
        """Get the entity call graph."""
        return self.entity_graph
