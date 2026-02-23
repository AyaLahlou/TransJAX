"""
Visualization components for Fortran analysis results.
Creates graphs, charts, and other visual representations of code structure and dependencies.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

try:
    import seaborn as sns

    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..config.project_config import FortranProjectConfig
from ..analysis.call_graph_builder import CallGraphBuilder
from ..analysis.translation_decomposer import TranslationUnit

logger = logging.getLogger(__name__)


class FortranVisualizer:
    """Creates visualizations for Fortran code analysis results."""

    def __init__(self, config: FortranProjectConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up matplotlib style
        plt.style.use("default")
        if SEABORN_AVAILABLE:
            sns.set_palette("husl")

    def visualize_module_dependencies(
        self, call_graph_builder: CallGraphBuilder, save_path: Optional[Path] = None
    ) -> Optional[Path]:
        """Create a visualization of module dependencies."""
        module_graph = call_graph_builder.get_module_graph()

        if not module_graph.nodes():
            logger.warning("No module graph to visualize")
            return None

        # Filter out external dependencies for cleaner visualization
        internal_nodes = [
            n
            for n in module_graph.nodes()
            if not module_graph.nodes[n].get("external", False)
        ]
        internal_graph = module_graph.subgraph(internal_nodes)

        if not internal_graph.nodes():
            logger.warning("No internal modules to visualize")
            return None

        # Set up the plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Create layout
        try:
            pos = nx.spring_layout(internal_graph, k=3, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(internal_graph)

        # Draw nodes
        node_sizes = [
            internal_graph.nodes[n].get("line_count", 100) / 10 + 100
            for n in internal_graph.nodes()
        ]
        node_colors = [
            internal_graph.nodes[n].get("subroutines", 0)
            + internal_graph.nodes[n].get("functions", 0)
            for n in internal_graph.nodes()
        ]

        nodes = nx.draw_networkx_nodes(
            internal_graph,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap="viridis",
            alpha=0.8,
            ax=ax,
        )

        # Draw edges
        nx.draw_networkx_edges(
            internal_graph,
            pos,
            edge_color="gray",
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle="->",
            ax=ax,
        )

        # Draw labels
        nx.draw_networkx_labels(
            internal_graph, pos, font_size=8, font_weight="bold", ax=ax
        )

        # Customize plot
        ax.set_title(
            f"{self.config.project_name} - Module Dependencies",
            fontsize=16,
            fontweight="bold",
        )
        ax.axis("off")

        # Add colorbar
        if nodes:
            cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
            cbar.set_label("Number of Procedures", rotation=270, labelpad=20)

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=10,
                label="Module",
            ),
            plt.Line2D([0], [0], color="gray", label="Dependency"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "module_dependencies.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Module dependency visualization saved to {save_path}")
        return save_path

    def visualize_translation_units(
        self, translation_units: List[TranslationUnit], save_path: Optional[Path] = None
    ) -> Optional[Path]:
        """Create visualizations for translation units."""
        if not translation_units:
            logger.warning("No translation units to visualize")
            return None

        # Create subplots
        fig, axes_array = plt.subplots(2, 2, figsize=(16, 12))  # type: ignore
        ax1, ax2, ax3, ax4 = axes_array[0, 0], axes_array[0, 1], axes_array[1, 0], axes_array[1, 1]  # type: ignore
        fig.suptitle(
            f"{self.config.project_name} - Translation Units Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Units by type
        unit_types: Dict[str, int] = {}
        for unit in translation_units:
            unit_types[unit.unit_type] = unit_types.get(unit.unit_type, 0) + 1

        colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, len(unit_types)))
        wedges, texts, autotexts = ax1.pie(
            unit_types.values(),
            labels=unit_types.keys(),
            autopct="%1.1f%%",
            colors=colors,
        )
        ax1.set_title("Translation Units by Type")

        # 2. Line count distribution
        line_counts = [unit.line_count for unit in translation_units]
        ax2.hist(line_counts, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax2.set_title("Distribution of Line Counts")
        ax2.set_xlabel("Lines of Code")
        ax2.set_ylabel("Number of Units")
        ax2.grid(True, alpha=0.3)

        # 3. Complexity scores
        complexity_scores = [unit.complexity_score for unit in translation_units]
        if max(complexity_scores) > 0:
            ax3.hist(
                complexity_scores,
                bins=15,
                alpha=0.7,
                color="lightcoral",
                edgecolor="black",
            )
            ax3.set_title("Distribution of Complexity Scores")
            ax3.set_xlabel("Complexity Score")
            ax3.set_ylabel("Number of Units")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(
                0.5,
                0.5,
                "No complexity data",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title("Complexity Scores (No Data)")

        # 4. Priority distribution
        priorities: Dict[int, int] = {}
        for unit in translation_units:
            priorities[unit.priority] = priorities.get(unit.priority, 0) + 1

        if priorities:
            priority_labels = [f"Priority {p}" for p in sorted(priorities.keys())]
            priority_values = [priorities[p] for p in sorted(priorities.keys())]
            bars = ax4.bar(
                priority_labels,
                priority_values,
                color="lightgreen",
                alpha=0.8,
                edgecolor="black",
            )
            ax4.set_title("Units by Translation Priority")
            ax4.set_ylabel("Number of Units")
            ax4.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                )
        else:
            ax4.text(
                0.5,
                0.5,
                "No priority data",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Translation Priorities (No Data)")

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "translation_units_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Translation units visualization saved to {save_path}")
        return save_path

    def create_project_overview(
        self, statistics: Dict[str, Any], save_path: Optional[Path] = None
    ) -> Optional[Path]:
        """Create a project overview visualization."""
        if not statistics:
            logger.warning("No statistics to visualize")
            return None

        fig, axes_array = plt.subplots(2, 2, figsize=(16, 12))  # type: ignore
        ax1, ax2, ax3, ax4 = axes_array[0, 0], axes_array[0, 1], axes_array[1, 0], axes_array[1, 1]  # type: ignore
        fig.suptitle(
            f"{self.config.project_name} - Project Overview",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Basic statistics
        basic_stats = {
            "Files": statistics.get("total_files", 0),
            "Lines": statistics.get("total_lines", 0),
            "Subroutines": statistics.get("total_subroutines", 0),
            "Functions": statistics.get("total_functions", 0),
            "Types": statistics.get("total_types", 0),
        }

        if any(basic_stats.values()):
            bars = ax1.bar(
                basic_stats.keys(),
                basic_stats.values(),
                color=["skyblue", "lightgreen", "orange", "pink", "lightcoral"],
            )
            ax1.set_title("Project Statistics")
            ax1.set_ylabel("Count")
            ax1.tick_params(axis="x", rotation=45)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                )
        else:
            ax1.text(
                0.5,
                0.5,
                "No statistics available",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax1.set_title("Project Statistics (No Data)")

        # 2. File size distribution (if available)
        if "file_sizes" in statistics:
            file_sizes = statistics["file_sizes"]
            ax2.hist(
                file_sizes, bins=20, alpha=0.7, color="lightblue", edgecolor="black"
            )
            ax2.set_title("File Size Distribution")
            ax2.set_xlabel("Lines of Code")
            ax2.set_ylabel("Number of Files")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "File size data not available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("File Size Distribution")

        # 3. Module complexity (placeholder)
        ax3.text(
            0.5,
            0.5,
            "Module Complexity\n(Feature not implemented)",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("Module Complexity")

        # 4. Dependencies summary
        if "external_dependencies" in statistics:
            ext_deps = statistics["external_dependencies"]
            if ext_deps:
                ax4.barh(range(len(ext_deps)), [1] * len(ext_deps))
                ax4.set_yticks(range(len(ext_deps)))
                ax4.set_yticklabels(ext_deps)
                ax4.set_title("External Dependencies")
                ax4.set_xlabel("Usage")
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "No external dependencies",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                )
                ax4.set_title("External Dependencies")
        else:
            ax4.text(
                0.5,
                0.5,
                "Dependency data not available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("External Dependencies")

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "project_overview.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Project overview visualization saved to {save_path}")
        return save_path

    def create_interactive_dependency_graph(
        self, call_graph_builder: CallGraphBuilder, save_path: Optional[Path] = None
    ) -> Optional[Path]:
        """Create an interactive dependency graph using Plotly."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, skipping interactive visualization")
            return None

        module_graph = call_graph_builder.get_module_graph()

        if not module_graph.nodes():
            logger.warning("No module graph to visualize")
            return None

        # Filter internal nodes
        internal_nodes = [
            n
            for n in module_graph.nodes()
            if not module_graph.nodes[n].get("external", False)
        ]
        internal_graph = module_graph.subgraph(internal_nodes)

        if not internal_graph.nodes():
            logger.warning("No internal modules to visualize")
            return None

        # Get layout
        try:
            pos = nx.spring_layout(internal_graph, k=3, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(internal_graph)

        # Extract node information
        node_trace = go.Scatter(
            x=[pos[node][0] for node in internal_graph.nodes()],
            y=[pos[node][1] for node in internal_graph.nodes()],
            mode="markers+text",
            text=list(internal_graph.nodes()),
            textposition="middle center",
            hovertemplate="<b>%{text}</b><br>"
            + "Lines: %{customdata[0]}<br>"
            + "Subroutines: %{customdata[1]}<br>"
            + "Functions: %{customdata[2]}<extra></extra>",
            customdata=[
                [
                    internal_graph.nodes[n].get("line_count", 0),
                    internal_graph.nodes[n].get("subroutines", 0),
                    internal_graph.nodes[n].get("functions", 0),
                ]
                for n in internal_graph.nodes()
            ],
            marker=dict(
                size=[
                    max(10, internal_graph.nodes[n].get("line_count", 100) / 50)
                    for n in internal_graph.nodes()
                ],
                color=[
                    internal_graph.nodes[n].get("subroutines", 0)
                    + internal_graph.nodes[n].get("functions", 0)
                    for n in internal_graph.nodes()
                ],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Procedures"),
                line=dict(width=2, color="black"),
            ),
        )

        # Extract edge information
        edge_x = []
        edge_y = []

        for edge in internal_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color="gray"),
            hoverinfo="none",
            mode="lines",
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"{self.config.project_name} - Interactive Module Dependencies",
                title_font_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Dependencies between modules",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(color="gray", size=12),
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        # Save figure
        if save_path is None:
            save_path = self.output_dir / "interactive_dependencies.html"

        fig.write_html(str(save_path))
        logger.info(f"Interactive dependency graph saved to {save_path}")
        return save_path

    def create_translation_priority_chart(
        self, translation_units: List[TranslationUnit], save_path: Optional[Path] = None
    ) -> Optional[Path]:
        """Create a detailed chart showing translation priority and complexity."""
        if not translation_units:
            logger.warning("No translation units to visualize")
            return None

        # Sort units by priority and complexity
        sorted_units = sorted(
            translation_units, key=lambda u: (u.priority, u.complexity_score)
        )

        fig, axes_array = plt.subplots(2, 1, figsize=(14, 10))  # type: ignore
        ax1, ax2 = axes_array[0], axes_array[1]  # type: ignore
        fig.suptitle(
            f"{self.config.project_name} - Translation Priority Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Priority vs Complexity scatter plot
        priorities = [u.priority for u in translation_units]
        complexities = [u.complexity_score for u in translation_units]
        line_counts = [u.line_count for u in translation_units]

        scatter = ax1.scatter(
            priorities,
            complexities,
            s=line_counts,
            alpha=0.6,
            c=line_counts,
            cmap="viridis",
        )
        ax1.set_xlabel("Translation Priority")
        ax1.set_ylabel("Complexity Score")
        ax1.set_title("Translation Units: Priority vs Complexity")
        ax1.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("Lines of Code", rotation=270, labelpad=20)

        # 2. Translation order timeline
        unit_names = [
            (
                f"{u.module_name}::{u.entity_name}"[:30] + "..."
                if len(f"{u.module_name}::{u.entity_name}") > 30
                else f"{u.module_name}::{u.entity_name}"
            )
            for u in sorted_units[:20]
        ]  # Show top 20

        y_pos = np.arange(len(unit_names))
        efforts = [u.estimated_effort for u in sorted_units[:20]]

        effort_colors = {"low": "lightgreen", "medium": "orange", "high": "red"}
        colors = [effort_colors.get(effort, "gray") for effort in efforts]

        bars = ax2.barh(y_pos, [1] * len(unit_names), color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(unit_names, fontsize=8)
        ax2.set_xlabel("Translation Order (First → Last)")
        ax2.set_title("Recommended Translation Order (Top 20 Units)")
        ax2.grid(True, alpha=0.3, axis="x")

        # Add legend for effort levels
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="lightgreen", label="Low Effort"),
            Patch(facecolor="orange", label="Medium Effort"),
            Patch(facecolor="red", label="High Effort"),
        ]
        ax2.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "translation_priority_chart.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Translation priority chart saved to {save_path}")
        return save_path

    def generate_all_visualizations(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Path]:
        """Generate all available visualizations."""
        logger.info("Generating all visualizations")

        generated_files = {}

        try:
            # Module dependencies
            if "call_graph_builder" in analysis_results:
                path = self.visualize_module_dependencies(
                    analysis_results["call_graph_builder"]
                )
                if path:
                    generated_files["module_dependencies"] = path

                # Interactive version
                path = self.create_interactive_dependency_graph(
                    analysis_results["call_graph_builder"]
                )
                if path:
                    generated_files["interactive_dependencies"] = path

            # Translation units
            if "translation_units" in analysis_results:
                path = self.visualize_translation_units(
                    analysis_results["translation_units"]
                )
                if path:
                    generated_files["translation_units"] = path

                path = self.create_translation_priority_chart(
                    analysis_results["translation_units"]
                )
                if path:
                    generated_files["translation_priority"] = path

            # Project overview
            if "statistics" in analysis_results:
                path = self.create_project_overview(analysis_results["statistics"])
                if path:
                    generated_files["project_overview"] = path

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

        logger.info(f"Generated {len(generated_files)} visualizations")
        return generated_files
