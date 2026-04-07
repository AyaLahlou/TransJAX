"""
ProjectDocGenerator — generate DESIGN.md and CLAUDE.md from analyzer output.

Reads every artifact produced by ``transjax analyze``:
  - analysis_results.json   (modules, subroutines, dependency metrics, graph stats)
  - translation_order.json  (file-level topological order with depth/rank)
  - translation_units.json  (decomposed units with complexity/effort/priority)
  - translation_state.json  (per-module translation progress)

DESIGN.md (written to <dest>/DESIGN.md)
    Full architecture document for the target ESM: project statistics,
    dependency graph analysis, module breakdown with subroutine interfaces,
    translation units, data flow, parity tolerances, risk areas.

CLAUDE.md (written to <output_dir>/CLAUDE.md)
    Per-project agent context file: HPC rules, current status, full module
    order with complexity signals, conventions, critical rules.

Usage
-----
gen = ProjectDocGenerator(
    analysis_dir=Path("transjax_analysis"),
    output_dir=Path("pipeline_out"),
    fortran_dir=Path("fortran_src"),
    gcm_model_name="CTSM",
)
gen.write_all(dest_dir=Path("pipeline_out/docs"))
"""

import json
import logging
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Width constants for ASCII tables / diagrams
_W = 80


class ProjectDocGenerator:
    """Generate DESIGN.md, CLAUDE.md, and CHANGELOG.md from analyzer output."""

    def __init__(
        self,
        analysis_dir: Path,
        output_dir: Path,
        fortran_dir: Path,
        gcm_model_name: str = "ESM",
        parity_rtol: float = 1e-10,
        parity_atol: float = 1e-12,
        model_name: str = "claude-sonnet-4-6",
    ) -> None:
        self.analysis_dir   = Path(analysis_dir).resolve()
        self.output_dir     = Path(output_dir).resolve()
        self.fortran_dir    = Path(fortran_dir).resolve()
        self.gcm_model_name = gcm_model_name
        self.parity_rtol    = parity_rtol
        self.parity_atol    = parity_atol
        self.model_name     = model_name

        self._analysis: Optional[Dict[str, Any]] = None
        self._order: Optional[List[Dict[str, Any]]] = None
        self._units_data: Optional[Dict[str, Any]] = None
        self._state: Optional[Dict[str, Any]] = None

    # ──────────────────────────────────────────────────────────────────
    # Public entry points
    # ──────────────────────────────────────────────────────────────────

    def write_all(self, dest_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Write DESIGN.md, CLAUDE.md, CHANGELOG.md. Return {name: path}."""
        dest = Path(dest_dir).resolve() if dest_dir else self.output_dir / "docs"
        dest.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Path] = {}

        design_path = dest / "DESIGN.md"
        design_path.write_text(self.generate_design_md(), encoding="utf-8")
        paths["DESIGN.md"] = design_path

        claude_path = self.output_dir / "CLAUDE.md"
        claude_path.write_text(self.generate_claude_md(), encoding="utf-8")
        paths["CLAUDE.md"] = claude_path

        changelog_path = self.output_dir / "CHANGELOG.md"
        if not changelog_path.exists():
            changelog_path.write_text(self.generate_changelog(), encoding="utf-8")
            paths["CHANGELOG.md"] = changelog_path

        logger.info("Documentation written to %s", dest)
        return paths

    # ──────────────────────────────────────────────────────────────────
    # DESIGN.md
    # ──────────────────────────────────────────────────────────────────

    def generate_design_md(self) -> str:
        """Render a comprehensive DESIGN.md from all analyzer artifacts."""
        analysis = self._load_analysis()
        order    = self._load_order()
        units    = self._load_units()
        state    = self._load_state()

        # Convenience accessors
        parsing_stats  = analysis.get("parsing", {}).get("statistics", {})
        dep_analysis   = analysis.get("dependencies", {}).get("analysis", {})
        dep_metrics    = analysis.get("dependencies", {}).get("metrics", {})
        mod_graph      = dep_metrics.get("module_graph", {})
        trans_stats    = analysis.get("translation", {}).get("statistics", {})
        order_summary  = self._load_order_summary()
        modules_data   = analysis.get("modules", {})
        modules_state  = {e["module"]: e for e in state.get("ordered_modules", [])}

        n_files_total  = order_summary.get("total_files", len(order))
        n_done         = sum(1 for e in modules_state.values() if e.get("status") == "success")
        n_mod_total    = len([m for fe in order for m in fe.get("modules", [])])
        git_hash       = _git_head_hash(self.analysis_dir)

        lines: List[str] = []

        # ── Header ────────────────────────────────────────────────────
        lines += [
            f"# {self.gcm_model_name} — Fortran→JAX Translation Design",
            "",
            f"> **Generated**: {_now()}  ",
            f"> **Fortran source**: `{self.fortran_dir}`  ",
            f"> **Analysis dir**: `{self.analysis_dir}`  ",
            f"> **Git ref**: `{git_hash}`  ",
            f"> **Status**: {n_done}/{n_mod_total} modules translated",
            "",
        ]

        # ── Strategic Goals ───────────────────────────────────────────
        lines += _section("Strategic Goals") + [
            f"1. **Numerical parity** — every subroutine matches Fortran output within "
            f"`rtol={self.parity_rtol}`, `atol={self.parity_atol}` for all golden test cases.",
            "2. **Automatic differentiation** — all JAX code is JIT-compilable and supports "
            "`jax.grad` / `jax.jacobian` for sensitivity analysis and ML-based calibration.",
            "3. **GPU/TPU portability** — pure functions, no in-place mutation, `jnp` throughout.",
            "4. **Reproducibility** — deterministic translation (`temperature=0.0`) and "
            "test results across runs and platforms.",
            "5. **Resumability** — every pipeline step checkpointed; interrupted runs resume "
            "from the last completed module.",
            "",
        ]

        # ── Project Statistics ────────────────────────────────────────
        lines += _section("Project Statistics")
        lines += self._project_stats_table(parsing_stats, order_summary, trans_stats)

        # ── Architecture Overview ─────────────────────────────────────
        lines += _section("Architecture Overview")
        lines += self._architecture_diagram()

        # ── Data Flow ─────────────────────────────────────────────────
        lines += _section("Data Flow")
        lines += self._data_flow_section(order)

        # ── Dependency Graph Analysis ──────────────────────────────────
        lines += _section("Dependency Graph Analysis")
        lines += self._dependency_graph_section(dep_analysis, mod_graph, order_summary)

        # ── External Dependencies ─────────────────────────────────────
        ext_deps = dep_analysis.get("external_dependencies", [])
        if ext_deps:
            lines += _section("External Dependencies")
            lines += self._external_deps_section(ext_deps, modules_data)

        # ── Module Breakdown ──────────────────────────────────────────
        lines += _section("Module Breakdown")
        lines += self._module_breakdown_table(modules_data, order, modules_state)

        # ── Module Details ────────────────────────────────────────────
        lines += _section("Module Details")
        lines += self._module_details(modules_data, order)

        # ── Translation Units ─────────────────────────────────────────
        lines += _section("Translation Units")
        lines += self._translation_units_section(units, trans_stats)

        # ── Dependency Levels (parallel groups) ───────────────────────
        lines += _section("Implementation Order")
        lines += self._implementation_order(order, dep_analysis, modules_state)

        # ── Translation Pipeline Stages ───────────────────────────────
        lines += _section("Translation Pipeline Stages")
        lines += [
            "Each module passes through four sequential stages:",
            "",
            "| Stage | Agent | Input | Output | Skip condition |",
            "|---|---|---|---|---|",
            "| **1. FTest** | `FtestAgent` | Fortran source | Compiled Fortran driver programs | `--skip-ftest` |",
            "| **2. Golden** | `GoldenAgent` | Driver programs | `*.json` golden I/O reference data | `--skip-golden` |",
            "| **3. Translate** | `TranslatorAgent` | Fortran + analysis + golden context | `{module}.py` JAX module | File exists + state=success |",
            "| **4. Parity + Repair** | `ParityRepairAgent` | JAX module + golden data | Parity report + RALPH log | Previous PASS in report |",
            "",
            "The RALPH loop (Run→Assess→Loop→Patch→Halt) governs stages 3–4 repair iterations.",
            "",
        ]

        # ── Parity Tolerances ─────────────────────────────────────────
        lines += _section("Parity Tolerances")
        lines += [
            f"| Tolerance | Value | Scope |",
            f"|---|---|---|",
            f"| Relative (`rtol`) | `{self.parity_rtol}` | All subroutines, all golden cases |",
            f"| Absolute (`atol`) | `{self.parity_atol}` | All subroutines, all golden cases |",
            "",
            "A subroutine **PASSES** parity iff for every golden test case:",
            "```python",
            f"jnp.allclose(jax_output, fortran_output, rtol={self.parity_rtol}, atol={self.parity_atol})",
            "```",
            "Tolerances are configured in `default_config.yaml` and overridable per-run "
            "with `--rtol` / `--atol`.",
            "",
        ]

        # ── Testing Philosophy ────────────────────────────────────────
        lines += _section("Testing Philosophy")
        lines += [
            "- **Golden-first**: Trust compiled Fortran as the oracle. Parity tests compare "
            "against golden I/O captured from compiled Fortran drivers — not analytical expectations.",
            "- **RALPH repair loop**: Structured Run→Assess→Loop→Patch→Halt repair with a "
            "structured `{module}_ralph_log.json` written per repair attempt.",
            "- **No fudge factors**: Never relax tolerances to make a test pass. Fix the "
            "translation instead.",
            "- **Intermediate quantities**: Test subroutine internal outputs, not just final "
            "results — catches dtype and indexing bugs early.",
            "- **Multi-case coverage**: Golden agent generates physically diverse cases "
            "(tropical/polar, midday/midnight, boundary conditions) to stress edge cases.",
            "",
        ]

        # ── Risk Areas ────────────────────────────────────────────────
        lines += _section("Risk Areas")
        lines += self._risk_areas(dep_analysis, units, order_summary)

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    # DESIGN.md section renderers
    # ──────────────────────────────────────────────────────────────────

    def _project_stats_table(
        self,
        parsing_stats: Dict[str, Any],
        order_summary: Dict[str, Any],
        trans_stats: Dict[str, Any],
    ) -> List[str]:
        lines = [
            "| Metric | Value |",
            "|---|---|",
        ]
        pairs: List[Tuple[str, Any]] = [
            ("Fortran source files",   parsing_stats.get("total_files", "—")),
            ("Total lines of code",    f"{parsing_stats.get('total_lines', 0):,}"),
            ("Modules",                parsing_stats.get("total_modules", "—")),
            ("Subroutines",            parsing_stats.get("total_subroutines", "—")),
            ("Functions",              parsing_stats.get("total_functions", "—")),
            ("Derived types",          parsing_stats.get("total_types", "—")),
            ("Translation units",      trans_stats.get("total_units", "—")),
            ("Avg lines/unit",         f"{trans_stats.get('average_lines_per_unit', 0):.1f}"),
            ("Largest unit",           trans_stats.get("largest_unit", "—")),
            ("Most complex unit",      trans_stats.get("most_complex_unit", "—")),
            ("Dependency graph depth", order_summary.get("max_depth", "—")),
            ("Is DAG (no cycles)",     "✓ Yes" if order_summary.get("is_dag", True) else "✗ No — cycles present"),
        ]
        for label, val in pairs:
            lines.append(f"| {label} | {val} |")
        lines.append("")

        # Effort breakdown
        by_effort = trans_stats.get("units_by_effort", {})
        if by_effort:
            lines += [
                "**Translation units by effort:**",
                "",
                "| Effort | Count | Description |",
                "|---|---|---|",
                f"| Low    | {by_effort.get('low', 0):>4} | complexity < 5  (simple utility routines) |",
                f"| Medium | {by_effort.get('medium', 0):>4} | complexity 5–15 (standard physics routines) |",
                f"| High   | {by_effort.get('high', 0):>4} | complexity > 15 (large, interface-heavy routines) |",
                "",
            ]
        return lines

    def _architecture_diagram(self) -> List[str]:
        return [
            "```",
            "Fortran source (.f90 / .F90)",
            "         │",
            "         ▼",
            "  ┌──────────────────────────┐",
            "  │     transjax analyze      │  fparser2 · call-graph · decomposer",
            "  └──────────────────────────┘",
            "         │",
            "         ├─ analysis_results.json    (modules, subroutines, graph metrics)",
            "         ├─ translation_order.json   (topological file ordering)",
            "         ├─ translation_units.json   (decomposed units, complexity, effort)",
            "         └─ graphs/                  (module_graph.graphml, entity_graph.json)",
            "         │",
            "         ▼",
            "  ┌──────────────────────────┐",
            "  │    transjax pipeline      │  PipelineRunner (one Slurm job per module)",
            "  └──────────────────────────┘",
            "         │",
            "         ├─[1] FtestAgent          Fortran source → compiled driver programs",
            "         ├─[2] GoldenAgent         drivers → golden I/O JSON (N cases/subroutine)",
            "         ├─[3] TranslatorAgent      Fortran + units → {module}.py JAX module",
            "         └─[4] ParityRepairAgent    JAX module + golden → parity report",
            "                   │",
            "                   └─ RALPH loop (Run→Assess→Loop→Patch→Halt)",
            "                         │",
            "                         └─ {module}_ralph_log.json",
            "```",
            "",
        ]

    def _data_flow_section(self, order: List[Dict[str, Any]]) -> List[str]:
        lines = [
            "The translation pipeline processes modules in dependency order. "
            "Each module's output feeds into downstream modules:",
            "",
            "```",
            "Fortran USE dependencies  →  translation_order.json  →  PipelineRunner",
            "                                                               │",
            "                                              ┌────────────────┴────────────────┐",
            "                                              │ For each module (depth 0 first): │",
            "                                              │  1. FTest drivers compiled        │",
            "                                              │  2. Golden I/O captured           │",
            "                                              │  3. JAX module generated          │",
            "                                              │  4. Parity verified + repaired    │",
            "                                              │  5. git commit (if auto_commit)   │",
            "                                              └───────────────────────────────────┘",
            "                                                               │",
            "                                              pipeline_state.json  (resumable)",
            "                                              translation_state.json",
            "```",
            "",
        ]

        # Show concrete data flow from modules
        if order:
            lines += ["**Module data flow (dependency edges):**", ""]
            shown = 0
            for fe in order:
                deps = fe.get("depends_on_files", [])
                mods = fe.get("modules", [])
                if deps and mods:
                    dep_names = [Path(d).stem for d in deps[:3]]
                    dep_str = ", ".join(dep_names) + (" …" if len(deps) > 3 else "")
                    lines.append(f"- `{', '.join(mods)}` ← depends on: {dep_str}")
                    shown += 1
                if shown >= 10:
                    lines.append(f"- _(… {len(order) - 10} more files)_")
                    break
            lines.append("")
        return lines

    def _dependency_graph_section(
        self,
        dep_analysis: Dict[str, Any],
        mod_graph: Dict[str, Any],
        order_summary: Dict[str, Any],
    ) -> List[str]:
        is_dag = order_summary.get("is_dag", True)
        n_cycles = len(dep_analysis.get("circular_dependencies", []))

        lines = [
            "| Graph Property | Value |",
            "|---|---|",
            f"| Nodes (modules) | {mod_graph.get('nodes', '—')} |",
            f"| Edges (USE dependencies) | {mod_graph.get('edges', '—')} |",
            f"| Graph density | {mod_graph.get('density', 0):.4f} |",
            f"| Average degree | {mod_graph.get('average_degree', 0):.2f} |",
            f"| Is DAG | {'Yes ✓' if is_dag else 'No ✗ — cycles present'} |",
            f"| Circular dependency groups | {n_cycles} |",
            f"| Max dependency depth | {order_summary.get('max_depth', '—')} |",
            f"| Orphaned modules | {len(dep_analysis.get('orphaned_modules', []))} |",
            "",
        ]

        # Hub modules (bottlenecks — high in-degree)
        hub_modules = dep_analysis.get("hub_modules", [])
        if hub_modules:
            lines += [
                f"**Hub modules** (many dependents — translate these carefully first):",
                "",
            ]
            in_centrality = mod_graph.get("in_degree_centrality", {})
            for hub in hub_modules[:10]:
                c = in_centrality.get(hub, 0)
                lines.append(f"- `{hub}` (in-degree centrality: {c:.3f})")
            lines.append("")

        # Leaf modules (high out-degree — depend on many others)
        leaf_modules = dep_analysis.get("leaf_modules", [])
        if leaf_modules:
            lines += [
                f"**Leaf modules** (depend on many others — translate these last):",
                "",
            ]
            for leaf in leaf_modules[:10]:
                lines.append(f"- `{leaf}`")
            lines.append("")

        # Circular dependencies
        cycles = dep_analysis.get("circular_dependencies", [])
        if cycles:
            lines += [
                f"**⚠ Circular dependencies detected ({n_cycles} group(s)):**",
                "",
                "These modules have mutual USE dependencies. The pipeline uses a "
                "heuristic to break cycles, but verify numerical correctness carefully.",
                "",
            ]
            for cycle in cycles[:5]:
                cycle_str = " → ".join(f"`{m}`" for m in cycle)
                if cycle:
                    cycle_str += f" → `{cycle[0]}`"
                lines.append(f"- {cycle_str}")
            if len(cycles) > 5:
                lines.append(f"- _(… {len(cycles) - 5} more cycles)_")
            lines.append("")

        # Depth distribution
        depth_groups = order_summary.get("depth_groups", {})
        if depth_groups:
            lines += [
                "**Files per dependency depth (parallelizable groups):**",
                "",
                "| Depth | Files | Can run in parallel |",
                "|---|---|---|",
            ]
            for depth_str in sorted(depth_groups, key=int):
                count = depth_groups[depth_str]
                parallel = "Yes" if count > 1 else "—"
                lines.append(f"| {depth_str} | {count} | {parallel} |")
            lines.append("")

        return lines

    def _external_deps_section(
        self, ext_deps: List[str], modules_data: Dict[str, Any]
    ) -> List[str]:
        lines = [
            "The following external Fortran modules are referenced but not translated "
            "(they must be available at link time or mocked in JAX):",
            "",
        ]
        # Group by likely library
        known_libs = {
            "netcdf": ["netcdf", "nc_"], "mpi": ["mpi", "mpi_"],
            "hdf5": ["hdf5", "h5_"], "lapack": ["lapack", "blas"],
            "shr": ["shr_"], "pio": ["pio"],
        }
        grouped: Dict[str, List[str]] = defaultdict(list)
        for dep in sorted(ext_deps):
            placed = False
            for lib, prefixes in known_libs.items():
                if any(dep.lower().startswith(p) or p in dep.lower() for p in prefixes):
                    grouped[lib].append(dep)
                    placed = True
                    break
            if not placed:
                grouped["other"].append(dep)

        for lib, mods in sorted(grouped.items()):
            lines.append(f"**{lib.upper()}**: {', '.join(f'`{m}`' for m in mods)}")
        lines.append("")
        return lines

    def _module_breakdown_table(
        self,
        modules_data: Dict[str, Any],
        order: List[Dict[str, Any]],
        modules_state: Dict[str, Any],
    ) -> List[str]:
        lines = [
            "Modules listed in translation order (lowest depth first).",
            "",
            "| Rank | Module | File | LOC | Subs | Fns | Types | n_deps | Depth | Effort | Status |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        for fe in order:
            rank     = fe.get("rank", "?")
            depth    = fe.get("depth", 0)
            n_deps   = fe.get("n_internal_deps", 0)
            loc      = fe.get("line_count", "?")
            n_subs   = fe.get("n_subroutines", "?")
            n_fns    = fe.get("n_functions", "?")
            n_types  = fe.get("n_types", "?")
            src_file = Path(fe.get("file", "?")).name

            for mod in fe.get("modules", []):
                st = modules_state.get(mod, {}).get("status", "pending")
                icon = {"success": "✓", "failed": "✗", "pending": "…"}.get(st, "?")

                # Effort from units data
                effort = self._module_effort(mod)

                lines.append(
                    f"| {rank} | `{mod}` | `{src_file}` | {loc} | {n_subs} | {n_fns} | "
                    f"{n_types} | {n_deps} | {depth} | {effort} | {icon} {st} |"
                )

        lines.append("")
        return lines

    def _module_details(
        self, modules_data: Dict[str, Any], order: List[Dict[str, Any]]
    ) -> List[str]:
        """One subsection per module listing its subroutines, functions, types, and USE list."""
        lines: List[str] = []
        ordered_modules = [m for fe in order for m in fe.get("modules", [])]

        for mod_name in ordered_modules:
            mod = modules_data.get(mod_name)
            if not mod:
                continue

            file_name = Path(mod.get("file_path", "?")).name
            loc       = mod.get("line_count", "?")
            subs      = mod.get("subroutines", [])
            fns       = mod.get("functions", [])
            types     = mod.get("types", [])
            uses      = mod.get("uses", [])

            lines += [f"### `{mod_name}`", ""]
            lines.append(f"**File:** `{file_name}` · **LOC:** {loc}")

            if uses:
                use_names = [u.get("module", str(u)) if isinstance(u, dict) else str(u) for u in uses]
                lines.append(f"**Uses:** {', '.join(f'`{u}`' for u in use_names)}")
            lines.append("")

            # Subroutines
            if subs:
                lines.append(f"**Subroutines** ({len(subs)}):")
                # Try to get line info from entities
                entity_map = self._entity_map(mod)
                for s in subs:
                    loc_info = ""
                    e = entity_map.get(s.lower())
                    if e:
                        loc_info = f" (lines {e.get('line_start','?')}–{e.get('line_end','?')})"
                    lines.append(f"- `{s}`{loc_info}")
                lines.append("")

            # Functions
            if fns:
                lines.append(f"**Functions** ({len(fns)}): " +
                              ", ".join(f"`{f}`" for f in fns))
                lines.append("")

            # Types
            if types:
                lines.append(f"**Derived types** ({len(types)}): " +
                              ", ".join(f"`{t}`" for t in types))
                lines.append("")

        return lines

    def _translation_units_section(
        self, units: Dict[str, Any], trans_stats: Dict[str, Any]
    ) -> List[str]:
        units_list = units.get("translation_units", [])
        by_effort  = trans_stats.get("units_by_effort", {})
        by_type    = trans_stats.get("units_by_type", {})
        by_priority = trans_stats.get("units_by_priority", {})

        lines = [
            "The analyzer decomposes each Fortran module into translation units — "
            "logical chunks sized for a single LLM call (max 600 lines each).",
            "",
            "| Type | Count | Description |",
            "|---|---|---|",
            f"| `module`         | {by_type.get('module', 0):>4} | Module declaration block |",
            f"| `root`           | {by_type.get('root', 0):>4} | Complete subroutine/function |",
            f"| `inner`          | {by_type.get('inner', 0):>4} | Chunk of a large subroutine |",
            f"| `interface`      | {by_type.get('interface', 0):>4} | Interface block |",
            f"| `type_definition`| {by_type.get('type_definition', 0):>4} | Derived type definition |",
            "",
            "**Priority order** (1 = first to translate):",
            "",
            "| Priority | Count | What |",
            "|---|---|---|",
            f"| 1 | {by_priority.get('1', by_priority.get(1, 0))} | Module declarations |",
            f"| 2 | {by_priority.get('2', by_priority.get(2, 0))} | Type definitions |",
            f"| 3 | {by_priority.get('3', by_priority.get(3, 0))} | Interface blocks |",
            f"| 4 | {by_priority.get('4', by_priority.get(4, 0))} | Simple procedures (no child units) |",
            f"| 5 | {by_priority.get('5', by_priority.get(5, 0))} | Default procedures |",
            f"| 6 | {by_priority.get('6', by_priority.get(6, 0))} | Complex procedure roots |",
            f"| 7 | {by_priority.get('7', by_priority.get(7, 0))} | Inner/chunk units of large procedures |",
            "",
        ]

        # High-complexity units table
        high_effort = [
            u for u in units_list
            if u.get("estimated_effort") == "high"
        ]
        high_effort.sort(key=lambda u: u.get("complexity_score", 0), reverse=True)

        if high_effort:
            lines += [
                f"**High-effort units** ({len(high_effort)} units — require most repair iterations):",
                "",
                "| Unit | Module | Lines | Complexity | Has Interfaces | Has Types |",
                "|---|---|---|---|---|---|",
            ]
            for u in high_effort[:20]:
                uid   = u.get("id", "?")
                mod   = u.get("module_name", "?")
                lc    = u.get("line_count", "?")
                score = f"{u.get('complexity_score', 0):.1f}"
                hi    = "✓" if u.get("has_interfaces") else "—"
                ht    = "✓" if u.get("has_types") else "—"
                lines.append(f"| `{uid}` | `{mod}` | {lc} | {score} | {hi} | {ht} |")
            lines.append("")

        return lines

    def _implementation_order(
        self,
        order: List[Dict[str, Any]],
        dep_analysis: Dict[str, Any],
        modules_state: Dict[str, Any],
    ) -> List[str]:
        """Depth-group layout showing parallelizable batches."""
        depth_to_files: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for fe in order:
            depth_to_files[fe.get("depth", 0)].append(fe)

        lines = [
            "Files at the **same depth** have no dependency on each other and can be "
            "translated in parallel (e.g. as separate Slurm array tasks).",
            "",
        ]

        for depth in sorted(depth_to_files):
            group = depth_to_files[depth]
            n     = len(group)
            label = "no dependencies" if depth == 0 else f"depends on depth {depth - 1}"
            lines += [f"### Depth {depth} — {n} file(s) ({label})", ""]
            lines += [
                "| Rank | File | Module(s) | LOC | Subs | Internal deps | Status |",
                "|---|---|---|---|---|---|---|",
            ]
            for fe in group:
                rank    = fe.get("rank", "?")
                fname   = Path(fe.get("file", "?")).name
                mods    = fe.get("modules", [])
                mods_str = ", ".join(f"`{m}`" for m in mods)
                loc     = fe.get("line_count", "?")
                n_subs  = fe.get("n_subroutines", "?")
                n_deps  = fe.get("n_internal_deps", 0)
                circ    = " ⚠" if fe.get("circular_dep_involved") else ""

                # Combine status for all modules in the file
                statuses = []
                for m in mods:
                    st = modules_state.get(m, {}).get("status", "pending")
                    statuses.append(st)
                if all(s == "success" for s in statuses):
                    st_str = "✓ done"
                elif any(s == "failed" for s in statuses):
                    st_str = "✗ failed"
                else:
                    st_str = "… pending"

                lines.append(
                    f"| {rank} | `{fname}`{circ} | {mods_str} | {loc} | {n_subs} | {n_deps} | {st_str} |"
                )
            lines.append("")

        # Betweenness centrality — most critical modules
        centrality = dep_analysis.get("metrics", {})
        bc = {}
        if not centrality:
            # Try nested path
            from_analysis = self._load_analysis().get("dependencies", {}).get("metrics", {})
            bc = from_analysis.get("module_graph", {}).get("betweenness_centrality", {})
        else:
            bc = centrality.get("module_graph", {}).get("betweenness_centrality", {})

        if bc:
            top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_bc and top_bc[0][1] > 0:
                lines += [
                    "**Most critical modules** (highest betweenness centrality — "
                    "failures here block the most downstream modules):",
                    "",
                ]
                for mod, score in top_bc:
                    lines.append(f"- `{mod}` (score: {score:.4f})")
                lines.append("")

        return lines

    def _risk_areas(
        self,
        dep_analysis: Dict[str, Any],
        units: Dict[str, Any],
        order_summary: Dict[str, Any],
    ) -> List[str]:
        """Generate a risk table derived from actual analyzer data."""
        risks: List[Tuple[str, str, str]] = []

        # Cycle risk
        cycles = dep_analysis.get("circular_dependencies", [])
        if cycles:
            cycle_mods = ", ".join(f"`{c[0]}`" for c in cycles[:3] if c)
            risks.append((
                "Circular dependencies",
                "High",
                f"{len(cycles)} cycle(s) detected ({cycle_mods} …). "
                "Pipeline uses heuristic cycle-breaking — verify parity carefully.",
            ))

        # High-effort units
        units_list = units.get("translation_units", [])
        high_count = sum(1 for u in units_list if u.get("estimated_effort") == "high")
        if high_count > 0:
            top_complex = max(units_list, key=lambda u: u.get("complexity_score", 0), default={})
            risks.append((
                "High-complexity units",
                "Medium",
                f"{high_count} high-effort translation unit(s). "
                f"Largest: `{top_complex.get('id', '?')}` "
                f"(score={top_complex.get('complexity_score', 0):.1f}). "
                "Expect multiple RALPH iterations.",
            ))

        # Non-DAG
        if not order_summary.get("is_dag", True):
            risks.append((
                "Non-DAG dependency graph",
                "High",
                "Fortran source has circular USE dependencies. "
                "Translation order is approximate. Validate integration tests extra carefully.",
            ))

        # Generic Fortran→JAX risks (always present)
        risks += [
            (
                "INTENT(INOUT) arrays",
                "Medium",
                "Fortran subroutines modify arguments in-place. "
                "Must map to explicit return values. Verify with golden data.",
            ),
            (
                "1-indexed loops",
                "Low",
                "Fortran arrays are 1-indexed. All loop bounds must shift to 0-indexed in JAX. "
                "Boundary test cases in golden data catch these.",
            ),
            (
                "`REAL(r8)` / `float64`",
                "Medium",
                "Fortran `REAL(r8)` is 64-bit. JAX defaults to 32-bit. "
                "Must enable `jax.config.update('jax_enable_x64', True)` globally.",
            ),
            (
                "`WHERE` / masked arrays",
                "Medium",
                "Fortran `WHERE` constructs map to `jnp.where`. "
                "Test NaN propagation and boundary masking in golden cases.",
            ),
            (
                "Module-level `SAVE` state",
                "High",
                "Fortran `SAVE` variables persist across calls (implicit global state). "
                "Must be exposed as explicit function arguments in JAX. "
                "Identify all `SAVE` variables during translation.",
            ),
            (
                "Compiler-specific intrinsics",
                "Low",
                "Some Fortran intrinsics (`EOSHIFT`, `CSHIFT`, `SPREAD`) have no direct JAX equivalent. "
                "Verify with golden data across multiple physically distinct cases.",
            ),
        ]

        lines = [
            "| Risk | Severity | Mitigation |",
            "|---|---|---|",
        ]
        for risk, severity, mitigation in risks:
            sev_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(severity, severity)
            lines.append(f"| {risk} | {sev_icon} {severity} | {mitigation} |")
        lines.append("")
        return lines

    # ──────────────────────────────────────────────────────────────────
    # CLAUDE.md
    # ──────────────────────────────────────────────────────────────────

    def generate_claude_md(self) -> str:
        analysis = self._load_analysis()
        order    = self._load_order()
        state    = self._load_state()

        modules_state = {e["module"]: e for e in state.get("ordered_modules", [])}
        n_total = len([m for fe in order for m in fe.get("modules", [])])
        n_done  = sum(1 for e in modules_state.values() if e.get("status") == "success")
        pct     = int(100 * n_done / n_total) if n_total else 0
        dep_analysis = analysis.get("dependencies", {}).get("analysis", {})

        lines: List[str] = []

        lines += [
            f"# {self.gcm_model_name} Translation — Agent Guide",
            "",
            f"> **Generated**: {_now()}  ",
            f"> **Status**: {n_done}/{n_total} modules ({pct}%) translated and parity-verified",
            "",
        ]

        # Quick reference
        lines += _section("Quick Reference") + [
            "| Resource | Path |",
            "|---|---|",
            "| Architecture design | `docs/DESIGN.md` |",
            "| Translation changelog | `CHANGELOG.md` |",
            "| Pipeline state | `pipeline_state.json` |",
            "| Translation state | `translation_state.json` |",
            f"| Fortran source | `{self.fortran_dir}` |",
            "| JAX modules | `jax/src/` |",
            "| Golden I/O | `ftest/*/tests/golden/` |",
            "| Parity reports | `parity/*/parity_report.json` |",
            "| RALPH logs | `reports/*_ralph_log.json` |",
            "| Slurm scripts | `slurm_scripts/*.sbatch` |",
            "",
        ]

        # HPC environment
        lines += _section("HPC Environment Rules") + [
            "- **Always use `--use-tmux`** on HPC nodes — claude CLI runs inside a named tmux session",
            "  and survives SSH disconnects.",
            "- **One Slurm job per module**: `transjax slurm-submit --dry-run` generates sbatch scripts.",
            "  `tmux attach -t transjax-{module}` to inspect a running job.",
            "- **Do not run `transjax pipeline` without `--modules`** on a login node.",
            "- Modules at the same dependency depth are independent and can run as a Slurm array.",
            "- After any crash: re-run the same command — state is checkpointed per module.",
            "",
        ]

        # Session orientation
        lines += _section("Starting a New Session") + [
            "1. Read this file, then `docs/DESIGN.md`.",
            "2. Check progress: `transjax status --output-dir .`",
            "3. Inspect the next module: `transjax convert --next`",
            "4. Read the module's subroutine list in `docs/DESIGN.md` **before** translating.",
            "5. After translation, check `{module}_ralph_log.json` for repair history.",
            "",
        ]

        # Hub modules warning
        hub_modules = dep_analysis.get("hub_modules", [])
        if hub_modules:
            lines += _section("Critical Hub Modules") + [
                "These modules are depended on by many others. "
                "Parity failures here cascade to all downstream modules.",
                "",
            ]
            for hub in hub_modules[:10]:
                lines.append(f"- `{hub}`")
            lines.append("")

        # Module dependency order
        lines += _section("Module Dependency Order")
        lines += self._claude_module_table(order, modules_state)

        # Agent teams
        lines += _section("Agent Teams") + [
            "| Agent | Role |",
            "|---|---|",
            "| `TranslatorAgent` | Fortran → JAX translation |",
            "| `RepairAgent` | Unit test failure repair (RALPH) |",
            "| `ParityAgent` | Numerical parity tests |",
            "| `ParityRepairAgent` | Parity failure repair (RALPH) |",
            "| `FtestAgent` | Fortran driver generation |",
            "| `GoldenAgent` | Golden I/O capture |",
            "| `IntegratorAgent` | System integration (`model_run.py`) |",
            "",
            "Per-module execution order:",
            "```",
            "FtestAgent → GoldenAgent → TranslatorAgent → ParityRepairAgent",
            "```",
            "",
        ]

        # Development principles
        lines += _section("Development Principles") + [
            "1. **Fortran is oracle** — if JAX disagrees with Fortran, JAX is wrong.",
            "2. **Tests are everything** — never approve a translation without parity.",
            "3. **RALPH discipline** — read `{module}_ralph_log.json` before starting repair.",
            "4. **Keep CHANGELOG.md current** — every successful module commit adds an entry.",
            "5. **One commit per module** — small, auditable, reversible.",
            "6. **Never weaken tolerances** — fix the translation instead.",
            "7. **Context window hygiene** — `pytest -q --tb=short` keeps output concise.",
            "8. **Specialized roles** — use the right agent for the right task.",
            "9. **Document for the next session** — update this file after major changes.",
            "",
        ]

        # Coding conventions
        lines += _section("Coding Conventions") + [
            "- **Pure functions** — no global state; all inputs as explicit arguments.",
            "- **Immutable outputs** — return new arrays/NamedTuples; never modify inputs.",
            "- **`jnp.float64`** — always; `jax.config.update('jax_enable_x64', True)` at module top.",
            "- **JIT-compatible control flow** — use `jnp.where` not `if arr[i] > 0`.",
            "- **NamedTuples for state** — `BoundsState = NamedTuple('BoundsState', [...])`.",
            "- **Type hints** — all public functions; `jax.Array` for array arguments.",
            "- **Module header** — comment block with: Fortran source path, subroutine, author, date.",
            "- **`__all__`** — export all public functions explicitly.",
            "",
        ]

        # Critical rules
        lines += _section("Critical Rules") + [
            "- **Never add fudge factors** to pass parity — fix the physics.",
            "- **Never change function signatures** during repair without updating all callers.",
            "- **Test at multiple golden cases** — edge cases catch dtype and indexing bugs.",
            "- **Run the full parity suite** after every repair — not just the failing test.",
            "- **Never commit a FAIL module** — all parity tests must pass before committing.",
            f"- **Tolerances are fixed**: `rtol={self.parity_rtol}`, `atol={self.parity_atol}` — non-negotiable.",
            "- **SAVE variables must be explicit** — identify all Fortran `SAVE` vars before translating.",
            "",
        ]

        return "\n".join(lines)

    def _claude_module_table(
        self,
        order: List[Dict[str, Any]],
        modules_state: Dict[str, Any],
    ) -> List[str]:
        lines = [
            "Translate lower-depth modules first. Same-depth modules are independent.",
            "",
            "| # | Module | Depth | LOC | Subs | Effort | Status |",
            "|---|---|---|---|---|---|---|",
        ]
        i = 1
        for fe in order:
            depth  = fe.get("depth", 0)
            loc    = fe.get("line_count", "?")
            n_subs = fe.get("n_subroutines", "?")
            for mod in fe.get("modules", []):
                st    = modules_state.get(mod, {}).get("status", "pending")
                icon  = {"success": "✓", "failed": "✗", "pending": "…"}.get(st, "?")
                effort = self._module_effort(mod)
                lines.append(
                    f"| {i} | `{mod}` | {depth} | {loc} | {n_subs} | {effort} | {icon} {st} |"
                )
                i += 1
        lines.append("")
        return lines

    # ──────────────────────────────────────────────────────────────────
    # CHANGELOG.md stub
    # ──────────────────────────────────────────────────────────────────

    def generate_changelog(self) -> str:
        return (
            f"# {self.gcm_model_name} Translation Changelog\n\n"
            f"> Started {_now()}\n\n"
            "## Modules\n\n"
            "<!-- Entries prepended automatically by GitCoordinator -->\n"
        )

    # ──────────────────────────────────────────────────────────────────
    # Data loaders (lazy + cached)
    # ──────────────────────────────────────────────────────────────────

    def _load_analysis(self) -> Dict[str, Any]:
        if self._analysis is None:
            path = self.analysis_dir / "analysis_results.json"
            self._analysis = _load_json(path)
        return self._analysis  # type: ignore[return-value]

    def _load_order(self) -> List[Dict[str, Any]]:
        if self._order is None:
            path = self.analysis_dir / "translation_order.json"
            data = _load_json(path)
            self._order = data.get("files", [])
        return self._order  # type: ignore[return-value]

    def _load_order_summary(self) -> Dict[str, Any]:
        path = self.analysis_dir / "translation_order.json"
        data = _load_json(path)
        return data.get("summary", {})

    def _load_units(self) -> Dict[str, Any]:
        if self._units_data is None:
            path = self.analysis_dir / "translation_units.json"
            self._units_data = _load_json(path)
        return self._units_data  # type: ignore[return-value]

    def _load_state(self) -> Dict[str, Any]:
        if self._state is None:
            path = self.output_dir / "translation_state.json"
            self._state = _load_json(path)
        return self._state  # type: ignore[return-value]

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _module_effort(self, module_name: str) -> str:
        """Return the highest effort level of any unit in this module."""
        units = self._load_units().get("translation_units", [])
        efforts = [
            u.get("estimated_effort", "medium")
            for u in units
            if u.get("module_name", "").lower() == module_name.lower()
        ]
        if not efforts:
            return "—"
        order = {"high": 2, "medium": 1, "low": 0}
        return max(efforts, key=lambda e: order.get(e, 1))

    def _entity_map(self, mod: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Build name → entity dict from a module's entities list."""
        result: Dict[str, Dict[str, Any]] = {}
        for e in mod.get("entities", []):
            name = (e.get("name") or "").lower()
            if name:
                result[name] = e
        return result


# ──────────────────────────────────────────────────────────────────────
# Module-level utilities
# ──────────────────────────────────────────────────────────────────────

def _section(title: str) -> List[str]:
    return [f"## {title}", ""]


def _load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse %s: %s", path, exc)
    return {}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _git_head_hash(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"
