"""
Translation unit decomposer for Fortran projects.
Breaks down large Fortran modules into manageable translation units for easier porting.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ..parser.fortran_parser import ModuleInfo, FortranEntity
from ..config.project_config import FortranProjectConfig

logger = logging.getLogger(__name__)


@dataclass
class TranslationUnit:
    """Represents a translation unit for code conversion."""

    id: str
    unit_type: str  # 'root', 'inner', 'interface', 'type_definition'
    entity_name: str
    entity_type: str  # 'subroutine', 'function', 'type', 'module'
    module_name: str
    file_path: str
    line_start: int
    line_end: int
    line_count: int

    # Hierarchical structure
    parent_id: Optional[str] = None
    child_ids: Optional[List[str]] = None

    # Dependencies
    depends_on: Optional[List[str]] = None
    used_by: Optional[List[str]] = None

    # Content metadata
    has_interfaces: bool = False
    has_types: bool = False
    complexity_score: float = 0.0

    # Translation metadata
    priority: int = 0
    estimated_effort: str = "medium"  # low, medium, high
    notes: Optional[List[str]] = None

    def __post_init__(self):
        if self.child_ids is None:
            self.child_ids = []
        if self.depends_on is None:
            self.depends_on = []
        if self.used_by is None:
            self.used_by = []
        if self.notes is None:
            self.notes = []


class TranslationUnitDecomposer:
    """Decomposes Fortran modules into manageable translation units."""

    def __init__(self, config: FortranProjectConfig):
        self.config = config
        self.translation_units: List[TranslationUnit] = []
        self.unit_counter = 0

    def decompose_modules(
        self, modules: Dict[str, ModuleInfo]
    ) -> List[TranslationUnit]:
        """Decompose all modules into translation units."""
        logger.info("Decomposing modules into translation units")

        self.translation_units = []
        self.unit_counter = 0

        for module_name, module_info in modules.items():
            logger.debug(f"Decomposing module: {module_name}")
            module_units = self._decompose_module(module_info)
            self.translation_units.extend(module_units)

        # Analyze dependencies between units
        self._analyze_unit_dependencies()

        # Calculate complexity scores
        self._calculate_complexity_scores()

        # Assign priorities
        self._assign_priorities()

        logger.info(f"Created {len(self.translation_units)} translation units")
        return self.translation_units

    def _decompose_module(self, module_info: ModuleInfo) -> List[TranslationUnit]:
        """Decompose a single module into translation units."""
        units = []

        # Read the source file
        try:
            with open(
                module_info.file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read {module_info.file_path}: {e}")
            return []

        # Create module-level unit for module declaration and module variables
        module_unit = self._create_module_unit(module_info, lines)
        if module_unit:
            units.append(module_unit)

        # Decompose interfaces
        interface_units = self._decompose_interfaces(module_info, lines)
        units.extend(interface_units)

        # Decompose type definitions
        type_units = self._decompose_types(module_info, lines)
        units.extend(type_units)

        # Decompose procedures (subroutines and functions)
        procedure_units = self._decompose_procedures(module_info, lines)
        units.extend(procedure_units)

        return units

    def _create_module_unit(
        self, module_info: ModuleInfo, lines: List[str]
    ) -> Optional[TranslationUnit]:
        """Create a translation unit for module-level declarations."""
        module_start, module_end = self._find_module_bounds(lines, module_info.name)

        if module_start == -1:
            return None

        # Find where the actual procedures start
        procedure_start = self._find_first_procedure_start(lines, module_start)

        if procedure_start == -1:
            # No procedures, entire module is one unit
            unit_end = module_end
        else:
            # Module unit ends before first procedure
            unit_end = procedure_start - 1

        unit_id = self._generate_unit_id(module_info.name, "module")

        return TranslationUnit(
            id=unit_id,
            unit_type="module",
            entity_name=module_info.name,
            entity_type="module",
            module_name=module_info.name,
            file_path=module_info.file_path,
            line_start=module_start,
            line_end=unit_end,
            line_count=unit_end - module_start + 1,
            has_interfaces=len(module_info.interfaces) > 0,
            has_types=len(module_info.types) > 0,
        )

    def _decompose_interfaces(
        self, module_info: ModuleInfo, lines: List[str]
    ) -> List[TranslationUnit]:
        """Decompose interface blocks into translation units."""
        units = []

        # Find interface blocks using regex
        content = "".join(lines)
        interface_pattern = r"^\s*interface\s+(\w+)?\s*$.*?^\s*end\s+interface"

        for match in re.finditer(
            interface_pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE
        ):
            interface_name = match.group(1) if match.group(1) else "unnamed_interface"
            start_pos = match.start()
            end_pos = match.end()

            # Convert character positions to line numbers
            start_line = content[:start_pos].count("\n") + 1
            end_line = content[:end_pos].count("\n") + 1

            unit_id = self._generate_unit_id(
                module_info.name, f"interface_{interface_name}"
            )

            unit = TranslationUnit(
                id=unit_id,
                unit_type="interface",
                entity_name=interface_name,
                entity_type="interface",
                module_name=module_info.name,
                file_path=module_info.file_path,
                line_start=start_line,
                line_end=end_line,
                line_count=end_line - start_line + 1,
                has_interfaces=True,
            )
            units.append(unit)

        return units

    def _decompose_types(
        self, module_info: ModuleInfo, lines: List[str]
    ) -> List[TranslationUnit]:
        """Decompose type definitions into translation units."""
        units = []

        for entity in module_info.entities:
            if entity.entity_type == "type" and entity.line_start > 0:
                unit_id = self._generate_unit_id(
                    module_info.name, f"type_{entity.name}"
                )

                unit = TranslationUnit(
                    id=unit_id,
                    unit_type="type_definition",
                    entity_name=entity.name,
                    entity_type="type",
                    module_name=module_info.name,
                    file_path=module_info.file_path,
                    line_start=entity.line_start,
                    line_end=entity.line_end,
                    line_count=(
                        entity.line_end - entity.line_start + 1
                        if entity.line_end > 0
                        else 0
                    ),
                    has_types=True,
                )
                units.append(unit)

        return units

    def _decompose_procedures(
        self, module_info: ModuleInfo, lines: List[str]
    ) -> List[TranslationUnit]:
        """Decompose procedures (subroutines and functions) into translation units."""
        units = []

        for entity in module_info.entities:
            if (
                entity.entity_type in ["subroutine", "function"]
                and entity.line_start > 0
            ):
                procedure_units = self._decompose_single_procedure(
                    entity, module_info, lines
                )
                units.extend(procedure_units)

        return units

    def _decompose_single_procedure(
        self, entity: FortranEntity, module_info: ModuleInfo, lines: List[str]
    ) -> List[TranslationUnit]:
        """Decompose a single procedure into one or more translation units."""
        units: List[TranslationUnit] = []

        if entity.line_end <= 0:
            return units

        line_count = entity.line_end - entity.line_start + 1

        if line_count <= self.config.max_translation_unit_lines:
            # Small procedure - single translation unit
            unit_id = self._generate_unit_id(
                module_info.name, f"{entity.entity_type}_{entity.name}"
            )

            unit = TranslationUnit(
                id=unit_id,
                unit_type="root",
                entity_name=entity.name,
                entity_type=entity.entity_type,
                module_name=module_info.name,
                file_path=module_info.file_path,
                line_start=entity.line_start,
                line_end=entity.line_end,
                line_count=line_count,
            )
            units.append(unit)
        else:
            # Large procedure - decompose into multiple units
            procedure_lines = lines[entity.line_start - 1 : entity.line_end]
            chunks = self._split_procedure_into_chunks(
                procedure_lines, entity, module_info
            )

            # Create root unit for procedure signature and declarations
            root_unit_id = self._generate_unit_id(
                module_info.name, f"{entity.entity_type}_{entity.name}_root"
            )

            # Find where executable statements start
            declaration_end = self._find_declaration_end(procedure_lines)
            root_end_line = entity.line_start + min(declaration_end + 10, 20) - 1

            root_unit = TranslationUnit(
                id=root_unit_id,
                unit_type="root",
                entity_name=entity.name,
                entity_type=entity.entity_type,
                module_name=module_info.name,
                file_path=module_info.file_path,
                line_start=entity.line_start,
                line_end=root_end_line,
                line_count=root_end_line - entity.line_start + 1,
            )
            units.append(root_unit)

            # Create inner units for chunks
            for i, (chunk_start, chunk_end, description) in enumerate(chunks):
                inner_unit_id = self._generate_unit_id(
                    module_info.name, f"{entity.entity_type}_{entity.name}_inner_{i+1}"
                )

                inner_unit = TranslationUnit(
                    id=inner_unit_id,
                    unit_type="inner",
                    entity_name=f"{entity.name}_part_{i+1}",
                    entity_type=entity.entity_type,
                    module_name=module_info.name,
                    file_path=module_info.file_path,
                    line_start=chunk_start,
                    line_end=chunk_end,
                    line_count=chunk_end - chunk_start + 1,
                    parent_id=root_unit_id,
                    notes=[description],
                )
                if root_unit.child_ids is not None:
                    root_unit.child_ids.append(inner_unit_id)
                units.append(inner_unit)

        return units

    def _split_procedure_into_chunks(
        self, procedure_lines: List[str], entity: FortranEntity, module_info: ModuleInfo
    ) -> List[Tuple[int, int, str]]:
        """Split a large procedure into logical chunks."""
        chunks: List[Tuple[int, int, str]] = []

        # Find declaration section end
        declaration_end = self._find_declaration_end(procedure_lines)

        # Start chunking after declarations
        start_line = entity.line_start + declaration_end
        current_start = start_line
        current_size = 0

        for i in range(declaration_end, len(procedure_lines)):
            line = procedure_lines[i].strip()
            current_size += 1
            actual_line_num = entity.line_start + i

            # Check if we should end current chunk
            should_end_chunk = (
                current_size >= self.config.max_translation_unit_lines
                or (
                    current_size >= self.config.min_chunk_lines
                    and self._is_logical_boundary(line)
                )
            )

            if should_end_chunk:
                description = f"Code block {len(chunks) + 1}"
                chunks.append((current_start, actual_line_num, description))
                current_start = actual_line_num + 1
                current_size = 0

        # Add remaining lines
        if current_size > 0:
            description = f"Code block {len(chunks) + 1} (final)"
            chunks.append((current_start, entity.line_end, description))

        return chunks

    def _find_declaration_end(self, procedure_lines: List[str]) -> int:
        """Find where declarations end and executable statements begin."""
        for i, line in enumerate(procedure_lines):
            line_clean = line.strip().lower()

            # Skip empty lines and comments
            if not line_clean or line_clean.startswith("!"):
                continue

            # Check for executable statements
            executable_keywords = [
                "call",
                "if",
                "do",
                "select",
                "where",
                "forall",
                "go to",
                "goto",
                "return",
                "stop",
                "cycle",
                "exit",
            ]

            for keyword in executable_keywords:
                if line_clean.startswith(keyword):
                    return i

            # Check for assignment statements (contains '=')
            if "=" in line_clean and not any(
                decl in line_clean
                for decl in ["integer", "real", "character", "logical", "type"]
            ):
                return i

        # If no executable statements found, assume entire procedure is declarations
        return len(procedure_lines)

    def _is_logical_boundary(self, line: str) -> bool:
        """Check if a line represents a good place to split code."""
        line_clean = line.strip().lower()

        boundary_patterns = [
            r"^\s*end\s*if",
            r"^\s*end\s*do",
            r"^\s*end\s*select",
            r"^\s*else",
            r"^\s*elseif",
            r"^\s*case\s*\(",
            r"^\s*!\s*[-=]{3,}",  # Comment separators
        ]

        for pattern in boundary_patterns:
            if re.match(pattern, line):
                return True

        return False

    def _find_module_bounds(
        self, lines: List[str], module_name: str
    ) -> Tuple[int, int]:
        """Find start and end lines of a module."""
        start_line = -1
        end_line = -1

        module_pattern = rf"^\s*module\s+{re.escape(module_name)}\s*$"
        end_pattern = rf"^\s*end\s+module(\s+{re.escape(module_name)})?\s*$"

        for i, line in enumerate(lines):
            if start_line == -1 and re.search(module_pattern, line, re.IGNORECASE):
                start_line = i + 1  # 1-indexed
            elif start_line != -1 and re.search(end_pattern, line, re.IGNORECASE):
                end_line = i + 1  # 1-indexed
                break

        return start_line, end_line

    def _find_first_procedure_start(self, lines: List[str], module_start: int) -> int:
        """Find the line number of the first procedure in a module."""
        for i in range(module_start, len(lines)):
            line = lines[i].strip().lower()
            if line.startswith("subroutine ") or line.startswith("function "):
                return i + 1  # 1-indexed
        return -1

    def _generate_unit_id(self, module_name: str, suffix: str) -> str:
        """Generate a unique ID for a translation unit."""
        self.unit_counter += 1
        return f"{module_name.lower()}_{suffix}_{self.unit_counter:03d}"

    def _analyze_unit_dependencies(self) -> None:
        """Analyze dependencies between translation units."""
        # This is a simplified implementation
        # A more sophisticated version would analyze actual code dependencies

        module_dependencies: Dict[str, List[TranslationUnit]] = {}

        # Group units by module
        for unit in self.translation_units:
            if unit.module_name not in module_dependencies:
                module_dependencies[unit.module_name] = []
            module_dependencies[unit.module_name].append(unit)

        # Set up basic hierarchical dependencies
        for unit in self.translation_units:
            if unit.parent_id:
                parent = next(
                    (u for u in self.translation_units if u.id == unit.parent_id), None
                )
                if parent:
                    if unit.depends_on is not None:
                        unit.depends_on.append(parent.id)
                    if parent.used_by is not None:
                        parent.used_by.append(unit.id)

    def _calculate_complexity_scores(self) -> None:
        """Calculate complexity scores for translation units."""
        for unit in self.translation_units:
            score = 0.0

            # Base score on line count
            score += unit.line_count * 0.1

            # Higher score for units with interfaces or types
            if unit.has_interfaces:
                score += 10.0
            if unit.has_types:
                score += 5.0

            # Higher score for root units of large procedures
            if unit.unit_type == "root" and unit.child_ids:
                score += len(unit.child_ids) * 2.0

            # Score based on entity type
            type_multipliers = {
                "module": 1.0,
                "interface": 2.0,
                "type": 1.5,
                "subroutine": 1.2,
                "function": 1.2,
            }
            score *= type_multipliers.get(unit.entity_type, 1.0)

            unit.complexity_score = score

    def _assign_priorities(self) -> None:
        """Assign translation priorities to units."""
        # Sort by complexity (simpler units first)
        sorted_units = sorted(self.translation_units, key=lambda u: u.complexity_score)

        for i, unit in enumerate(sorted_units):
            if unit.unit_type == "module":
                unit.priority = 1  # Module declarations first
            elif unit.unit_type == "type_definition":
                unit.priority = 2  # Type definitions next
            elif unit.unit_type == "interface":
                unit.priority = 3  # Interfaces next
            elif unit.unit_type == "root" and not unit.child_ids:
                unit.priority = 4  # Simple procedures
            elif unit.unit_type == "root" and unit.child_ids:
                unit.priority = 6  # Complex procedure roots
            elif unit.unit_type == "inner":
                unit.priority = 7  # Inner units last
            else:
                unit.priority = 5  # Default

            # Assign effort estimates
            if unit.complexity_score < 5:
                unit.estimated_effort = "low"
            elif unit.complexity_score < 15:
                unit.estimated_effort = "medium"
            else:
                unit.estimated_effort = "high"

    def get_units_by_priority(self) -> List[TranslationUnit]:
        """Get translation units sorted by priority."""
        return sorted(
            self.translation_units, key=lambda u: (u.priority, u.complexity_score)
        )

    def get_units_by_module(self, module_name: str) -> List[TranslationUnit]:
        """Get all translation units for a specific module."""
        return [u for u in self.translation_units if u.module_name == module_name]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the translation units."""
        stats: Dict[str, Any] = {
            "total_units": len(self.translation_units),
            "units_by_type": {},
            "units_by_priority": {},
            "units_by_effort": {},
            "average_lines_per_unit": 0,
            "total_lines": 0,
            "largest_unit": None,
            "most_complex_unit": None,
        }

        if not self.translation_units:
            return stats

        # Count by type
        for unit in self.translation_units:
            stats["units_by_type"][unit.unit_type] = (
                stats["units_by_type"].get(unit.unit_type, 0) + 1
            )
            stats["units_by_priority"][unit.priority] = (
                stats["units_by_priority"].get(unit.priority, 0) + 1
            )
            stats["units_by_effort"][unit.estimated_effort] = (
                stats["units_by_effort"].get(unit.estimated_effort, 0) + 1
            )
            stats["total_lines"] += unit.line_count

        stats["average_lines_per_unit"] = stats["total_lines"] / len(
            self.translation_units
        )

        # Find largest and most complex units
        stats["largest_unit"] = max(
            self.translation_units, key=lambda u: u.line_count
        ).id
        stats["most_complex_unit"] = max(
            self.translation_units, key=lambda u: u.complexity_score
        ).id

        return stats

    def export_units(self, output_path: Path, format: str = "json") -> None:
        """Export translation units to file."""
        output_path = Path(output_path)

        # Convert to serializable format
        units_data = []
        for unit in self.translation_units:
            unit_dict = unit.__dict__.copy()
            units_data.append(unit_dict)

        export_data = {
            "translation_units": units_data,
            "statistics": self.get_statistics(),
            "config": {
                "max_translation_unit_lines": self.config.max_translation_unit_lines,
                "min_chunk_lines": self.config.min_chunk_lines,
            },
        }

        if format.lower() == "json":
            import json

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == "yaml":
            import yaml

            with open(output_path, "w") as f:
                yaml.dump(export_data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(
            f"Exported {len(self.translation_units)} translation units to {output_path}"
        )
