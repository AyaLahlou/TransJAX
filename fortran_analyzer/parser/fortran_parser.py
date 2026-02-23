"""
Generic Fortran parser for analyzing any Fortran codebase.
This module provides parsing capabilities that are not tied to any specific project.
"""

import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from fparser.two.parser import ParserFactory
    from fparser.common.readfortran import FortranFileReader
    from fparser.two.Fortran2003 import (
        Module,
        Use_Stmt,
        Subroutine_Subprogram,
        Function_Subprogram,
        Derived_Type_Def,
        Subroutine_Stmt,
        Function_Stmt,
        Call_Stmt,
        Declaration_Construct,
        Variable_Declaration,
        Implicit_Stmt,
        Interface_Block,
        Abstract_Interface_Block,
    )

    FPARSER_AVAILABLE = True
except ImportError:
    FPARSER_AVAILABLE = False

from ..config.project_config import FortranProjectConfig

logger = logging.getLogger(__name__)


@dataclass
class FortranEntity:
    """Represents a Fortran entity (module, subroutine, function, type, etc.)."""

    name: str
    entity_type: str  # 'module', 'subroutine', 'function', 'type', 'variable'
    file_path: str
    line_start: int
    line_end: int
    parent: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class ModuleInfo:
    """Information about a Fortran module."""

    name: str
    file_path: str
    uses: List[Dict[str, Union[str, List[str], None]]]  # Dependencies
    subroutines: List[str]
    functions: List[str]
    types: List[str]
    variables: List[str]
    interfaces: List[str]
    line_count: int
    entities: List[FortranEntity]


class FortranParser:
    """Generic Fortran parser that can analyze any Fortran codebase."""

    def __init__(self, config: FortranProjectConfig):
        self.config = config
        self.results: Dict[str, Any] = {
            "modules": {},
            "files": {},
            "dependencies": {},
            "call_graph": {},
            "entities": [],
            "statistics": {},
        }

        if FPARSER_AVAILABLE:
            self.parser = ParserFactory().create(std=config.fortran_standard)
        else:
            logger.warning("fparser not available, using regex-based parsing")
            self.parser = None

    def find_fortran_files(self) -> List[Path]:
        """Find all Fortran files in the configured source directories."""
        fortran_files = []

        for source_dir in self.config.source_dirs:
            source_path = Path(source_dir)
            if not source_path.exists():
                logger.warning(f"Source directory does not exist: {source_dir}")
                continue

            # Use include patterns
            for pattern in self.config.include_patterns:
                files = source_path.glob(pattern)
                fortran_files.extend([f for f in files if f.is_file()])

        # Filter by extensions
        fortran_files = [
            f for f in fortran_files if f.suffix in self.config.fortran_extensions
        ]

        # Apply exclude patterns
        if self.config.exclude_patterns:
            filtered_files = []
            for file_path in fortran_files:
                exclude = False
                for exclude_pattern in self.config.exclude_patterns:
                    if file_path.match(exclude_pattern):
                        exclude = True
                        break
                if not exclude:
                    filtered_files.append(file_path)
            fortran_files = filtered_files

        # Apply exclude directories
        if self.config.exclude_dirs:
            filtered_files = []
            for file_path in fortran_files:
                exclude = False
                for exclude_dir in self.config.exclude_dirs:
                    if Path(exclude_dir) in file_path.parents:
                        exclude = True
                        break
                if not exclude:
                    filtered_files.append(file_path)
            fortran_files = filtered_files

        return sorted(set(fortran_files))

    def parse_with_fparser(self, file_path: Path) -> Optional[ModuleInfo]:
        """Parse Fortran file using fparser2."""
        if not FPARSER_AVAILABLE:
            return None

        try:
            reader = FortranFileReader(str(file_path))
            parse_tree = self.parser(reader)
            return self._extract_module_info_fparser(parse_tree, file_path)
        except Exception as e:
            logger.error(f"Failed to parse {file_path} with fparser: {e}")
            return None

    def parse_with_regex(self, file_path: Path) -> Optional[ModuleInfo]:
        """Parse Fortran file using regex patterns."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.splitlines()

            return self._extract_module_info_regex(content, lines, file_path)
        except Exception as e:
            logger.error(f"Failed to parse {file_path} with regex: {e}")
            return None

    def _extract_module_info_fparser(self, parse_tree, file_path: Path) -> ModuleInfo:
        """Extract module information using fparser AST."""
        module_name = None
        uses = []
        subroutines = []
        functions = []
        types = []
        variables: List[str] = []
        interfaces = []
        entities = []

        for node in parse_tree.walk():
            if isinstance(node, Module):
                module_name = str(node.children[1])

            elif isinstance(node, Use_Stmt):
                use_info = self._extract_use_stmt(node)
                if use_info:
                    uses.append(use_info)

            elif isinstance(node, Subroutine_Subprogram):
                sub_info = self._extract_subroutine_info_fparser(node, file_path)
                if sub_info:
                    subroutines.append(sub_info.name)
                    entities.append(sub_info)

            elif isinstance(node, Function_Subprogram):
                func_info = self._extract_function_info_fparser(node, file_path)
                if func_info:
                    functions.append(func_info.name)
                    entities.append(func_info)

            elif isinstance(node, Derived_Type_Def):
                type_info = self._extract_type_info_fparser(node, file_path)
                if type_info:
                    types.append(type_info.name)
                    entities.append(type_info)

            elif isinstance(node, Interface_Block):
                interface_name = self._extract_interface_name(node)
                if interface_name:
                    interfaces.append(interface_name)

        if not module_name:
            module_name = file_path.stem

        line_count = len(open(file_path, "r").readlines())

        return ModuleInfo(
            name=module_name,
            file_path=str(file_path),
            uses=uses,
            subroutines=subroutines,
            functions=functions,
            types=types,
            variables=variables,
            interfaces=interfaces,
            line_count=line_count,
            entities=entities,
        )

    def _extract_module_info_regex(
        self, content: str, lines: List[str], file_path: Path
    ) -> ModuleInfo:
        """Extract module information using regex patterns."""
        module_name = None
        uses = []
        subroutines = []
        functions = []
        types = []
        variables: List[str] = []
        interfaces: List[str] = []
        entities = []

        # Extract module name
        module_match = re.search(
            r"^\s*module\s+(\w+)", content, re.MULTILINE | re.IGNORECASE
        )
        if module_match:
            module_name = module_match.group(1)
        else:
            module_name = file_path.stem

        # Extract use statements
        use_pattern = r"^\s*use\s+(\w+)(?:\s*,\s*only\s*:\s*(.+?))?$"
        for match in re.finditer(use_pattern, content, re.MULTILINE | re.IGNORECASE):
            use_info = {
                "module": match.group(1),
                "only": match.group(2).split(",") if match.group(2) else None,
            }
            uses.append(use_info)

        # Extract subroutines
        sub_pattern = r"^\s*subroutine\s+(\w+)"
        for match in re.finditer(sub_pattern, content, re.MULTILINE | re.IGNORECASE):
            subroutine_name = match.group(1)
            subroutines.append(subroutine_name)

            # Find line numbers
            line_start, line_end = self._find_entity_bounds(
                lines, subroutine_name, "subroutine"
            )

            entity = FortranEntity(
                name=subroutine_name,
                entity_type="subroutine",
                file_path=str(file_path),
                line_start=line_start,
                line_end=line_end,
                parent=module_name,
            )
            entities.append(entity)

        # Extract functions
        func_pattern = r"^\s*function\s+(\w+)"
        for match in re.finditer(func_pattern, content, re.MULTILINE | re.IGNORECASE):
            function_name = match.group(1)
            functions.append(function_name)

            line_start, line_end = self._find_entity_bounds(
                lines, function_name, "function"
            )

            entity = FortranEntity(
                name=function_name,
                entity_type="function",
                file_path=str(file_path),
                line_start=line_start,
                line_end=line_end,
                parent=module_name,
            )
            entities.append(entity)

        # Extract types
        type_pattern = r"^\s*type\s*(?:::\s*)?(\w+)"
        for match in re.finditer(type_pattern, content, re.MULTILINE | re.IGNORECASE):
            type_name = match.group(1)
            if type_name.lower() not in ["public", "private"]:  # Skip access specifiers
                types.append(type_name)

                line_start, line_end = self._find_entity_bounds(
                    lines, type_name, "type"
                )

                entity = FortranEntity(
                    name=type_name,
                    entity_type="type",
                    file_path=str(file_path),
                    line_start=line_start,
                    line_end=line_end,
                    parent=module_name,
                )
                entities.append(entity)

        return ModuleInfo(
            name=module_name,
            file_path=str(file_path),
            uses=uses,
            subroutines=subroutines,
            functions=functions,
            types=types,
            variables=variables,
            interfaces=interfaces,
            line_count=len(lines),
            entities=entities,
        )

    def _find_entity_bounds(
        self, lines: List[str], entity_name: str, entity_type: str
    ) -> Tuple[int, int]:
        """Find start and end line numbers for a Fortran entity."""
        start_patterns = {
            "subroutine": rf"^\s*subroutine\s+{re.escape(entity_name)}\s*[\(\s!]",
            "function": rf"^\s*function\s+{re.escape(entity_name)}\s*[\(\s!]",
            "type": rf"^\s*type\s*(?:::\s*)?{re.escape(entity_name)}\s*$",
        }

        end_patterns = {
            "subroutine": rf"^\s*end\s+subroutine(\s+{re.escape(entity_name)})?\s*$",
            "function": rf"^\s*end\s+function(\s+{re.escape(entity_name)})?\s*$",
            "type": rf"^\s*end\s+type(\s+{re.escape(entity_name)})?\s*$",
        }

        start_pattern = start_patterns.get(entity_type)
        end_pattern = end_patterns.get(entity_type)

        if not start_pattern or not end_pattern:
            return -1, -1

        start_line = -1
        end_line = -1

        for i, line in enumerate(lines):
            if start_line == -1 and re.search(start_pattern, line, re.IGNORECASE):
                start_line = i + 1  # 1-indexed
            elif start_line != -1 and re.search(end_pattern, line, re.IGNORECASE):
                end_line = i + 1  # 1-indexed
                break

        return start_line, end_line

    def _extract_use_stmt(
        self, use_node
    ) -> Optional[Dict[str, Union[str, List[str], None]]]:
        """Extract information from a USE statement."""
        try:
            module_name = str(use_node.children[1])
            only_list: Optional[List[str]] = None

            if len(use_node.children) > 2 and use_node.children[2]:
                # Has ONLY clause
                only_list = [
                    str(item).strip()
                    for item in use_node.children[2].children[1].children
                ]

            return {"module": module_name, "only": only_list}
        except:
            return None

    def _extract_subroutine_info_fparser(
        self, sub_node, file_path: Path
    ) -> Optional[FortranEntity]:
        """Extract subroutine information from fparser AST."""
        try:
            # Find subroutine statement
            for node in sub_node.walk():
                if isinstance(node, Subroutine_Stmt):
                    name = str(node.children[1])
                    # TODO: Extract line numbers from node
                    return FortranEntity(
                        name=name,
                        entity_type="subroutine",
                        file_path=str(file_path),
                        line_start=1,  # TODO: Get actual line numbers
                        line_end=1,
                        attributes={"arguments": self._extract_arguments(node)},
                    )
        except:
            pass
        return None

    def _extract_function_info_fparser(
        self, func_node, file_path: Path
    ) -> Optional[FortranEntity]:
        """Extract function information from fparser AST."""
        try:
            # Find function statement
            for node in func_node.walk():
                if isinstance(node, Function_Stmt):
                    name = str(node.children[1])
                    return FortranEntity(
                        name=name,
                        entity_type="function",
                        file_path=str(file_path),
                        line_start=1,  # TODO: Get actual line numbers
                        line_end=1,
                        attributes={"arguments": self._extract_arguments(node)},
                    )
        except:
            pass
        return None

    def _extract_type_info_fparser(
        self, type_node, file_path: Path
    ) -> Optional[FortranEntity]:
        """Extract type information from fparser AST."""
        try:
            type_name = str(type_node.children[1].children[1])
            return FortranEntity(
                name=type_name,
                entity_type="type",
                file_path=str(file_path),
                line_start=1,  # TODO: Get actual line numbers
                line_end=1,
            )
        except:
            pass
        return None

    def _extract_arguments(self, stmt_node) -> List[str]:
        """Extract argument list from subroutine/function statement."""
        try:
            if len(stmt_node.children) > 2 and stmt_node.children[2]:
                return [str(arg).strip() for arg in stmt_node.children[2].children]
        except:
            pass
        return []

    def _extract_interface_name(self, interface_node) -> Optional[str]:
        """Extract interface name if available."""
        try:
            # This is a simplified extraction - real implementation would be more complex
            return "interface"  # Placeholder
        except:
            return None

    def parse_file(self, file_path: Path) -> Optional[ModuleInfo]:
        """Parse a single Fortran file."""
        logger.debug(f"Parsing file: {file_path}")

        # Try fparser first if available
        if FPARSER_AVAILABLE:
            result = self.parse_with_fparser(file_path)
            if result:
                return result

        # Fallback to regex parsing
        return self.parse_with_regex(file_path)

    def parse_project(self) -> Dict:
        """Parse entire Fortran project."""
        logger.info(f"Starting analysis of project: {self.config.project_name}")

        fortran_files = self.find_fortran_files()
        logger.info(f"Found {len(fortran_files)} Fortran files")

        modules = {}
        all_entities = []

        for file_path in fortran_files:
            module_info = self.parse_file(file_path)
            if module_info:
                modules[module_info.name] = module_info
                all_entities.extend(module_info.entities)

                # Store file-level info
                self.results["files"][str(file_path)] = {
                    "module_name": module_info.name,
                    "line_count": module_info.line_count,
                    "subroutines": len(module_info.subroutines),
                    "functions": len(module_info.functions),
                    "types": len(module_info.types),
                }

        # Build dependency graph
        dependencies = self._build_dependency_graph(modules)

        # Calculate statistics
        statistics = self._calculate_statistics(modules, all_entities)

        self.results.update(
            {
                "modules": modules,
                "dependencies": dependencies,
                "entities": all_entities,
                "statistics": statistics,
            }
        )

        logger.info(
            f"Analysis complete. Found {len(modules)} modules, {len(all_entities)} entities"
        )
        return self.results

    def _build_dependency_graph(self, modules: Dict[str, ModuleInfo]) -> Dict:
        """Build dependency relationships between modules."""
        dependencies = {}

        for module_name, module_info in modules.items():
            deps = []
            for use in module_info.uses:
                dep_module = use["module"]
                # Filter out system modules if configured
                if (
                    self.config.system_modules
                    and dep_module in self.config.system_modules
                ):
                    continue
                deps.append(dep_module)
            dependencies[module_name] = deps

        return dependencies

    def _calculate_statistics(
        self, modules: Dict[str, ModuleInfo], entities: List[FortranEntity]
    ) -> Dict:
        """Calculate project statistics."""
        total_lines = sum(m.line_count for m in modules.values())
        total_subroutines = sum(len(m.subroutines) for m in modules.values())
        total_functions = sum(len(m.functions) for m in modules.values())
        total_types = sum(len(m.types) for m in modules.values())

        return {
            "total_files": len(modules),
            "total_lines": total_lines,
            "total_subroutines": total_subroutines,
            "total_functions": total_functions,
            "total_types": total_types,
            "total_entities": len(entities),
            "average_lines_per_file": total_lines / len(modules) if modules else 0,
            "average_subroutines_per_module": (
                total_subroutines / len(modules) if modules else 0
            ),
        }

    def get_results(self) -> Dict:
        """Get parsing results."""
        return self.results

    def save_results(self, output_path: Union[str, Path], format: str = "json") -> None:
        """Save parsing results to file."""
        output_path = Path(output_path)

        # Convert ModuleInfo objects to dictionaries for serialization
        serializable_results = self._make_serializable(self.results)

        if format.lower() == "json":
            import json

            with open(output_path, "w") as f:
                json.dump(serializable_results, f, indent=2)
        elif format.lower() == "yaml":
            import yaml

            with open(output_path, "w") as f:
                yaml.dump(serializable_results, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _make_serializable(self, obj) -> Union[Dict, List, str, int, float, bool, None]:
        """Convert objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (ModuleInfo, FortranEntity)):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
