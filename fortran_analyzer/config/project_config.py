"""
Configuration management for Fortran code analysis.
This module provides a flexible configuration system that can adapt to different Fortran codebases.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class FortranProjectConfig:
    """Configuration class for Fortran project analysis."""

    # Project identification
    project_name: str
    project_root: str

    # Source code directories
    source_dirs: List[str] = field(default_factory=list)
    include_dirs: List[str] = field(default_factory=list)
    exclude_dirs: List[str] = field(default_factory=list)

    # File patterns
    fortran_extensions: List[str] = field(
        default_factory=lambda: [".f90", ".F90", ".f", ".F", ".f95", ".F95"]
    )
    include_patterns: List[str] = field(
        default_factory=lambda: ["**/*.f90", "**/*.F90"]
    )
    exclude_patterns: List[str] = field(default_factory=list)

    # Analysis settings
    max_translation_unit_lines: int = 150
    min_chunk_lines: int = 50
    preserve_interfaces: bool = True
    track_dependencies: bool = True

    # Parser settings
    fortran_standard: str = "f2003"  # f77, f90, f95, f2003, f2008
    case_sensitive: bool = False

    # Module/subroutine naming conventions
    naming_conventions: Dict[str, str] = field(default_factory=dict)

    # Dependency analysis
    external_libraries: List[str] = field(default_factory=list)
    system_modules: List[str] = field(default_factory=list)

    # Output settings
    output_dir: str = "output"
    generate_graphs: bool = True
    generate_metrics: bool = True

    # Visualization settings
    visualization: Dict[str, Union[str, bool, int, Tuple[int, int]]] = field(
        default_factory=lambda: {
            "node_color": "lightblue",
            "edge_color": "gray",
            "font_size": 10,
            "figure_size": (12, 8),
            "show_labels": True,
        }
    )

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        self.project_root = str(Path(self.project_root).resolve())
        self.source_dirs = [str(Path(self.project_root) / d) for d in self.source_dirs]
        self.include_dirs = [
            str(Path(self.project_root) / d) for d in self.include_dirs
        ]
        self.exclude_dirs = [
            str(Path(self.project_root) / d) for d in self.exclude_dirs
        ]
        self.output_dir = str(Path(self.project_root) / self.output_dir)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "FortranProjectConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Convert lists back to tuples where expected (e.g., figure_size)
        if "visualization" in data and isinstance(data["visualization"], dict):
            viz = data["visualization"]
            if "figure_size" in viz and isinstance(viz["figure_size"], list):
                viz["figure_size"] = tuple(viz["figure_size"])

        return cls(**data)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "FortranProjectConfig":
        """Load configuration from a JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""

        def convert_tuples(obj):
            """Convert tuples to lists for YAML serialization."""
            if isinstance(obj, dict):
                return {k: convert_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tuples(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            else:
                return obj

        data = convert_tuples(self.__dict__.copy())
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)

    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        data = self.__dict__.copy()
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    def validate(self) -> bool:
        """Validate the configuration."""
        errors = []

        # Check if project root exists
        if not Path(self.project_root).exists():
            errors.append(f"Project root does not exist: {self.project_root}")

        # Check if at least one source directory exists
        if not self.source_dirs:
            errors.append("No source directories specified")
        else:
            existing_dirs = [d for d in self.source_dirs if Path(d).exists()]
            if not existing_dirs:
                errors.append(
                    f"None of the source directories exist: {self.source_dirs}"
                )

        # Validate Fortran standard
        valid_standards = ["f77", "f90", "f95", "f2003", "f2008"]
        if self.fortran_standard not in valid_standards:
            errors.append(
                f"Invalid Fortran standard: {self.fortran_standard}. Must be one of: {valid_standards}"
            )

        # Validate line limits
        if self.max_translation_unit_lines < self.min_chunk_lines:
            errors.append("max_translation_unit_lines must be >= min_chunk_lines")

        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False

        return True


class ConfigurationManager:
    """Manages configuration templates for different types of Fortran projects."""

    TEMPLATES: Dict[str, Dict[str, Any]] = {
        "ctsm": {
            "project_name": "CTSM",
            "source_dirs": [
                "clm_src_biogeophys", 
                "clm_src_main", 
                "clm_src_utils", 
                "clm_src_cpl",
                "cime_src_share_util",
                "multilayer_canopy",
                "offline_driver"
            ],
            "fortran_extensions": [".F90", ".f90"],
            "system_modules": [
                "shr_kind_mod",
                "clm_varpar",
                "clm_varctl",
                "clm_varcon",
            ],
            "external_libraries": ["netcdf", "esmf"],
            "naming_conventions": {"module_suffix": "Mod", "type_suffix": "_type"},
        },
        "scientific_computing": {
            "project_name": "Scientific Computing Project",
            "source_dirs": ["src"],
            "fortran_extensions": [".f90", ".F90"],
            "system_modules": ["iso_fortran_env", "iso_c_binding"],
            "external_libraries": ["lapack", "blas", "mpi"],
            "preserve_interfaces": True,
        },
        "numerical_library": {
            "project_name": "Numerical Library",
            "source_dirs": ["src", "lib"],
            "fortran_extensions": [".f90", ".F90", ".f95"],
            "track_dependencies": True,
            "generate_metrics": True,
            "max_translation_unit_lines": 100,
        },
        "climate_model": {
            "project_name": "Climate Model",
            "source_dirs": ["src/physics", "src/dynamics", "src/main"],
            "fortran_extensions": [".F90"],
            "external_libraries": ["netcdf", "mpi", "esmf"],
            "system_modules": ["shr_kind_mod", "shr_const_mod"],
            "preserve_interfaces": True,
        },
        "generic": {
            "project_name": "Generic Fortran Project",
            "source_dirs": ["src"],
            "fortran_extensions": [".f90", ".F90"],
            "system_modules": ["iso_fortran_env"],
            "track_dependencies": True,
        },
    }

    @classmethod
    def create_config_from_template(
        cls, template_name: str, project_root: str, overrides: Optional[Dict] = None
    ) -> FortranProjectConfig:
        """Create a configuration from a predefined template."""
        if template_name not in cls.TEMPLATES:
            raise ValueError(
                f"Unknown template: {template_name}. Available: {list(cls.TEMPLATES.keys())}"
            )

        template_data = cls.TEMPLATES[template_name].copy()
        template_data["project_root"] = project_root

        if overrides:
            template_data.update(overrides)

        return FortranProjectConfig(**template_data)

    @classmethod
    def list_templates(cls) -> List[str]:
        """List available configuration templates."""
        return list(cls.TEMPLATES.keys())

    @classmethod
    def get_template_info(cls, template_name: str) -> Dict[str, Any]:
        """Get information about a specific template."""
        if template_name not in cls.TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")

        return cls.TEMPLATES[template_name].copy()

    @classmethod
    def auto_detect_project_type(cls, project_root: str) -> str:
        """Attempt to automatically detect project type from directory structure."""
        root_path = Path(project_root)

        # Check for CTSM indicators
        if (
            (root_path / "src" / "biogeophys").exists()
            or any(root_path.glob("**/*clm*"))
            or any(root_path.glob("**/*ctsm*"))
        ):
            return "ctsm"

        # Check for climate model indicators
        if (root_path / "src" / "physics").exists() and (
            root_path / "src" / "dynamics"
        ).exists():
            return "climate_model"

        # Check for library structure
        if (root_path / "lib").exists() or (root_path / "include").exists():
            return "numerical_library"

        # Check for scientific computing patterns
        if any(root_path.glob("**/*mpi*")) or any(root_path.glob("**/*parallel*")):
            return "scientific_computing"

        return "generic"


def load_config(config_path: Union[str, Path]) -> FortranProjectConfig:
    """Load configuration from file, supporting both YAML and JSON."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix.lower() in [".yaml", ".yml"]:
        return FortranProjectConfig.from_yaml(config_path)
    elif config_path.suffix.lower() == ".json":
        return FortranProjectConfig.from_json(config_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def create_default_config(
    project_root: str, template: str = "generic"
) -> FortranProjectConfig:
    """Create a default configuration for a project."""
    manager = ConfigurationManager()
    detected_type = manager.auto_detect_project_type(project_root)

    if template == "auto":
        template = detected_type

    logger.info(f"Creating configuration using template: {template}")
    return manager.create_config_from_template(template, project_root)
