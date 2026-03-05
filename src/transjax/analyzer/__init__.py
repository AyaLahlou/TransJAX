"""
transjax.analyzer — Static analysis of Fortran codebases.

Provides tools for parsing, analysing, and preparing Fortran code for
translation to Python/JAX.
"""

from transjax.analyzer.analyzer import FortranAnalyzer, create_analyzer_for_project, quick_analyze
from transjax.analyzer.config.project_config import ConfigurationManager, FortranProjectConfig

__all__ = [
    "FortranAnalyzer",
    "FortranProjectConfig",
    "ConfigurationManager",
    "create_analyzer_for_project",
    "quick_analyze",
]
