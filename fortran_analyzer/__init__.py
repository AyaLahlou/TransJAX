"""
Fortran Analyzer - A generic framework for analyzing Fortran codebases.

This package provides tools for parsing, analyzing, and preparing Fortran code
for translation to other languages or for general code understanding.
"""

__version__ = "1.0.0"
__author__ = "Fortran Analyzer Team"

from .analyzer import FortranAnalyzer, create_analyzer_for_project, quick_analyze
from .config.project_config import FortranProjectConfig, ConfigurationManager

__all__ = [
    "FortranAnalyzer",
    "FortranProjectConfig",
    "ConfigurationManager",
    "create_analyzer_for_project",
    "quick_analyze",
]
