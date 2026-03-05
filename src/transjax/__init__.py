"""
TransJAX
========

A publishable Python package for translating Fortran scientific code to JAX
using LLM agents (Claude / Anthropic).

Sub-packages
------------
transjax.analyzer
    Static analysis of Fortran codebases: parsing, dependency graphs,
    and translation-unit decomposition.

transjax.agents
    Multi-agent pipeline: translation, test generation, and iterative
    repair powered by the Anthropic API.

Quick start
-----------
>>> from transjax import OrchestratorAgent
>>> from transjax.analyzer import FortranAnalyzer
"""

__version__ = "0.1.0"
__author__ = "Aya Lahlou"

# -- analyzer public API --------------------------------------------------
# -- agents public API ----------------------------------------------------
from transjax.agents.base_agent import BaseAgent
from transjax.agents.orchestrator import OrchestratorAgent
from transjax.agents.repair_agent import RepairAgent, RepairResult
from transjax.agents.test_agent import TestAgent, TestGenerationResult
from transjax.agents.translator import TranslationResult, TranslatorAgent
from transjax.analyzer.analyzer import (
    FortranAnalyzer,
    create_analyzer_for_project,
    quick_analyze,
)
from transjax.analyzer.config.project_config import ConfigurationManager, FortranProjectConfig

__all__ = [
    # package metadata
    "__version__",
    # analyzer
    "FortranAnalyzer",
    "FortranProjectConfig",
    "ConfigurationManager",
    "create_analyzer_for_project",
    "quick_analyze",
    # agents
    "BaseAgent",
    "TranslatorAgent",
    "TranslationResult",
    "TestAgent",
    "TestGenerationResult",
    "RepairAgent",
    "RepairResult",
    "OrchestratorAgent",
]
