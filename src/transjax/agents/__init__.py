"""
transjax.agents — Multi-agent system for Fortran-to-JAX translation.

Uses Claude (Anthropic) to translate Fortran scientific code to JAX,
generate tests, and iteratively repair failures.
"""

from transjax.agents.base_agent import BaseAgent
from transjax.agents.orchestrator import OrchestratorAgent
from transjax.agents.repair_agent import RepairAgent, RepairResult
from transjax.agents.test_agent import TestAgent, TestGenerationResult
from transjax.agents.translator import TranslationResult, TranslatorAgent

__all__ = [
    "BaseAgent",
    "TranslatorAgent",
    "TranslationResult",
    "TestAgent",
    "TestGenerationResult",
    "RepairAgent",
    "RepairResult",
    "OrchestratorAgent",
]
