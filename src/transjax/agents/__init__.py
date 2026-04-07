"""
transjax.agents — Multi-agent system for Fortran-to-JAX translation.

Uses Claude (Anthropic) to translate Fortran scientific code to JAX,
generate tests, and iteratively repair failures.
"""

from transjax.agents.base_agent import BaseAgent
from transjax.agents.ftest_agent import FtestAgent, FtestFrameworkResult, FtestResult
from transjax.agents.golden_agent import GoldenAgent, GoldenCase, GoldenData, GoldenRunResult
from transjax.agents.orchestrator import OrchestratorAgent
from transjax.agents.parity_agent import (
    ParityAgent,
    ParityRunResult,
    ParitySubroutineResult,
)
from transjax.agents.parity_repair_agent import (
    IterationLog,
    ParityRepairAgent,
    ParityRepairResult,
)
from transjax.agents.integration_repair_agent import (
    IntegrationRepairAgent,
    IntegrationRepairResult,
    RepairIteration,
)
from transjax.agents.integrator_agent import IntegrationResult, IntegratorAgent
from transjax.agents.pipeline_runner import PipelineRunner
from transjax.agents.repair_agent import RalphIteration, RepairAgent, RepairResult
from transjax.agents.test_agent import TestAgent, TestGenerationResult
from transjax.agents.translator import TranslationResult, TranslatorAgent
from transjax.agents.utils.doc_generator import ProjectDocGenerator
from transjax.agents.utils.git_coordinator import GitCoordinator
from transjax.agents.utils.slurm_runner import SlurmRunner
from transjax.agents.utils.tmux_runner import TmuxClaudeRunner

__all__ = [
    "BaseAgent",
    "TranslatorAgent",
    "TranslationResult",
    "TestAgent",
    "TestGenerationResult",
    "IterationLog",
    "ParityRepairAgent",
    "ParityRepairResult",
    "IntegrationRepairAgent",
    "IntegrationRepairResult",
    "IntegrationResult",
    "IntegratorAgent",
    "PipelineRunner",
    "RepairIteration",
    "RalphIteration",
    "RepairAgent",
    "RepairResult",
    "OrchestratorAgent",
    "FtestAgent",
    "FtestResult",
    "FtestFrameworkResult",
    "GoldenAgent",
    "GoldenCase",
    "GoldenData",
    "GoldenRunResult",
    "ParityAgent",
    "ParitySubroutineResult",
    "ParityRunResult",
    "TmuxClaudeRunner",
    "SlurmRunner",
    "GitCoordinator",
    "ProjectDocGenerator",
]
