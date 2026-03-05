"""
Tests for transjax.agents.orchestrator.OrchestratorAgent.

These tests verify that OrchestratorAgent can be instantiated without a real
Anthropic API key — no network calls are made.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from transjax.agents.orchestrator import OrchestratorAgent, ModuleStatus, PipelineResults


@pytest.fixture
def tmp_dirs():
    with tempfile.TemporaryDirectory() as fortran_dir, \
         tempfile.TemporaryDirectory() as output_dir:
        yield Path(fortran_dir), Path(output_dir)


def _make_orchestrator(fortran_dir: Path, output_dir: Path, **kwargs) -> OrchestratorAgent:
    """Create an OrchestratorAgent with mocked sub-agents."""
    with patch("transjax.agents.orchestrator.TranslatorAgent"), \
         patch("transjax.agents.orchestrator.TestAgent"), \
         patch("transjax.agents.orchestrator.RepairAgent"):
        return OrchestratorAgent(
            fortran_dir=fortran_dir,
            output_dir=output_dir,
            skip_tests=True,
            skip_repair=True,
            **kwargs,
        )


def test_orchestrator_instantiation(tmp_dirs):
    """OrchestratorAgent must instantiate without an API key."""
    fortran_dir, output_dir = tmp_dirs
    orch = _make_orchestrator(fortran_dir, output_dir)
    assert orch is not None


def test_output_directories_created(tmp_dirs):
    """Instantiation must create the expected output subdirectories."""
    fortran_dir, output_dir = tmp_dirs
    _make_orchestrator(fortran_dir, output_dir)

    for subdir in ["src", "tests", "docs", "reports", "static_analysis"]:
        assert (output_dir / subdir).is_dir(), f"Expected {subdir}/ to be created"


def test_module_status_dataclass():
    """ModuleStatus fields must have correct defaults."""
    status = ModuleStatus(name="my_module")
    assert status.name == "my_module"
    assert status.translated is False
    assert status.final_status == "pending"
    assert status.error_message is None


def test_pipeline_results_to_dict():
    """PipelineResults.to_dict() must return the expected keys."""
    results = PipelineResults(
        total_modules=3,
        translated_count=3,
        tests_generated=2,
        tests_passed=2,
        repairs_needed=0,
        final_failures=0,
    )
    d = results.to_dict()
    assert set(d.keys()) == {
        "translated_count",
        "tests_generated",
        "tests_passed",
        "repairs_needed",
        "final_failures",
    }
    assert d["translated_count"] == 3


def test_determine_translation_order_fallback(tmp_dirs):
    """Without dependency data, modules should be returned sorted."""
    fortran_dir, output_dir = tmp_dirs
    orch = _make_orchestrator(fortran_dir, output_dir)

    analysis_data = {
        "analysis_results": {
            "modules": {"beta": {}, "alpha": {}, "gamma": {}}
        }
    }
    order = orch._determine_translation_order(analysis_data)
    assert order == sorted(order)


def test_generate_summary_counts(tmp_dirs):
    """_generate_summary must count statuses correctly."""
    fortran_dir, output_dir = tmp_dirs
    orch = _make_orchestrator(fortran_dir, output_dir)

    orch.module_statuses = {
        "a": ModuleStatus(name="a", translated=True, final_status="success"),
        "b": ModuleStatus(name="b", translated=True, final_status="failed"),
        "c": ModuleStatus(name="c", translated=False, final_status="failed"),
    }

    summary = orch._generate_summary()
    assert summary.total_modules == 3
    assert summary.translated_count == 2
    assert summary.final_failures == 2
