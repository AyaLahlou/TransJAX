"""
Test Agent for generating tests for JAX translations.

This agent focuses on Python/JAX test generation:
1. Analyzes Python function signatures
2. Generates comprehensive synthetic test data
3. Creates pytest files with multiple test scenarios
4. Provides test templates and documentation
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from jax_agents.base_agent import BaseAgent
from jax_agents.utils.config_loader import get_llm_config
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class TestGenerationResult:
    """Complete result of test generation."""
    module_name: str
    pytest_file: str
    test_data_file: str
    test_documentation: str
    
    def _save_file(self, path: Path, content: str, label: str):
        """Helper to write content and log to console."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        console.print(f"[green]✓ Saved {label} to {path}[/green]")
    
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """Save all test artifacts to a single directory."""
        return {
            "pytest": output_dir / f"test_{self.module_name}.py",
            "test_data": output_dir / f"test_data_{self.module_name}.json",
            "documentation": output_dir / f"test_documentation_{self.module_name}.md"
        }
        # In actual usage, you'd call self._save_file for each.

    def save_structured(self, project_root: Path, source_directory: str) -> Dict[str, Path]:
        """Save test artifacts to standard project layout."""
        paths = {
            "pytest": project_root / "tests" / source_directory / f"test_{self.module_name}.py",
            "test_data": project_root / "tests" / "test_data" / f"test_data_{self.module_name}.json",
            "documentation": project_root / "CLM-ml_v1" / "docs" / "test_documentation" / f"test_documentation_{self.module_name}.md"
        }
        
        self._save_file(paths["pytest"], self.pytest_file, "pytest file")
        self._save_file(paths["test_data"], self.test_data_file, "test data")
        self._save_file(paths["documentation"], self.test_documentation, "documentation")
        
        return paths


class TestAgent(BaseAgent):
    """
    Agent for generating comprehensive tests for JAX translations.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        llm_config = get_llm_config()
        super().__init__(
            name="TestAgent",
            role="JAX test generator",
            model=model or llm_config.get("model", "claude-sonnet-4-5"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 32000),
        )

    def _extract_block(self, text: str, block_type: str = "json") -> str:
        """Helper to extract content from markdown code blocks reliably."""
        pattern = rf"```(?:{block_type})?\n?(.*?)\n?```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def generate_tests(
        self,
        module_name: str,
        python_code: str,
        output_dir: Optional[Path] = None,
        num_test_cases: int = 10,
        include_edge_cases: bool = True,
        include_performance_tests: bool = False,
        source_directory: Optional[str] = None,
    ) -> TestGenerationResult:
        """Analyze, Generate Data, and Create Pytest suite."""
        console.print(f"\n[bold cyan]🧪 Generating tests for {module_name}[/bold cyan]")
        
        # Step 1: Analyze Signature
        console.print("[cyan]Step 1/3: Analyzing Python function...[/cyan]")
        python_sig = self._analyze_python_signature(python_code, module_name)
        
        # Step 2: Generate Data
        console.print("[cyan]Step 2/3: Generating test data...[/cyan]")
        test_data = self._generate_test_data(python_sig, num_test_cases, include_edge_cases)
        
        # Step 3: Generate Pytest & Docs
        console.print("[cyan]Step 3/3: Generating pytest file...[/cyan]")
        pytest_file = self._generate_pytest(
            module_name, python_sig, test_data, include_performance_tests, source_directory
        )
        
        test_docs = self._generate_documentation(module_name, python_sig, test_data, num_test_cases)
        
        result = TestGenerationResult(
            module_name=module_name,
            pytest_file=pytest_file,
            test_data_file=json.dumps(test_data, indent=2),
            test_documentation=test_docs,
        )
        
        if output_dir:
            result.save_structured(output_dir, source_directory or "generated_tests")
            
        return result

    def _analyze_python_signature(self, code: str, module: str) -> Dict[str, Any]:
        from jax_agents.prompts.test_prompts_simplified import TEST_PROMPTS
        response = self.query_claude(
            prompt=TEST_PROMPTS["analyze_python_signature"].format(python_code=code, module_name=module),
            system_prompt=TEST_PROMPTS["system"]
        )
        return json.loads(self._extract_block(response, "json"))

    def _generate_test_data(self, sig: Dict[str, Any], n: int, edge: bool) -> Dict[str, Any]:
        from jax_agents.prompts.test_prompts_simplified import TEST_PROMPTS
        response = self.query_claude(
            prompt=TEST_PROMPTS["generate_test_data"].format(
                python_signature=json.dumps(sig, indent=2), num_cases=n, include_edge_cases=edge
            ),
            system_prompt=TEST_PROMPTS["system"]
        )
        return json.loads(self._extract_block(response, "json"))

    def _generate_pytest(self, module: str, sig: Dict[str, Any], data: Dict[str, Any], perf: bool, src: str = None) -> str:
        from jax_agents.prompts.test_prompts_simplified import TEST_PROMPTS
        response = self.query_claude(
            prompt=TEST_PROMPTS["generate_pytest"].format(
                module_name=module,
                source_directory=src or "clm_src_main",
                python_signature=json.dumps(sig, indent=2),
                test_data=json.dumps(data, indent=2),
                include_performance=perf
            ),
            system_prompt=TEST_PROMPTS["system"]
        )
        return self._extract_block(response, "python")

    def _generate_documentation(self, module: str, sig: Dict[str, Any], data: Dict[str, Any], n: int) -> str:
        from jax_agents.prompts.test_prompts_simplified import TEST_PROMPTS
        summary = {
            "num_cases": n,
            "test_types": list(set(tc.get("metadata", {}).get("type", "nominal") 
                                  for tc in data.get("test_cases", [])))
        }
        return self.query_claude(
            prompt=TEST_PROMPTS["generate_documentation"].format(
                module_name=module, 
                python_signature=json.dumps(sig, indent=2), 
                test_data_summary=json.dumps(summary, indent=2)
            ),
            system_prompt=TEST_PROMPTS["system"]
        )