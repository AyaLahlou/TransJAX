"""
Translator Agent for converting Fortran to JAX.

This agent translates Fortran CTSM code to JAX following established patterns,
utilizing static analysis to handle complex physics modules.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from jax_agents.base_agent import BaseAgent
from jax_agents.prompts.translation_prompts_v2 import TRANSLATION_PROMPTS
from jax_agents.utils.config_loader import get_llm_config
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class TranslationResult:
    """Result of translating a Fortran module to JAX."""
    module_name: str
    physics_code: str
    source_directory: Optional[str] = None
    params_code: Optional[str] = None
    test_code: Optional[str] = None
    translation_notes: str = ""
    
    def save_structured(self, project_root: Path) -> Dict[str, Path]:
        """Save code to structured directories (src/, tests/, docs/)."""
        saved_files = {}
        subdir = self.source_directory or "clm_src_main"
        
        # Define Path Map
        path_map = {
            "physics": project_root / "src" / subdir / f"{self.module_name}.py",
            "params": project_root / "src" / subdir / f"{self.module_name}_params.py",
            "test": project_root / "tests" / subdir / f"test_{self.module_name}.py",
            "notes": project_root / "docs" / "translation_notes" / f"{self.module_name}_notes.md"
        }

        # Save Physics
        path_map["physics"].parent.mkdir(parents=True, exist_ok=True)
        path_map["physics"].write_text(self.physics_code)
        saved_files["physics"] = path_map["physics"]

        # Conditional Saves
        if self.params_code:
            path_map["params"].write_text(self.params_code)
            saved_files["params"] = path_map["params"]
        
        if self.test_code:
            path_map["test"].parent.mkdir(parents=True, exist_ok=True)
            path_map["test"].write_text(self.test_code)
            saved_files["test"] = path_map["test"]

        if self.translation_notes:
            path_map["notes"].parent.mkdir(parents=True, exist_ok=True)
            path_map["notes"].write_text(self.translation_notes)
            saved_files["notes"] = path_map["notes"]

        console.print(f"[green]✓ Structured save complete for {self.module_name}[/green]")
        return saved_files

class TranslatorAgent(BaseAgent):
    """Agent for translating Fortran code to JAX."""
    
    def __init__(
        self,
        analysis_results_path: Optional[Path] = None,
        translation_units_path: Optional[Path] = None,
        jax_ctsm_dir: Optional[Path] = None,
        fortran_root: Optional[Path] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        llm_config = get_llm_config()
        super().__init__(
            name="Translator",
            role="Fortran to JAX code translator",
            model=model or llm_config.get("model", "claude-sonnet-4-5"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 48000),
        )
        
        self.jax_ctsm_dir = jax_ctsm_dir
        self.fortran_root = fortran_root
        self.reference_patterns = self._load_reference_patterns()
        
        # Load analysis data
        self.analysis_results = self._load_json(analysis_results_path) if analysis_results_path else None
        self.translation_units = self._load_json(translation_units_path) if translation_units_path else None

    def _extract_code(self, text: str) -> str:
        """Extracts Python code from markdown blocks."""
        match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def translate_module(self, module_name: str, fortran_file: Optional[Path] = None) -> TranslationResult:
        """Orchestrates the translation of a full module using translation units."""
        console.print(f"\n[bold cyan]🔄 Translating {module_name} to JAX[/bold cyan]")
        
        module_info = self._extract_module_info(module_name)
        if not module_info:
            raise ValueError(f"Module '{module_name}' missing from analysis results.")

        # Determine path and read source
        fortran_path = fortran_file or self._remap_fortran_path(module_info['file_path'])
        fortran_lines = fortran_path.read_text().splitlines()
        
        # Process Units
        module_units = self._get_module_units(module_name)
        translated_units = []
        
        for i, unit in enumerate(module_units, 1):
            console.print(f"[cyan]Unit {i}/{len(module_units)}: {unit.get('id')}[/cyan]")
            code = self._translate_unit(module_name, unit, fortran_lines, module_info)
            translated_units.append({
                "unit_id": unit.get("id"),
                "translated_code": code
            })

        # Assemble
        result = self._assemble_module(module_name, translated_units, module_info)
        result.source_directory = self._extract_source_directory(module_info.get('file_path', ''))
        
        return result

    def _translate_unit(self, module_name: str, unit: Dict, lines: List[str], info: Dict) -> str:
        """Translates a specific snippet/subroutine."""
        start, end = unit.get("line_start", 1) - 1, unit.get("line_end", len(lines))
        fortran_snippet = '\n'.join(lines[start:end])
        
        prompt = TRANSLATION_PROMPTS["translate_unit"].format(
            module_name=module_name,
            fortran_code=fortran_snippet,
            unit_info=json.dumps(unit, indent=2),
            reference_pattern=self._get_reference_pattern()
        )
        
        response = self.query_claude(prompt=prompt, system_prompt=TRANSLATION_PROMPTS["system"])
        return self._extract_code(response)

    def _assemble_module(self, name: str, units: List[Dict], info: Dict) -> TranslationResult:
        """Combines translated units into a final module structure."""
        prompt = TRANSLATION_PROMPTS["assemble_module"].format(
            module_name=name,
            translated_units=json.dumps(units, indent=2),
            module_info=json.dumps(info, indent=2),
            reference_pattern=self._get_reference_pattern()
        )
        
        response = self.query_claude(prompt=prompt, system_prompt=TRANSLATION_PROMPTS["system"])
        
        # Parse output and return TranslationResult
        return TranslationResult(
            module_name=name,
            physics_code=self._extract_code(response),
            translation_notes=response.split("```")[0].strip()
        )

    def _remap_fortran_path(self, original_path: str) -> Path:
        """Maps absolute paths from analysis JSON to local filesystem."""
        path_obj = Path(original_path)
        if self.fortran_root:
            # Attempt to find file within the provided fortran_root
            potential_path = self.fortran_root / path_obj.name
            if potential_path.exists():
                return potential_path
        return path_obj

    def _load_json(self, path: Path) -> Dict:
        return json.loads(path.read_text())

    def _extract_module_info(self, name: str) -> Optional[Dict]:
        modules = self.analysis_results.get("parsing", {}).get("modules", {})
        return next((data for mod, data in modules.items() if mod.lower() == name.lower()), None)

    def _get_module_units(self, name: str) -> List[Dict]:
        units = self.translation_units.get("translation_units", [])
        filtered = [u for u in units if u.get("module_name", "").lower() == name.lower()]
        return sorted(filtered, key=lambda u: u.get("line_start", 0))

    def _load_reference_patterns(self) -> Dict[str, str]:
        """Loads existing JAX-CTSM code to serve as few-shot examples."""
        patterns = {}
        if self.jax_ctsm_dir:
            ref_path = self.jax_ctsm_dir / "src/jax_ctsm/physics/maintenance_respiration.py"
            if ref_path.exists():
                patterns["main"] = ref_path.read_text()[:5000] # Cap for context
        return patterns

    def _get_reference_pattern(self) -> str:
        return self.reference_patterns.get("main", "# No reference pattern found.")

    def _extract_source_directory(self, file_path: str) -> str:
        known_dirs = ['clm_src_main', 'clm_src_biogeophys', 'clm_src_utils']
        for part in Path(file_path).parts:
            if part in known_dirs: return part
        return 'clm_src_main'