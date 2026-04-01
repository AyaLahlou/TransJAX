"""
Translator Agent for converting Fortran to JAX.

Translates Fortran subroutines/functions to JAX by:
1. Pre-parsing the Fortran interface (no hallucination of argument counts/types)
2. Translating each unit with a structured interface contract
3. Assembling the module (skipped for single-unit modules)
4. Validating the output with ast.parse + heuristic checks
"""

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from transjax.agents.base_agent import BaseAgent
from transjax.agents.prompts.translation_prompts import (
    MODULE_ASSEMBLY_PROMPT,
    TRANSLATOR_SYSTEM_PROMPT,
    UNIT_TRANSLATION_PROMPT,
    WHOLE_MODULE_PROMPT,
)
from transjax.agents.utils.config_loader import get_llm_config

console = Console()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Fortran → JAX dtype mapping
# ─────────────────────────────────────────────────────────────────────────────

_FORTRAN_TO_JAX: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"real\s*\(\s*r?8\s*\)",          re.I), "jnp.float64"),
    (re.compile(r"real\s*\*\s*8",                  re.I), "jnp.float64"),
    (re.compile(r"double\s+precision",             re.I), "jnp.float64"),
    (re.compile(r"real\s*\(\s*r?4\s*\)",          re.I), "jnp.float32"),
    (re.compile(r"real\s*\*\s*4",                  re.I), "jnp.float32"),
    (re.compile(r"\breal\b",                        re.I), "jnp.float64"),  # default → f64
    (re.compile(r"integer\s*\(\s*\w+\s*\)",        re.I), "jnp.int32"),
    (re.compile(r"\binteger\b",                    re.I), "jnp.int32"),
    (re.compile(r"\blogical\b",                    re.I), "jnp.bool_"),
    (re.compile(r"complex\s*\(\s*r?8\s*\)",       re.I), "jnp.complex128"),
    (re.compile(r"complex\s*\(\s*r?4\s*\)",       re.I), "jnp.complex64"),
    (re.compile(r"\bcomplex\b",                    re.I), "jnp.complex128"),
    (re.compile(r"character",                      re.I), "str"),
]

# Regex patterns for Fortran source parsing
_RE_SUBROUTINE = re.compile(
    r"^\s*(?:pure\s+|elemental\s+)*subroutine\s+(\w+)\s*\(([^)]*)\)",
    re.I | re.M,
)
_RE_FUNCTION = re.compile(
    r"^\s*(?:(?:pure|elemental|recursive)\s+)*"
    r"(?:[\w\s\*()]+?\s+)?function\s+(\w+)\s*\(([^)]*)\)",
    re.I | re.M,
)
_RE_INTENT = re.compile(r"intent\s*\(\s*(\w+)\s*\)", re.I)
_RE_DIMENSION = re.compile(r"dimension\s*\(([^)]+)\)", re.I)
_RE_ARR_SUFFIX = re.compile(r"\(([^)]+)\)")


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TranslationResult:
    """Result of translating a Fortran module to JAX."""
    module_name: str
    physics_code: str
    source_directory: Optional[str] = None
    params_code: Optional[str] = None
    test_code: Optional[str] = None
    translation_notes: str = ""

    def save(self, output_dir: Path) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}

        physics_file = output_dir / f"{self.module_name}.py"
        physics_file.write_text(self.physics_code)
        saved_files["physics"] = physics_file
        console.print(f"[green]✓ Saved physics module to {physics_file}[/green]")

        if self.params_code:
            params_file = output_dir / f"{self.module_name}_params.py"
            params_file.write_text(self.params_code)
            saved_files["params"] = params_file
            console.print(f"[green]✓ Saved parameters to {params_file}[/green]")

        if self.test_code:
            test_file = output_dir / f"test_{self.module_name}.py"
            test_file.write_text(self.test_code)
            saved_files["test"] = test_file
            console.print(f"[green]✓ Saved tests to {test_file}[/green]")

        if self.translation_notes:
            notes_file = output_dir / f"{self.module_name}_translation_notes.md"
            notes_file.write_text(self.translation_notes)
            saved_files["notes"] = notes_file

        return saved_files

    def save_structured(self, project_root: Path) -> Dict[str, Path]:
        saved_files = {}
        source_subdir = self.source_directory
        src_target_dir = (
            project_root / "src" / source_subdir if source_subdir
            else project_root / "src"
        )
        src_target_dir.mkdir(parents=True, exist_ok=True)

        physics_file = src_target_dir / f"{self.module_name}.py"
        physics_file.write_text(self.physics_code)
        saved_files["physics"] = physics_file
        console.print(f"[green]✓ Saved physics module to {physics_file}[/green]")

        if self.params_code:
            params_file = src_target_dir / f"{self.module_name}_params.py"
            params_file.write_text(self.params_code)
            saved_files["params"] = params_file

        if self.test_code:
            test_target_dir = (
                project_root / "tests" / source_subdir if source_subdir
                else project_root / "tests"
            )
            test_target_dir.mkdir(parents=True, exist_ok=True)
            test_file = test_target_dir / f"test_{self.module_name}.py"
            test_file.write_text(self.test_code)
            saved_files["test"] = test_file

        if self.translation_notes:
            docs_dir = project_root / "docs" / "translation_notes"
            docs_dir.mkdir(parents=True, exist_ok=True)
            notes_file = docs_dir / f"{self.module_name}_translation_notes.md"
            notes_file.write_text(self.translation_notes)
            saved_files["notes"] = notes_file

        return saved_files


# ─────────────────────────────────────────────────────────────────────────────
# TranslatorAgent
# ─────────────────────────────────────────────────────────────────────────────

class TranslatorAgent(BaseAgent):
    """
    Translates Fortran ESM code to JAX/Python.

    Key improvements over legacy translator:
    - Pre-parses Fortran interface before sending to Claude (eliminates
      argument-count / intent / dtype hallucinations).
    - Uses structured UNIT_TRANSLATION_PROMPT with interface contract.
    - Passes full previously-translated code as context (not 200-char snippets).
    - Validates output with ast.parse + heuristic checks.
    - Skips assembly LLM call for single-unit modules.
    """

    def __init__(
        self,
        analysis_results_path: Optional[Path] = None,
        translation_units_path: Optional[Path] = None,
        reference_dir: Optional[Path] = None,
        fortran_root: Optional[Path] = None,
        gcm_model_name: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        translation_mode: str = "units",
    ):
        llm_config = get_llm_config()
        super().__init__(
            name="Translator",
            role="Fortran to JAX code translator",
            model=model or llm_config.get("model", "claude-sonnet-4-5"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 48000),
        )

        self.reference_dir = reference_dir
        self.fortran_root = fortran_root
        self.gcm_model_name = gcm_model_name or "unspecified"
        self.translation_mode = translation_mode  # "units" | "whole"
        self.reference_patterns = self._load_reference_patterns()

        self.analysis_results = self._load_json(analysis_results_path) if analysis_results_path else None
        self.translation_units = self._load_json(translation_units_path) if translation_units_path else None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def translate_module(
        self,
        module_name: str,
        fortran_file: Optional[Path] = None,
        analysis: Optional[Any] = None,
        output_dir: Optional[Path] = None,
    ) -> TranslationResult:
        """Translate a complete Fortran module to JAX.

        Dispatches to ``translate_module_whole`` when ``translation_mode == 'whole'``
        or to the unit-by-unit pipeline when ``translation_mode == 'units'``.
        """
        if self.translation_mode == "whole":
            return self.translate_module_whole(
                module_name, fortran_file=fortran_file, output_dir=output_dir
            )

        console.print(f"\n[bold cyan]🔄 Translating {module_name} to JAX[/bold cyan]")

        module_info = self._extract_module_info(module_name)
        if not module_info:
            raise ValueError(f"Module '{module_name}' not found in analysis results")

        fortran_path = fortran_file or self._remap_fortran_path(module_info["file_path"])
        console.print(f"[dim]Reading from: {fortran_path}[/dim]")
        fortran_code = Path(fortran_path).read_text()
        fortran_lines = fortran_code.split("\n")

        module_units = self._get_module_units(module_name)

        if not module_units:
            console.print("[yellow]⚠ No translation units found — using full-module fallback[/yellow]")
            return self._translate_module_legacy(module_name, fortran_code, module_info, output_dir)

        console.print(f"[cyan]Found {len(module_units)} translation units[/cyan]")

        translated_units: List[Dict[str, Any]] = []
        for i, unit in enumerate(module_units, 1):
            unit_id = unit.get("id", "unknown")
            console.print(
                f"[cyan]Translating unit {i}/{len(module_units)}: "
                f"{unit_id} ({unit.get('unit_type', 'unknown')})[/cyan]"
            )
            translated_code = self._translate_unit(
                module_name=module_name,
                unit=unit,
                fortran_lines=fortran_lines,
                module_info=module_info,
                previously_translated=translated_units,
            )
            translated_units.append(
                {
                    "unit_id": unit_id,
                    "unit_type": unit.get("unit_type", "unknown"),
                    "translated_code": translated_code,
                    "original_lines": f"{unit.get('line_start', 0)}-{unit.get('line_end', 0)}",
                }
            )

        result = self._assemble_module(module_name, translated_units, module_info)
        result.source_directory = self._extract_source_directory(module_info.get("file_path", ""))
        console.print("[green]✓ Translation complete![/green]")

        if output_dir:
            result.save(output_dir)
        return result

    # ------------------------------------------------------------------ #
    # Whole-module translation (--mode whole)                             #
    # ------------------------------------------------------------------ #

    def translate_module_whole(
        self,
        module_name: str,
        fortran_file: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> TranslationResult:
        """
        Translate a complete Fortran module in a single LLM call.

        Sends the full Fortran source together with pre-parsed interface
        contracts for every subroutine/function in the file.  Produces one
        Python module with all routines translated.

        Use when:
        - The module is small-to-medium (< ~800 lines of Fortran).
        - You want a single coherent translation instead of unit-by-unit assembly.
        - Context between subroutines matters (shared locals, SAVE vars, etc.).
        """
        console.print(f"\n[bold cyan]🔄 Translating {module_name} to JAX (whole-module mode)[/bold cyan]")

        module_info = self._extract_module_info(module_name)
        if not module_info:
            raise ValueError(f"Module '{module_name}' not found in analysis results")

        fortran_path = fortran_file or self._remap_fortran_path(module_info["file_path"])
        console.print(f"[dim]Reading from: {fortran_path}[/dim]")
        fortran_code = Path(fortran_path).read_text()

        # Pre-parse every subroutine/function in the file
        all_ifaces = self._parse_all_interfaces(fortran_code)
        n_routines = len(all_ifaces)
        console.print(f"[dim]Found {n_routines} routine(s) to translate[/dim]")

        # Build the interface summary block for the prompt
        interface_summary = self._format_interface_summary(all_ifaces)

        prompt = WHOLE_MODULE_PROMPT.format(
            gcm_model_name=self.gcm_model_name,
            module_name=module_name,
            source_file=module_info.get("file_path", "unknown"),
            n_routines=n_routines,
            interface_summary=interface_summary,
            fortran_code=fortran_code,
        )

        response = self.query_claude(
            prompt=prompt,
            system_prompt=TRANSLATOR_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
        )
        code = self._extract_code(response)

        # Validate each parsed routine name is present
        for iface in all_ifaces:
            name = iface.get("subroutine_name", "")
            if name:
                warnings = self._validate_translation(code, name)
                for w in warnings:
                    console.print(f"[yellow]  ⚠ [{name}] {w}[/yellow]")

        result = TranslationResult(
            module_name=module_name,
            physics_code=code if code else response,
            source_directory=self._extract_source_directory(module_info.get("file_path", "")),
        )
        console.print("[green]✓ Whole-module translation complete![/green]")

        if output_dir:
            result.save(output_dir)
        return result

    def _parse_all_interfaces(self, fortran_code: str) -> List[Dict[str, Any]]:
        """
        Find every subroutine/function in *fortran_code* and parse its interface.

        Uses the balanced-depth approach to isolate each routine's body, then
        calls ``_parse_fortran_interface`` on each one.

        Returns a list of interface dicts (one per routine) in source order.
        """
        ifaces: List[Dict[str, Any]] = []
        lines = fortran_code.split("\n")

        # Patterns to detect routine start / end
        start_re = re.compile(
            r"^\s*(?:(?:pure|elemental|recursive)\s+)*"
            r"(?:subroutine|(?:[\w\s\*()]+?\s+)?function)\s+\w+\s*\(",
            re.I,
        )
        end_re = re.compile(r"^\s*end\s+(?:subroutine|function)\b", re.I)

        i = 0
        while i < len(lines):
            if start_re.match(lines[i]):
                start = i
                depth = 1
                i += 1
                while i < len(lines) and depth > 0:
                    stripped = lines[i].strip()
                    if not stripped.startswith("!"):
                        if start_re.match(lines[i]):
                            depth += 1
                        elif end_re.match(lines[i]):
                            depth -= 1
                    i += 1
                # lines[start:i] is the complete routine body
                routine_src = "\n".join(lines[start:i])
                iface = self._parse_fortran_interface(routine_src)
                if iface["subroutine_name"]:  # skip if parse found nothing
                    ifaces.append(iface)
            else:
                i += 1

        return ifaces

    def _format_interface_summary(self, ifaces: List[Dict[str, Any]]) -> str:
        """Format pre-parsed interfaces into a human-readable block for the prompt."""
        if not ifaces:
            return "(no subroutines/functions detected)"

        parts: List[str] = []
        for iface in ifaces:
            name = iface["subroutine_name"]
            parts.append(f"### {name}")
            parts.append(iface["interface_table"])
            parts.append(f"Python signature: {iface['python_signature']}")
            if iface["return_fields"].strip() and "no output" not in iface["return_fields"]:
                parts.append(f"Return fields:\n{iface['return_fields']}")
            parts.append("")  # blank line between routines
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Interface pre-parser                                                 #
    # ------------------------------------------------------------------ #

    def _parse_fortran_interface(self, fortran_code: str) -> Dict[str, Any]:
        """
        Extract the subroutine/function interface from Fortran source.

        Processes source line-by-line (after joining Fortran continuation
        lines) to avoid cross-line regex contamination.

        Returns a dict with:
          subroutine_name, arg_names (ordered), args (name→{intent,
          fortran_type, jax_type, shape, is_array}), inputs, inouts, outputs,
          interface_table (str), python_signature (str), return_fields (str).

        Falls back gracefully on parse failures.
        """
        def _jax_type(fortran_type: str) -> str:
            for pat, jax in _FORTRAN_TO_JAX:
                if pat.search(fortran_type):
                    return jax
            return "jnp.float64"

        # ── join Fortran continuation lines ──────────────────────────────
        raw_lines = fortran_code.split("\n")
        joined: List[str] = []
        i = 0
        while i < len(raw_lines):
            line = raw_lines[i].rstrip()
            while line.rstrip().endswith("&"):
                line = line.rstrip()[:-1]
                i += 1
                if i < len(raw_lines):
                    line += raw_lines[i].strip()
            joined.append(line)
            i += 1

        # ── find subroutine / function declaration ────────────────────────
        name = ""
        arg_names: List[str] = []
        for line in joined:
            stripped = line.strip()
            if stripped.startswith("!"):
                continue
            m = _RE_SUBROUTINE.match(line)
            if not m:
                m = _RE_FUNCTION.match(line)
            if m:
                name = m.group(1).lower()
                raw_args = m.group(2) if m.lastindex >= 2 else ""
                arg_names = [a.strip().lower() for a in raw_args.split(",") if a.strip()]
                break

        # ── initialise arg metadata ───────────────────────────────────────
        args: Dict[str, Dict[str, Any]] = {
            a: {"intent": "in", "fortran_type": "real(r8)",
                "jax_type": "jnp.float64", "shape": "", "is_array": False}
            for a in arg_names
        }

        def _split_vars(var_part: str) -> List[str]:
            """Split a variable list at top-level commas (ignoring those inside parens)."""
            result: List[str] = []
            depth = 0
            current: List[str] = []
            for ch in var_part:
                if ch == "(":
                    depth += 1
                    current.append(ch)
                elif ch == ")":
                    depth -= 1
                    current.append(ch)
                elif ch == "," and depth == 0:
                    result.append("".join(current).strip())
                    current = []
                else:
                    current.append(ch)
            if current:
                result.append("".join(current).strip())
            return [v for v in result if v]

        # ── parse declaration lines (must contain `::`) ───────────────────
        for line in joined:
            stripped = line.strip()
            if stripped.startswith("!") or "::" not in line:
                continue

            type_part, var_part = line.split("::", 1)
            type_part = type_part.strip()
            var_part  = var_part.strip()

            # Extract intent
            intent_m = _RE_INTENT.search(type_part)
            intent = intent_m.group(1).lower() if intent_m else "in"

            # Extract dimension attribute (e.g. dimension(n,m))
            dim_m = _RE_DIMENSION.search(type_part)
            global_shape = dim_m.group(1) if dim_m else ""

            # Clean type spec: strip attribute keywords after the first comma
            # e.g. "real(r8), intent(in), dimension(n)" → "real(r8)"
            type_clean = re.split(r",\s*(?:intent|dimension|allocatable|pointer|target|optional)\b",
                                   type_part, maxsplit=1, flags=re.I)[0].strip()

            jtype = _jax_type(type_clean)

            # Process each variable on this declaration line
            # (split at top-level commas only — array shapes can contain commas)
            for var_expr in _split_vars(var_part):
                if not var_expr:
                    continue
                var_name = re.split(r"[\s(]", var_expr)[0].lower()
                if var_name not in args:
                    continue

                shape_m = _RE_ARR_SUFFIX.search(var_expr)
                if shape_m:
                    shape, is_arr = shape_m.group(1), True
                elif global_shape:
                    shape, is_arr = global_shape, True
                else:
                    shape, is_arr = "", False

                args[var_name] = {
                    "intent": intent,
                    "fortran_type": type_clean,
                    "jax_type": jtype,
                    "shape": shape,
                    "is_array": is_arr,
                }

        # ── categorise ───────────────────────────────────────────────────
        inputs  = [a for a in arg_names if args[a]["intent"] == "in"]
        inouts  = [a for a in arg_names if args[a]["intent"] == "inout"]
        outputs = [a for a in arg_names if args[a]["intent"] == "out"]

        # ── interface table ───────────────────────────────────────────────
        rows = [
            "  {:<20} {:<10} {:<22} {:<16} {}".format(
                "Name", "Intent", "Fortran type", "JAX type", "Shape"),
            "  " + "-" * 80,
        ]
        for a in arg_names:
            info = args[a]
            rows.append("  {:<20} {:<10} {:<22} {:<16} {}".format(
                a, info["intent"], info["fortran_type"],
                info["jax_type"], info["shape"] or "scalar",
            ))
        interface_table = "\n".join(rows)

        # ── python signature ──────────────────────────────────────────────
        py_args = [
            "{}: {}".format(a, "jax.Array" if args[a]["is_array"] else args[a]["jax_type"])
            for a in arg_names if args[a]["intent"] in ("in", "inout")
        ]
        ret_name = (name.capitalize() if name else "Unknown") + "Result"
        python_signature = f"def {name or 'unknown'}({', '.join(py_args)}) -> {ret_name}:"

        # ── return fields ─────────────────────────────────────────────────
        out_fields = inouts + outputs
        if out_fields:
            return_fields = "\n".join(
                "  {}: {}".format(a, "jax.Array" if args[a]["is_array"] else args[a]["jax_type"])
                for a in out_fields
            )
        else:
            return_fields = "  (no output fields — subroutine has no intent(out)/intent(inout))"

        return {
            "subroutine_name": name,
            "arg_names": arg_names,
            "args": args,
            "inputs": inputs,
            "inouts": inouts,
            "outputs": outputs,
            "interface_table": interface_table,
            "python_signature": python_signature,
            "return_fields": return_fields,
        }

    # ------------------------------------------------------------------ #
    # Translation helpers                                                  #
    # ------------------------------------------------------------------ #

    def _translate_unit(
        self,
        module_name: str,
        unit: Dict[str, Any],
        fortran_lines: List[str],
        module_info: Dict[str, Any],
        previously_translated: List[Dict[str, Any]],
    ) -> str:
        """Translate a single translation unit with interface pre-parsing."""
        line_start = unit.get("line_start", 1) - 1
        line_end   = unit.get("line_end", len(fortran_lines))
        unit_fortran = "\n".join(fortran_lines[line_start:line_end])

        # Pre-parse the Fortran interface to ground the prompt
        iface = self._parse_fortran_interface(unit_fortran)

        # Build "already translated" context — full signatures, not truncated snippets
        if previously_translated:
            already_parts = []
            for prev in previously_translated:
                code = prev["translated_code"]
                # Extract just the function definition lines (up to first blank line after def)
                sig_lines = []
                in_sig = False
                for line in code.split("\n"):
                    if line.strip().startswith("def ") or line.strip().startswith("@"):
                        in_sig = True
                    if in_sig:
                        sig_lines.append(line)
                        if line.strip() == "" and len(sig_lines) > 2:
                            break
                already_parts.append(
                    f"# --- {prev['unit_id']} ({prev['unit_type']}) ---\n"
                    + ("\n".join(sig_lines) if sig_lines else f"# (full code: {len(code)} chars)")
                )
            already_translated_str = "\n\n".join(already_parts)
        else:
            already_translated_str = "(none — this is the first unit)"

        complexity = unit.get("complexity_score", "unknown")

        prompt = UNIT_TRANSLATION_PROMPT.format(
            gcm_model_name=self.gcm_model_name,
            module_name=module_name,
            source_file=module_info.get("file_path", "unknown"),
            line_start=unit.get("line_start", 1),
            line_end=unit.get("line_end", len(fortran_lines)),
            complexity=complexity,
            subroutine_name=iface["subroutine_name"] or unit.get("id", "unknown"),
            interface_table=iface["interface_table"],
            python_signature=iface["python_signature"],
            return_fields=iface["return_fields"],
            fortran_code=unit_fortran,
            already_translated=already_translated_str,
        )

        response = self.query_claude(
            prompt=prompt,
            system_prompt=TRANSLATOR_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
        )
        code = self._extract_code(response)

        # Validate and warn — don't raise, let the user see the output
        warnings = self._validate_translation(code, iface["subroutine_name"] or module_name)
        for w in warnings:
            console.print(f"[yellow]  ⚠ {w}[/yellow]")

        return code

    def _assemble_module(
        self,
        module_name: str,
        translated_units: List[Dict[str, Any]],
        module_info: Dict[str, Any],
    ) -> TranslationResult:
        """
        Assemble translated units into a complete module.

        If there is only one unit, skip the LLM assembly call (avoids
        structural regressions introduced by a second pass).
        """
        if len(translated_units) == 1:
            # Single unit: just return the code as-is with a minimal header guard
            code = translated_units[0]["translated_code"]
            return TranslationResult(
                module_name=module_name,
                physics_code=code,
                source_directory=self._extract_source_directory(module_info.get("file_path", "")),
            )

        # Multiple units: ask Claude to deduplicate and order them
        public_api = [u["unit_id"] for u in translated_units]
        all_units_source = "\n\n".join(
            f"# ════ Unit: {u['unit_id']} (lines {u['original_lines']}) ════\n"
            + u["translated_code"]
            for u in translated_units
        )

        prompt = MODULE_ASSEMBLY_PROMPT.format(
            n_units=len(translated_units),
            module_name=module_name,
            gcm_model_name=self.gcm_model_name,
            original_file=module_info.get("file_path", "unknown"),
            public_api=json.dumps(public_api),
            all_units_source=all_units_source,
        )

        response = self.query_claude(
            prompt=prompt,
            system_prompt=TRANSLATOR_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
        )
        return self._parse_translation_response(response, module_name, module_info)

    def _translate_module_legacy(
        self,
        module_name: str,
        fortran_code: str,
        module_info: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> TranslationResult:
        """Fallback: treat entire file as one translation unit."""
        console.print("[cyan]Generating JAX translation (legacy/full-module mode)...[/cyan]")
        fortran_lines = fortran_code.split("\n")
        synthetic_unit = {
            "id": f"{module_name}_full",
            "unit_type": "root",
            "line_start": 1,
            "line_end": len(fortran_lines),
            "parent_id": None,
            "complexity_score": "high",
        }
        translated_code = self._translate_unit(
            module_name=module_name,
            unit=synthetic_unit,
            fortran_lines=fortran_lines,
            module_info=module_info,
            previously_translated=[],
        )
        translated_units = [{
            "unit_id": synthetic_unit["id"],
            "unit_type": synthetic_unit["unit_type"],
            "translated_code": translated_code,
            "original_lines": f"1-{len(fortran_lines)}",
        }]
        # Single synthetic unit — assembly is skipped
        return self._assemble_module(module_name, translated_units, module_info)

    # ------------------------------------------------------------------ #
    # Translation validation                                               #
    # ------------------------------------------------------------------ #

    def _validate_translation(self, code: str, expected_name: str) -> List[str]:
        """
        Run lightweight sanity checks on the translated code.

        Returns a list of warning strings (empty = all clear).
        Does NOT raise — callers log the warnings and continue.
        """
        warnings: List[str] = []

        # 1. Syntax check
        try:
            ast.parse(code)
        except SyntaxError as exc:
            warnings.append(f"Syntax error in translated code: {exc}")

        # 2. Expected function name present
        if expected_name and f"def {expected_name}" not in code.lower():
            # Try case-insensitive
            if not re.search(rf"\bdef\s+{re.escape(expected_name)}\b", code, re.I):
                warnings.append(
                    f"Function '{expected_name}' not found in translation — "
                    "name may have been altered by the model"
                )

        # 3. JAX config present
        if "jax_enable_x64" not in code:
            warnings.append("jax.config.update('jax_enable_x64', True) not found — float64 may not work on GPU")

        # 4. Bare numpy import
        if re.search(r"^\s*import\s+numpy\b", code, re.M):
            warnings.append("'import numpy' found — use 'import jax.numpy as jnp' only")

        # 5. In-place mutation
        if re.search(r"\w+\[\w.*\]\s*=", code):
            warnings.append(
                "Possible in-place array mutation detected (arr[i] = v) — "
                "use arr.at[i].set(v) for JIT compatibility"
            )

        # 6. Python for-loop in function body
        if re.search(r"^\s+for\s+\w+\s+in\s+range\s*\(", code, re.M):
            warnings.append(
                "Python for-loop detected inside function body — "
                "consider jax.lax.fori_loop or jnp vectorised ops"
            )

        return warnings

    # ------------------------------------------------------------------ #
    # Module / unit helpers                                                #
    # ------------------------------------------------------------------ #

    def _get_module_units(self, module_name: str) -> List[Dict[str, Any]]:
        if not self.translation_units:
            return []
        units = self.translation_units.get("translation_units", [])
        module_units = [
            u for u in units
            if u.get("module_name", "").lower() == module_name.lower()
        ]
        module_units.sort(key=lambda u: u.get("line_start", 0))
        return module_units

    def _get_module_dependencies(self, module_name: str) -> Dict[str, Any]:
        deps: Dict[str, Any] = {"uses": [], "used_by": []}
        if not self.analysis_results:
            return deps
        all_deps = self.analysis_results.get("parsing", {}).get("dependencies", {})
        if not all_deps:
            all_deps = self.analysis_results.get("dependencies", {}).get("analysis", {})
        if isinstance(all_deps, dict):
            if module_name in all_deps:
                deps["uses"] = all_deps[module_name]
            deps["used_by"] = [m for m, md in all_deps.items() if module_name in md]
        return deps

    # ------------------------------------------------------------------ #
    # Path / file helpers                                                  #
    # ------------------------------------------------------------------ #

    def _extract_source_directory(self, file_path: str) -> Optional[str]:
        if not file_path:
            return None
        _generic = {"src", "source", "lib", "include", ""}
        parts = Path(file_path).parts
        for part in reversed(parts[:-1]):
            if part not in _generic and not part.startswith("/"):
                return part
        return None

    def _remap_fortran_path(self, original_path: str) -> Path:
        path_obj = Path(original_path)
        try:
            if path_obj.exists():
                return path_obj
        except (PermissionError, OSError):
            pass
        if self.fortran_root:
            parts = path_obj.parts
            for i in range(len(parts)):
                if parts[i:]:
                    candidate = self.fortran_root / Path(*parts[i:])
                    try:
                        if candidate.exists():
                            return candidate
                    except (PermissionError, OSError):
                        pass
            filename_path = self.fortran_root / path_obj.name
            if filename_path.exists():
                return filename_path
        return path_obj

    def _load_json(self, json_path: Path) -> Dict[str, Any]:
        with open(json_path) as f:
            return json.load(f)

    def _extract_module_info(self, module_name: str) -> Optional[Dict[str, Any]]:
        if not self.analysis_results:
            return None
        modules = self.analysis_results.get("parsing", {}).get("modules", {})
        if not modules:
            modules = self.analysis_results.get("modules", {})
        for mod_name, mod_data in modules.items():
            if mod_name.lower() == module_name.lower():
                return mod_data
        return None

    def _load_reference_patterns(self) -> Dict[str, str]:
        patterns: Dict[str, str] = {}
        if self.reference_dir and self.reference_dir.exists():
            for py_file in sorted(self.reference_dir.rglob("*.py"))[:5]:
                try:
                    patterns[py_file.stem] = py_file.read_text()
                except (IOError, OSError):
                    pass
        return patterns

    def _get_reference_pattern(self) -> str:
        if self.reference_patterns:
            first_key = next(iter(self.reference_patterns))
            lines = self.reference_patterns[first_key].split("\n")[:100]
            return "\n".join(lines) + "\n\n# ... (abbreviated for context) ..."
        return (
            "# Reference pattern not available.\n"
            "# Apply JAX best practices: pure functions, NamedTuples, jnp ops.\n"
        )

    # ------------------------------------------------------------------ #
    # Response parsing                                                     #
    # ------------------------------------------------------------------ #

    def _parse_translation_response(
        self,
        response: str,
        module_name: str,
        module_info: Optional[Dict[str, Any]] = None,
    ) -> TranslationResult:
        physics_code = self._extract_code(response)
        notes = ""
        if "```" in response:
            notes = response[: response.find("```")].strip()
        return TranslationResult(
            module_name=module_name,
            physics_code=physics_code if physics_code else response,
            source_directory=(
                self._extract_source_directory(module_info.get("file_path", ""))
                if module_info else None
            ),
            translation_notes=notes,
        )

    def _extract_code(self, response: str) -> str:
        if "```python" in response:
            start = response.find("```python") + 9
            end   = response.find("```", start)
            return response[start:end].strip()
        if "```" in response:
            start = response.find("```") + 3
            end   = response.find("```", start)
            return response[start:end].strip()
        return response.strip()
