"""
Translator Agent for converting Fortran to JAX.

Translates a complete Fortran module to JAX in one pass by:
1. Pre-parsing all subroutine/function interfaces (no hallucinated args/dtypes)
2. Building a structured interface summary for the prompt
3. Either:
   a. SDK path: sending the full source + interface summary in one LLM call
   b. Tmux path: writing a task file and running a full Claude Code agentic
      session (--dangerously-skip-permissions) that reads/writes files itself
4. Validating the output with ast.parse + heuristic checks
"""

import ast
import json
import logging
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console

from transjax.agents.base_agent import BaseAgent
from transjax.agents.prompts.translation_prompts import (
    MODULE_TASK_PROMPT,
    MODULE_TRANSLATION_PROMPT,
    TRANSLATOR_SYSTEM_PROMPT,
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
    Translates a complete Fortran ESM module to JAX/Python.

    Always translates the full module in one pass (no unit-by-unit splitting).

    Two execution paths:
    - SDK path (default): one LLM call via Anthropic SDK with the full Fortran
      source + pre-parsed interface contracts embedded in the prompt.
    - Tmux path (use_tmux=True): writes a task file and launches a full Claude
      Code agentic session (--dangerously-skip-permissions) that uses Read/
      Write/Bash tools to translate, validate, and signal completion.
    """

    def __init__(
        self,
        fortran_root: Optional[Path] = None,
        gcm_model_name: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_tmux: bool = False,
        tmux_poll_interval: float = 2.0,
        tmux_timeout: float = 900.0,
    ):
        llm_config = get_llm_config()
        super().__init__(
            name="Translator",
            role="Fortran to JAX code translator",
            model=model or llm_config.get("model", "claude-sonnet-4-6"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 48000),
            use_tmux=use_tmux,
            tmux_poll_interval=tmux_poll_interval,
            tmux_timeout=tmux_timeout,
        )

        self.fortran_root = fortran_root
        self.gcm_model_name = gcm_model_name or "unspecified"
        self.analysis_results: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load_analysis(self, analysis_results_path: Path) -> None:
        """Load analysis results for module info lookup (file paths, etc.)."""
        with open(analysis_results_path) as f:
            self.analysis_results = json.load(f)

    def translate_module(
        self,
        module_name: str,
        fortran_file: Optional[Path] = None,
        analysis: Optional[Any] = None,
        output_dir: Optional[Path] = None,
    ) -> TranslationResult:
        """Translate a complete Fortran module to JAX.

        Uses a full Claude Code agentic session when ``use_tmux=True``,
        otherwise sends one SDK call with the full source embedded.
        """
        console.print(f"\n[bold cyan]Translating {module_name} to JAX[/bold cyan]")

        fortran_path = self._resolve_fortran_path(module_name, fortran_file)
        console.print(f"[dim]Reading from: {fortran_path}[/dim]")
        fortran_code = Path(fortran_path).read_text()

        all_ifaces = self._parse_all_interfaces(fortran_code)
        n_routines = len(all_ifaces)
        console.print(f"[dim]Found {n_routines} routine(s) to translate[/dim]")
        interface_summary = self._format_interface_summary(all_ifaces)

        module_info = self._extract_module_info(module_name)

        if self.use_tmux:
            result = self._translate_via_tmux(
                module_name=module_name,
                fortran_path=fortran_path,
                interface_summary=interface_summary,
                n_routines=n_routines,
                output_dir=output_dir,
            )
        else:
            result = self._translate_via_sdk(
                module_name=module_name,
                fortran_code=fortran_code,
                interface_summary=interface_summary,
                n_routines=n_routines,
                module_info=module_info,
                all_ifaces=all_ifaces,
            )

        result.source_directory = self._extract_source_directory(
            module_info.get("file_path", "") if module_info else ""
        )
        console.print("[green]✓ Translation complete![/green]")

        if output_dir:
            result.save(output_dir)
        return result

    # ------------------------------------------------------------------ #
    # SDK translation path                                                 #
    # ------------------------------------------------------------------ #

    def _translate_via_sdk(
        self,
        module_name: str,
        fortran_code: str,
        interface_summary: str,
        n_routines: int,
        module_info: Optional[Dict[str, Any]],
        all_ifaces: List[Dict[str, Any]],
    ) -> TranslationResult:
        """One-shot LLM call with full Fortran source embedded in prompt."""
        prompt = MODULE_TRANSLATION_PROMPT.format(
            gcm_model_name=self.gcm_model_name,
            module_name=module_name,
            source_file=module_info.get("file_path", "unknown") if module_info else "unknown",
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

        for iface in all_ifaces:
            name = iface.get("subroutine_name", "")
            if name:
                for w in self._validate_translation(code, name):
                    console.print(f"[yellow]  ⚠ [{name}] {w}[/yellow]")

        notes = ""
        if "```" in response:
            notes = response[: response.find("```")].strip()

        return TranslationResult(
            module_name=module_name,
            physics_code=code if code else response,
            translation_notes=notes,
        )

    # ------------------------------------------------------------------ #
    # Tmux / agentic translation path                                      #
    # ------------------------------------------------------------------ #

    def _translate_via_tmux(
        self,
        module_name: str,
        fortran_path: Path,
        interface_summary: str,
        n_routines: int,
        output_dir: Optional[Path],
    ) -> TranslationResult:
        """
        Launch a full Claude Code agentic session to translate the module.

        Claude reads the Fortran file itself using its Read tool, writes the
        translated Python to ``output_file``, validates syntax, then writes
        ``DONE`` (or ``FAILED: <reason>``) to ``sentinel_file``.
        """
        if not self._tmux_runner:
            raise RuntimeError(
                "use_tmux=True but no tmux session set. "
                "Call set_tmux_session() before translate_module()."
            )

        # Determine output and sentinel file paths
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            work = output_dir
        else:
            work = self.tmux_work_dir or Path(tempfile.mkdtemp(prefix=f"transjax_{module_name}_"))

        output_file = work / f"{module_name}.py"
        sentinel_file = work / f"{module_name}.sentinel"
        task_file = work / f"{module_name}.task.md"

        # Clean stale files from a previous attempt
        output_file.unlink(missing_ok=True)
        sentinel_file.unlink(missing_ok=True)

        task = MODULE_TASK_PROMPT.format(
            gcm_model_name=self.gcm_model_name,
            module_name=module_name,
            fortran_file=str(fortran_path),
            output_file=str(output_file),
            sentinel_file=str(sentinel_file),
            interface_summary=interface_summary,
            n_routines=n_routines,
        )
        task_file.write_text(task, encoding="utf-8")

        console.print(
            f"[dim]Starting Claude Code session "
            f"(sentinel: {sentinel_file.name})[/dim]"
        )
        self._tmux_runner.run_task(task_file, sentinel_file)

        if not output_file.exists():
            raise RuntimeError(
                f"Translation task signalled DONE but output file not found: {output_file}"
            )

        code = output_file.read_text(encoding="utf-8")
        # Clean task file; keep output and sentinel for debugging
        task_file.unlink(missing_ok=True)

        return TranslationResult(
            module_name=module_name,
            physics_code=code,
        )

    # ------------------------------------------------------------------ #
    # Interface pre-parser                                                 #
    # ------------------------------------------------------------------ #

    def _parse_all_interfaces(self, fortran_code: str) -> List[Dict[str, Any]]:
        """
        Find every subroutine/function in *fortran_code* and parse its interface.

        Returns a list of interface dicts (one per routine) in source order.
        """
        ifaces: List[Dict[str, Any]] = []
        lines = fortran_code.split("\n")

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
                routine_src = "\n".join(lines[start:i])
                iface = self._parse_fortran_interface(routine_src)
                if iface["subroutine_name"]:
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
            parts.append("")
        return "\n".join(parts)

    def _parse_fortran_interface(self, fortran_code: str) -> Dict[str, Any]:
        """
        Extract the subroutine/function interface from Fortran source.

        Returns a dict with: subroutine_name, arg_names, args, inputs, inouts,
        outputs, interface_table, python_signature, return_fields.
        """
        def _jax_type(fortran_type: str) -> str:
            for pat, jax in _FORTRAN_TO_JAX:
                if pat.search(fortran_type):
                    return jax
            return "jnp.float64"

        # Join Fortran continuation lines
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

        # Find subroutine / function declaration
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

        args: Dict[str, Dict[str, Any]] = {
            a: {"intent": "in", "fortran_type": "real(r8)",
                "jax_type": "jnp.float64", "shape": "", "is_array": False}
            for a in arg_names
        }

        def _split_vars(var_part: str) -> List[str]:
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

        for line in joined:
            stripped = line.strip()
            if stripped.startswith("!") or "::" not in line:
                continue

            type_part, var_part = line.split("::", 1)
            type_part = type_part.strip()
            var_part  = var_part.strip()

            intent_m = _RE_INTENT.search(type_part)
            intent = intent_m.group(1).lower() if intent_m else "in"

            dim_m = _RE_DIMENSION.search(type_part)
            global_shape = dim_m.group(1) if dim_m else ""

            type_clean = re.split(
                r",\s*(?:intent|dimension|allocatable|pointer|target|optional)\b",
                type_part, maxsplit=1, flags=re.I,
            )[0].strip()

            jtype = _jax_type(type_clean)

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

        inputs  = [a for a in arg_names if args[a]["intent"] == "in"]
        inouts  = [a for a in arg_names if args[a]["intent"] == "inout"]
        outputs = [a for a in arg_names if args[a]["intent"] == "out"]

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

        py_args = [
            "{}: {}".format(a, "jax.Array" if args[a]["is_array"] else args[a]["jax_type"])
            for a in arg_names if args[a]["intent"] in ("in", "inout")
        ]
        ret_name = (name.capitalize() if name else "Unknown") + "Result"
        python_signature = f"def {name or 'unknown'}({', '.join(py_args)}) -> {ret_name}:"

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
    # Translation validation                                               #
    # ------------------------------------------------------------------ #

    def _validate_translation(self, code: str, expected_name: str) -> List[str]:
        """
        Lightweight sanity checks on translated code.

        Returns a list of warning strings (empty = all clear).
        Does NOT raise — callers log warnings and continue.
        """
        warnings: List[str] = []

        try:
            ast.parse(code)
        except SyntaxError as exc:
            warnings.append(f"Syntax error in translated code: {exc}")

        if expected_name and not re.search(rf"\bdef\s+{re.escape(expected_name)}\b", code, re.I):
            warnings.append(
                f"Function '{expected_name}' not found in translation — "
                "name may have been altered by the model"
            )

        if "jax_enable_x64" not in code:
            warnings.append(
                "jax.config.update('jax_enable_x64', True) not found — float64 may not work on GPU"
            )

        if re.search(r"^\s*import\s+numpy\b", code, re.M):
            warnings.append("'import numpy' found — use 'import jax.numpy as jnp' only")

        if re.search(r"\w+\[\w.*\]\s*=", code):
            warnings.append(
                "Possible in-place array mutation detected (arr[i] = v) — "
                "use arr.at[i].set(v) for JIT compatibility"
            )

        if re.search(r"^\s+for\s+\w+\s+in\s+range\s*\(", code, re.M):
            warnings.append(
                "Python for-loop detected inside function body — "
                "consider jax.lax.fori_loop or jnp vectorised ops"
            )

        return warnings

    # ------------------------------------------------------------------ #
    # Path / file helpers                                                  #
    # ------------------------------------------------------------------ #

    def _resolve_fortran_path(
        self, module_name: str, fortran_file: Optional[Path]
    ) -> Path:
        """Return the Fortran source path, remapping via analysis results if needed."""
        if fortran_file:
            return fortran_file
        module_info = self._extract_module_info(module_name)
        if module_info:
            return self._remap_fortran_path(module_info["file_path"])
        raise ValueError(
            f"Module '{module_name}' not found in analysis results and no fortran_file provided"
        )

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

    # ------------------------------------------------------------------ #
    # Response parsing                                                     #
    # ------------------------------------------------------------------ #

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
