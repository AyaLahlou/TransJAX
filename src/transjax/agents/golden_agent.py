"""
GoldenAgent — trusted-run golden data generator for Ftest suites.

For each compiled Fortran test driver produced by FtestAgent, GoldenAgent:

  1. Reads the Ftest report (ftest_report.json) to obtain subroutine metadata
     and the parsed calling interface (intent-in variable names and types).
  2. Asks Claude — acting as an ESM domain expert — to propose a set of
     physically representative input test cases for the subroutine.
  3. Runs each case through the compiled Fortran driver binary (subprocess)
     and captures the real KEY=VALUE output.
  4. Writes a JSON golden file to tests/golden/<module>_<subroutine>.json.

Golden file schema
------------------
{
  "subroutine": "CanopyFluxes",
  "module": "CanopyFluxesMod",
  "generated_at": "2026-03-26T12:00:00",
  "ftest_output_dir": "/path/to/ftest_output",
  "n_cases": 5,
  "cases": [
    {
      "id": "tropical_midday",
      "description": "...",
      "inputs":  {"tair": 303.15, ...},
      "outputs": {"sensible_heat": 85.3, ...},
      "stdout_raw": "sensible_heat=85.3\\n..."
    }
  ]
}

Usage with pytest
-----------------
Tests can load their golden file and compare live driver output against it:

    from pathlib import Path
    import json

    GOLDEN = json.loads(
        (Path(__file__).parent / "golden" / "CanopyFluxesMod_CanopyFluxes.json")
        .read_text()
    )

    def test_matches_golden(run_driver, DRIVER):
        for case in GOLDEN["cases"]:
            out = run_driver(DRIVER, case["inputs"])
            for key, expected in case["outputs"].items():
                assert abs(out[key] - expected) < 1e-4 * abs(expected) + 1e-10
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from transjax.agents.base_agent import BaseAgent
from transjax.agents.prompts.golden_prompts import (
    GOLDEN_INPUT_CASES_PROMPT,
    GOLDEN_SYSTEM_PROMPT,
)
from transjax.agents.utils.config_loader import get_llm_config

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GoldenCase:
    """One input/output pair captured from a trusted driver run."""
    id: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    stdout_raw: str = ""


@dataclass
class GoldenData:
    """All golden cases for a single subroutine."""
    subroutine_name: str
    module_name: str
    generated_at: str
    ftest_output_dir: str
    cases: List[GoldenCase] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subroutine": self.subroutine_name,
            "module": self.module_name,
            "generated_at": self.generated_at,
            "ftest_output_dir": self.ftest_output_dir,
            "n_cases": len(self.cases),
            "cases": [
                {
                    "id": c.id,
                    "description": c.description,
                    "inputs": c.inputs,
                    "outputs": c.outputs,
                    "stdout_raw": c.stdout_raw,
                }
                for c in self.cases
            ],
        }


@dataclass
class GoldenRunResult:
    """Aggregated outcome of a full golden-data generation run."""
    ftest_output_dir: Path
    golden_dir: Path
    subroutines_attempted: int = 0
    golden_written: int = 0
    skipped: int = 0
    errors: int = 0
    golden_files: List[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class GoldenAgent(BaseAgent):
    """
    Generates golden (trusted-run) data for all Ftest subroutine drivers.

    Reads the Ftest report, uses Claude's ESM domain knowledge to propose
    physically representative input cases, runs the compiled Fortran binaries,
    and records inputs + real outputs in JSON golden files.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        llm_config = get_llm_config()
        super().__init__(
            name="GoldenAgent",
            role=(
                "Expert ESM scientist and software engineer who generates physically "
                "representative golden test data for Fortran subroutine test drivers, "
                "covering typical, extreme, and edge-case regimes."
            ),
            model=model or llm_config.get("model", "claude-sonnet-4-6"),
            temperature=temperature if temperature is not None else llm_config.get("temperature", 0.0),
            max_tokens=max_tokens or llm_config.get("max_tokens", 48000),
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        ftest_output_dir: Path,
        n_cases: int = 5,
        gcm_model_name: str = "generic ESM",
        verbose: bool = False,
    ) -> GoldenRunResult:
        """
        Generate golden data for all compiled Ftest drivers.

        Args:
            ftest_output_dir: Root of the directory produced by ``transjax ftest``.
                              Must contain ``ftest_report.json`` and compiled
                              binaries under ``drivers/bin/``.
            n_cases:          Number of input/output cases to capture per subroutine.
            gcm_model_name:   Name of the ESM (e.g. "CTSM", "MOM6").  Injected
                              into prompts so Claude applies model-specific knowledge.
            verbose:          Enable DEBUG-level logging.

        Returns:
            GoldenRunResult summarising what was written.
        """
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        report_path = ftest_output_dir / "ftest_report.json"
        if not report_path.exists():
            raise FileNotFoundError(
                f"ftest_report.json not found in {ftest_output_dir}.\n"
                "Run `transjax ftest` first to generate the test framework."
            )

        report = json.loads(report_path.read_text())
        bin_dir = ftest_output_dir / "drivers" / "bin"
        golden_dir = ftest_output_dir / "tests" / "golden"
        golden_dir.mkdir(parents=True, exist_ok=True)

        run_result = GoldenRunResult(
            ftest_output_dir=ftest_output_dir,
            golden_dir=golden_dir,
        )

        subroutines = [s for s in report.get("subroutines", []) if s["status"] == "ok"]
        run_result.subroutines_attempted = len(subroutines)

        if not subroutines:
            console.print("[yellow]No successfully generated subroutines found in the report.[/yellow]")
            return run_result

        console.print(
            f"[cyan]Golden data generation:[/cyan] {len(subroutines)} subroutine(s), "
            f"{n_cases} cases each"
        )
        console.print(f"[cyan]ESM context:[/cyan]             {gcm_model_name}")
        console.print(f"[cyan]Driver binaries:[/cyan]         {bin_dir}")
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Generating golden data...", total=len(subroutines))

            for sub_info in subroutines:
                sub_name = sub_info["name"]
                mod_name = sub_info["module"]
                interface = sub_info.get("interface")

                progress.update(
                    task,
                    description=f"[cyan]{mod_name}[/cyan]::{sub_name}",
                )

                golden_path = golden_dir / f"{mod_name}_{sub_name}.json"
                driver_exe = bin_dir / f"test_{sub_name}"

                result_ok = self._process_subroutine(
                    sub_name=sub_name,
                    mod_name=mod_name,
                    interface=interface,
                    driver_exe=driver_exe,
                    golden_path=golden_path,
                    ftest_output_dir=ftest_output_dir,
                    n_cases=n_cases,
                    gcm_model_name=gcm_model_name,
                )

                if result_ok is None:
                    run_result.skipped += 1
                elif result_ok:
                    run_result.golden_written += 1
                    run_result.golden_files.append(golden_path)
                else:
                    run_result.errors += 1

                progress.advance(task)

        return run_result

    # -----------------------------------------------------------------------
    # Per-subroutine pipeline
    # -----------------------------------------------------------------------

    def _process_subroutine(
        self,
        sub_name: str,
        mod_name: str,
        interface: Optional[Dict[str, Any]],
        driver_exe: Path,
        golden_path: Path,
        ftest_output_dir: Path,
        n_cases: int,
        gcm_model_name: str,
    ) -> Optional[bool]:
        """
        Generate and capture golden cases for one subroutine.

        Returns True on success, False on error, None if skipped.
        """
        # Check binary exists
        if not driver_exe.exists():
            console.print(
                f"  [yellow]⏭[/yellow]  {sub_name}: binary not found "
                f"({driver_exe.name}) — run `make all` first"
            )
            return None

        # Build variable table for the prompt
        if not interface:
            console.print(
                f"  [yellow]⏭[/yellow]  {sub_name}: no interface in report "
                "(re-run `transjax ftest` to rebuild)"
            )
            return None

        variables_table = self._build_variables_table(interface)
        if not variables_table.strip():
            console.print(
                f"  [yellow]⏭[/yellow]  {sub_name}: no scalar intent-in variables found"
            )
            return None

        try:
            # 1 — Ask Claude for representative input cases
            input_cases = self._generate_input_cases(
                sub_name=sub_name,
                mod_name=mod_name,
                variables_table=variables_table,
                n_cases=n_cases,
                gcm_model_name=gcm_model_name,
            )

            # 2 — Run each case against the compiled binary
            golden_cases: List[GoldenCase] = []
            failed_runs = 0
            for case_dict in input_cases:
                captured = self._run_case(
                    driver_exe=driver_exe,
                    case_id=case_dict.get("id", f"case_{len(golden_cases)}"),
                    description=case_dict.get("description", ""),
                    inputs=case_dict.get("inputs", {}),
                )
                if captured is not None:
                    golden_cases.append(captured)
                else:
                    failed_runs += 1

            if not golden_cases:
                console.print(
                    f"  [red]✗[/red]  {sub_name}: all {n_cases} driver runs failed"
                )
                return False

            # 3 — Write golden JSON
            golden_data = GoldenData(
                subroutine_name=sub_name,
                module_name=mod_name,
                generated_at=datetime.now(timezone.utc).isoformat(),
                ftest_output_dir=str(ftest_output_dir),
                cases=golden_cases,
            )
            golden_path.write_text(json.dumps(golden_data.to_dict(), indent=2))

            status = "[green]✓[/green]"
            note = (
                f"{len(golden_cases)}/{n_cases} cases"
                + (f", {failed_runs} run(s) failed" if failed_runs else "")
            )
            console.print(f"  {status}  {sub_name}  [dim]{note} → {golden_path.name}[/dim]")
            return True

        except Exception as exc:
            logger.error("Golden generation failed for %s: %s", sub_name, exc)
            console.print(f"  [red]✗[/red]  {sub_name}: {exc}")
            return False

    # -----------------------------------------------------------------------
    # Input-case generation (Claude)
    # -----------------------------------------------------------------------

    def _generate_input_cases(
        self,
        sub_name: str,
        mod_name: str,
        variables_table: str,
        n_cases: int,
        gcm_model_name: str,
    ) -> List[Dict[str, Any]]:
        """Ask Claude for representative input cases and parse the JSON array."""
        prompt = GOLDEN_INPUT_CASES_PROMPT.format(
            subroutine_name=sub_name,
            module_name=mod_name,
            gcm_model_name=gcm_model_name,
            n_cases=n_cases,
            variables_table=variables_table,
        )
        response = self.query_claude(
            prompt, system_prompt=GOLDEN_SYSTEM_PROMPT, max_tokens=4096
        )
        return self._parse_json_array(response)

    # -----------------------------------------------------------------------
    # Driver execution
    # -----------------------------------------------------------------------

    def _run_case(
        self,
        driver_exe: Path,
        case_id: str,
        description: str,
        inputs: Dict[str, Any],
    ) -> Optional[GoldenCase]:
        """
        Run the compiled Fortran driver with *inputs* and return a GoldenCase.

        Returns None if the driver exits non-zero or times out.
        """
        namelist_input = self._build_namelist(inputs)
        try:
            proc = subprocess.run(
                [str(driver_exe)],
                input=namelist_input,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Driver %s timed out for case '%s'", driver_exe.name, case_id)
            return None

        if proc.returncode != 0:
            logger.warning(
                "Driver %s exited %d for case '%s': %s",
                driver_exe.name,
                proc.returncode,
                case_id,
                proc.stderr.strip(),
            )
            return None

        outputs = self._parse_kv_output(proc.stdout)
        return GoldenCase(
            id=case_id,
            description=description,
            inputs=inputs,
            outputs=outputs,
            stdout_raw=proc.stdout,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_variables_table(interface: Dict[str, Any]) -> str:
        """Format a human-readable table of scalar intent-in/inout variables."""
        rows = []
        for category in ("intent_in", "intent_inout"):
            for var in interface.get(category, []):
                if var.get("dimensions", "").strip():
                    continue  # skip arrays
                name = var.get("name", "?")
                ftype = var.get("type", "real")
                default = var.get("default_value", "")
                desc = var.get("description", "")
                rows.append(f"  {name:<20} {ftype:<20} default={default:<12} {desc}")
        return "\n".join(rows) if rows else ""

    @staticmethod
    def _build_namelist(inputs: Dict[str, Any]) -> str:
        """Construct a Fortran NAMELIST /inputs/ string from a Python dict."""
        lines = ["&inputs"]
        for key, val in inputs.items():
            if isinstance(val, bool):
                lines.append(f"  {key} = {'T' if val else 'F'},")
            elif isinstance(val, str):
                lines.append(f"  {key} = '{val}',")
            else:
                lines.append(f"  {key} = {val},")
        lines.append("/")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _parse_kv_output(stdout: str) -> Dict[str, Any]:
        """Parse KEY=VALUE lines from driver stdout into a Python dict."""
        outputs: Dict[str, Any] = {}
        for line in stdout.strip().splitlines():
            line = line.strip()
            if "=" not in line:
                continue
            key, _, raw = line.partition("=")
            key = key.strip()
            raw = raw.strip()
            try:
                outputs[key] = int(raw)
            except ValueError:
                try:
                    outputs[key] = float(raw)
                except ValueError:
                    upper = raw.upper()
                    if upper in ("T", ".TRUE.", "TRUE"):
                        outputs[key] = True
                    elif upper in ("F", ".FALSE.", "FALSE"):
                        outputs[key] = False
                    else:
                        outputs[key] = raw
        return outputs

    def _parse_json_array(self, response: str) -> List[Dict[str, Any]]:
        """Extract a JSON array from a Claude response."""
        import re

        # 1. ```json [ … ] ``` block
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
        if m:
            return json.loads(m.group(1))

        # 2. Bare array
        stripped = response.strip()
        if stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass

        # 3. Find first [ … ] block
        start = response.find("[")
        if start != -1:
            depth = 0
            for i, ch in enumerate(response[start:], start):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response[start : i + 1])
                        except json.JSONDecodeError:
                            break

        raise ValueError(
            f"Could not extract JSON array from Claude response "
            f"(first 400 chars):\n{response[:400]}"
        )
