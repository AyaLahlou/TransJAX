"""
SlurmRunner — generate and submit one Slurm job per module.

Each job:
1. Activates the conda / module environment.
2. Creates a named tmux session (``transjax-{module}``).
3. Runs ``transjax pipeline --modules {module} --use-tmux`` inside it.
4. Signals tmux done so the job script can exit cleanly.

Usage
-----
runner = SlurmRunner(
    output_dir=Path("pipeline_out"),
    fortran_dir=Path("fortran_src"),
    analysis_dir=Path("transjax_analysis"),
    partition="gpu",
)
runner.generate_all(modules=["clm_varcon", "clm_cntype"])   # write sbatch files
runner.submit_all(modules=["clm_varcon"])                   # submit to scheduler
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

_SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=transjax-{module}
#SBATCH --output={log_dir}/{module}-%j.out
#SBATCH --error={log_dir}/{module}-%j.err
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
{extra_directives}

# ── Environment setup ──────────────────────────────────────────────────────
{env_setup}

# ── Verify tools ──────────────────────────────────────────────────────────
command -v tmux   >/dev/null 2>&1 || {{ echo "ERROR: tmux not found"; exit 1; }}
command -v claude >/dev/null 2>&1 || {{ echo "ERROR: claude CLI not found"; exit 1; }}
command -v transjax >/dev/null 2>&1 || {{ echo "ERROR: transjax not found"; exit 1; }}

# ── Start tmux session ────────────────────────────────────────────────────
SESSION="transjax-{module}"
tmux new-session -d -s "$SESSION" -x 220 -y 50

# ── Run pipeline inside tmux ───────────────────────────────────────────────
CMD="{pipeline_cmd}"
tmux send-keys -t "$SESSION" "$CMD; tmux wait-for -S done-{module}" Enter

# ── Wait for completion ────────────────────────────────────────────────────
tmux wait-for done-{module}

echo "TransJAX pipeline finished for module: {module}"
"""

_DEFAULT_ENV_SETUP = """\
# Activate your environment — edit as needed for your cluster:
# module load anaconda3
# conda activate transjax
# OR
# source /path/to/transjax_venv/bin/activate
"""


class SlurmRunner:
    """Generate and optionally submit Slurm jobs for TransJAX module translation."""

    def __init__(
        self,
        output_dir: Path,
        fortran_dir: Path,
        analysis_dir: Path,
        partition: str = "cpu",
        time_limit: str = "2:00:00",
        mem: str = "16G",
        cpus: int = 4,
        gcm_model_name: str = "generic ESM",
        extra_directives: str = "",
        env_setup: Optional[str] = None,
    ) -> None:
        """
        Args:
            output_dir:        Pipeline output directory (passed to transjax pipeline).
            fortran_dir:       Fortran source directory.
            analysis_dir:      Directory containing ``translation_order.json``.
            partition:         Slurm partition name.
            time_limit:        Wall-clock limit (``H:MM:SS``).
            mem:               Memory per node (e.g. ``16G``).
            cpus:              CPUs per task.
            gcm_model_name:    GCM model name passed to pipeline.
            extra_directives:  Additional ``#SBATCH`` lines (newline-separated).
            env_setup:         Shell snippet to activate the environment.
                               Defaults to a commented placeholder.
        """
        self.output_dir = Path(output_dir).resolve()
        self.fortran_dir = Path(fortran_dir).resolve()
        self.analysis_dir = Path(analysis_dir).resolve()
        self.partition = partition
        self.time_limit = time_limit
        self.mem = mem
        self.cpus = cpus
        self.gcm_model_name = gcm_model_name
        self.extra_directives = extra_directives
        self.env_setup = env_setup or _DEFAULT_ENV_SETUP

        self._scripts_dir = self.output_dir / "slurm_scripts"
        self._log_dir = self.output_dir / "slurm_logs"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_sbatch(self, module_name: str) -> str:
        """Return the rendered sbatch script for one module (does not write file)."""
        pipeline_cmd = (
            f"transjax pipeline"
            f" --fortran-dir {self.fortran_dir}"
            f" --analysis-dir {self.analysis_dir}"
            f" --output-dir {self.output_dir}"
            f" --modules {module_name}"
            f" --gcm-model \"{self.gcm_model_name}\""
            f" --use-tmux"
        )
        return _SBATCH_TEMPLATE.format(
            module=module_name,
            log_dir=self._log_dir,
            partition=self.partition,
            time_limit=self.time_limit,
            mem=self.mem,
            cpus=self.cpus,
            extra_directives=self.extra_directives,
            env_setup=self.env_setup,
            pipeline_cmd=pipeline_cmd,
        )

    def generate_all(self, modules: List[str]) -> List[Path]:
        """Write one ``.sbatch`` file per module; return list of paths."""
        self._scripts_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        paths: List[Path] = []
        for module in modules:
            script = self.generate_sbatch(module)
            path = self._scripts_dir / f"{module}.sbatch"
            path.write_text(script, encoding="utf-8")
            paths.append(path)
            console.print(f"  [dim]wrote {path}[/dim]")

        console.print(f"[green]Generated {len(paths)} sbatch script(s) → {self._scripts_dir}[/green]")
        return paths

    def submit(self, module_name: str) -> str:
        """Submit one module's sbatch script; return the Slurm job ID."""
        script_path = self._scripts_dir / f"{module_name}.sbatch"
        if not script_path.exists():
            # Generate on the fly
            self.generate_all([module_name])

        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed for {module_name}:\n{result.stderr.strip()}"
            )

        # Output format: "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        console.print(f"  [green]Submitted {module_name} → Slurm job {job_id}[/green]")
        return job_id

    def submit_all(self, modules: List[str]) -> Dict[str, str]:
        """Submit all modules; return mapping of ``{module: job_id}``."""
        self.generate_all(modules)
        job_ids: Dict[str, str] = {}
        for module in modules:
            try:
                job_ids[module] = self.submit(module)
            except RuntimeError as exc:
                logger.error("Failed to submit %s: %s", module, exc)
                job_ids[module] = "ERROR"
        return job_ids
