"""
GitCoordinator — auto-commit translated modules for audit trail and Slurm coordination.

After each module successfully translates + passes parity, the coordinator:
1. Stages the JAX source, test, and RALPH log files.
2. Creates a structured commit: ``translate({module}): Fortran→JAX [PASS] parity=PASS``
3. Prepends an entry to CHANGELOG.md.

This gives a clean ``git log --oneline`` audit trail and lets parallel Slurm jobs
coordinate via the shared git repo without file-lock conflicts.

If git is not available or ``auto_commit=False``, all calls are silent no-ops.
"""

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


class GitCoordinator:
    """Commit translated module outputs to git after each successful pipeline step."""

    def __init__(
        self,
        repo_root: Path,
        output_dir: Path,
        auto_commit: bool = False,
    ) -> None:
        """
        Args:
            repo_root:   Root of the git repository (used for git commands).
            output_dir:  Pipeline output directory (staged files live here).
            auto_commit: If False all methods are silent no-ops.
        """
        self.repo_root = Path(repo_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.auto_commit = auto_commit
        self._git_ok: Optional[bool] = None  # cached result of is_git_repo()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_git_repo(self) -> bool:
        """Return True if repo_root is inside a git repository."""
        if self._git_ok is not None:
            return self._git_ok
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            cwd=self.repo_root,
        )
        self._git_ok = result.returncode == 0
        return self._git_ok

    def commit_module(
        self,
        module_name: str,
        status: str = "success",
        parity_result: str = "PASS",
        jax_src_dir: Optional[Path] = None,
        tests_dir: Optional[Path] = None,
        reports_dir: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Stage and commit translated module outputs.

        Args:
            module_name:   Module name (e.g. ``clm_varcon``).
            status:        Overall step status (``success`` / ``failed``).
            parity_result: Parity outcome string (``PASS`` / ``FAIL``).
            jax_src_dir:   Directory containing ``{module_name}.py``.
            tests_dir:     Directory containing ``test_{module_name}.py``.
            reports_dir:   Directory containing RALPH log and repair docs.

        Returns:
            Git commit hash on success, ``None`` if skipped.
        """
        if not self.auto_commit:
            return None
        if not self.is_git_repo():
            logger.warning("git_coordinator: not a git repo, skipping commit for %s", module_name)
            return None

        staged = self._stage_files(
            module_name=module_name,
            jax_src_dir=jax_src_dir or (self.output_dir / "jax" / "src"),
            tests_dir=tests_dir or (self.output_dir / "tests"),
            reports_dir=reports_dir or (self.output_dir / "reports"),
        )

        if not staged:
            logger.debug("git_coordinator: nothing to stage for %s", module_name)
            return None

        message = (
            f"translate({module_name}): Fortran→JAX [{status}] parity={parity_result}\n\n"
            f"Auto-committed by TransJAX pipeline\n"
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}"
        )

        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
        )

        if result.returncode != 0:
            logger.warning(
                "git_coordinator: commit failed for %s: %s",
                module_name,
                result.stderr.strip(),
            )
            return None

        # Extract commit hash from output
        commit_hash: Optional[str] = None
        for line in result.stdout.splitlines():
            if line.startswith("["):
                parts = line.split()
                if len(parts) >= 2:
                    commit_hash = parts[1].rstrip("]")
                break

        console.print(
            f"  [dim]git commit {commit_hash or '?'} — {module_name} ({parity_result})[/dim]"
        )
        return commit_hash

    def append_changelog(
        self,
        module_name: str,
        status: str,
        notes: str = "",
    ) -> None:
        """
        Prepend an entry to CHANGELOG.md (creates the file if absent).

        Args:
            module_name: Module that completed.
            status:      ``success`` / ``failed``.
            notes:       Optional extra detail (e.g. parity tolerance, iterations).
        """
        if not self.auto_commit:
            return

        changelog_path = self.repo_root / "CHANGELOG.md"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        icon = "✓" if status == "success" else "✗"
        entry = (
            f"- [{now}] {icon} `{module_name}` — {status}"
            + (f" ({notes})" if notes else "")
            + "\n"
        )

        if changelog_path.exists():
            existing = changelog_path.read_text(encoding="utf-8")
            # Insert after the first heading line
            lines = existing.splitlines(keepends=True)
            insert_at = 0
            for i, line in enumerate(lines):
                if line.startswith("## "):
                    insert_at = i + 1
                    break
            lines.insert(insert_at, entry)
            changelog_path.write_text("".join(lines), encoding="utf-8")
        else:
            changelog_path.write_text(
                "# TransJAX Translation Changelog\n\n"
                "## Modules\n\n"
                + entry,
                encoding="utf-8",
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _stage_files(
        self,
        module_name: str,
        jax_src_dir: Path,
        tests_dir: Path,
        reports_dir: Path,
    ) -> bool:
        """Stage relevant files; return True if at least one was staged."""
        candidates = [
            jax_src_dir / f"{module_name}.py",
            jax_src_dir / f"{module_name}_params.py",
            tests_dir / f"test_{module_name}.py",
            reports_dir / f"{module_name}_ralph_log.json",
        ]
        # Also pick up any parity docs
        parity_dir = self.output_dir / "parity" / module_name
        if parity_dir.exists():
            candidates += list(parity_dir.glob("*.md")) + list(parity_dir.glob("*.json"))

        staged_any = False
        for path in candidates:
            if path.exists():
                result = subprocess.run(
                    ["git", "add", str(path)],
                    capture_output=True,
                    cwd=self.repo_root,
                )
                if result.returncode == 0:
                    staged_any = True

        # Also stage CHANGELOG.md if it was updated
        changelog = self.repo_root / "CHANGELOG.md"
        if changelog.exists():
            subprocess.run(
                ["git", "add", str(changelog)],
                capture_output=True,
                cwd=self.repo_root,
            )

        return staged_any
