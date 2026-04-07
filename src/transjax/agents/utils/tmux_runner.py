"""
TmuxClaudeRunner — route claude CLI calls through a persistent tmux session.

One tmux session per module (named ``transjax-{module_name}``) is kept alive
for the lifetime of the pipeline run.  Every ``run_query()`` call:

1. Writes the prompt (and optional system prompt) to temp files in work_dir.
2. Builds a shell command that calls ``claude`` non-interactively and
   redirects output to a unique output file.
3. Sends the command to the tmux session via ``tmux send-keys``.
4. Polls a sentinel file for completion (avoids pane-capture escape-code issues).
5. Returns the captured text.

This approach works on HPC clusters where:
- No persistent OAuth session is available in the job environment.
- SSH disconnects must not kill in-flight LLM calls.
- Operators want to ``tmux attach`` to inspect a running session.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)


class TmuxClaudeRunner:
    """Run ``claude`` CLI inside a named tmux session and capture output."""

    def __init__(
        self,
        session_name: str,
        work_dir: Path,
        poll_interval: float = 2.0,
        timeout: float = 900.0,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            session_name: Tmux session name (e.g. ``transjax-clm_varcon``).
            work_dir:     Directory used for prompt/output temp files.
            poll_interval: Seconds between polls of the sentinel file.
            timeout:      Max seconds to wait for a single claude call.
            verbose:      Log the full claude command before running.
        """
        self.session_name = session_name
        self.work_dir = Path(work_dir)
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.verbose = verbose

        self._call_counter = 0
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if not shutil.which("tmux"):
            raise RuntimeError(
                "tmux is not available on PATH. "
                "Install tmux or set use_tmux: false in config."
            )
        if not shutil.which("claude"):
            raise RuntimeError(
                "claude CLI is not available on PATH. "
                "Install it with: npm install -g @anthropic-ai/claude-code"
            )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def ensure_session(self) -> None:
        """Create the tmux session if it does not already exist."""
        result = subprocess.run(
            ["tmux", "has-session", "-t", self.session_name],
            capture_output=True,
        )
        if result.returncode != 0:
            subprocess.run(
                [
                    "tmux", "new-session", "-d",
                    "-s", self.session_name,
                    "-x", "220", "-y", "50",
                ],
                check=True,
            )
            logger.info("Created tmux session: %s", self.session_name)
        else:
            logger.debug("Reusing existing tmux session: %s", self.session_name)

    def close(self) -> None:
        """Kill the tmux session (call at module completion)."""
        subprocess.run(
            ["tmux", "kill-session", "-t", self.session_name],
            capture_output=True,
        )
        logger.info("Killed tmux session: %s", self.session_name)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def run_query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Run one claude query in the tmux session and return the text response.

        Args:
            prompt:        User prompt to send to claude.
            system_prompt: Optional system/persona prompt.

        Returns:
            Claude's response text.

        Raises:
            RuntimeError: If claude exits non-zero or times out.
        """
        self.ensure_session()

        self._call_counter += 1
        call_id = self._call_counter
        prefix = self.work_dir / f".claude_call_{call_id:04d}"

        prompt_file  = prefix.with_suffix(".prompt.txt")
        system_file  = prefix.with_suffix(".system.txt")
        output_file  = prefix.with_suffix(".output.txt")
        exit_file    = prefix.with_suffix(".exit")

        # Clean any stale sentinel from a previous (failed) attempt
        exit_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)

        # Write inputs
        prompt_file.write_text(prompt, encoding="utf-8")
        if system_prompt:
            system_file.write_text(system_prompt, encoding="utf-8")

        # Build the shell one-liner sent to tmux
        cmd = self._build_command(
            prompt_file=prompt_file,
            system_file=system_file if system_prompt else None,
            output_file=output_file,
            exit_file=exit_file,
        )

        if self.verbose:
            logger.debug("tmux send-keys [session=%s]: %s", self.session_name, cmd)

        # Send command to session
        subprocess.run(
            ["tmux", "send-keys", "-t", self.session_name, cmd, "Enter"],
            check=True,
        )

        # Poll for completion
        output_text = self._wait_for_completion(output_file, exit_file)

        # Cleanup temp files (keep on error for debugging)
        for f in (prompt_file, system_file, output_file, exit_file):
            f.unlink(missing_ok=True)

        return output_text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_command(
        self,
        prompt_file: Path,
        system_file: Optional[Path],
        output_file: Path,
        exit_file: Path,
    ) -> str:
        """Return a shell one-liner to run claude and capture exit code."""
        system_flag = ""
        if system_file and system_file.exists():
            # Use --append-system-prompt so CLAUDE.md in cwd is preserved
            system_flag = f'--append-system-prompt "$(cat {system_file})"'

        # claude -p reads the prompt non-interactively and prints to stdout
        cmd = (
            f'claude -p "$(cat {prompt_file})" {system_flag} '
            f'> {output_file} 2>&1; '
            f'echo $? > {exit_file}'
        )
        return cmd

    def run_task(self, task_file: Path, sentinel_file: Path) -> str:
        """
        Run a full Claude Code agentic session (--dangerously-skip-permissions).

        Unlike ``run_query()``, this gives Claude access to Read/Write/Bash
        tools so it can read Fortran files, write the output Python file, and
        validate syntax itself.  Completion is detected by the ``sentinel_file``
        that Claude writes when it finishes (``DONE`` or ``FAILED: <reason>``).

        Args:
            task_file:     Path to the task markdown file Claude will receive as
                           its initial prompt (``-p "$(cat task_file)"``).
            sentinel_file: Path Claude is instructed to write when done.

        Returns:
            The sentinel file contents (``"DONE"`` on success).

        Raises:
            RuntimeError: If Claude writes ``FAILED: ...`` or times out.
        """
        self.ensure_session()

        exit_file = task_file.with_suffix(".exit")
        log_file  = task_file.with_suffix(".log")

        sentinel_file.unlink(missing_ok=True)
        exit_file.unlink(missing_ok=True)

        cmd = (
            f'claude --dangerously-skip-permissions -p "$(cat {task_file})" '
            f'> {log_file} 2>&1; '
            f'echo $? > {exit_file}'
        )

        if self.verbose:
            logger.debug("tmux run_task [session=%s]: %s", self.session_name, cmd)

        subprocess.run(
            ["tmux", "send-keys", "-t", self.session_name, cmd, "Enter"],
            check=True,
        )

        elapsed = 0.0
        while elapsed < self.timeout:
            if sentinel_file.exists():
                content = sentinel_file.read_text(encoding="utf-8").strip()
                if content.startswith("FAILED"):
                    raise RuntimeError(f"Claude task failed: {content}")
                return content

            if exit_file.exists():
                exit_code_str = exit_file.read_text(encoding="utf-8").strip()
                exit_code = int(exit_code_str) if exit_code_str.isdigit() else 1
                if exit_code != 0:
                    log_text = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
                    raise RuntimeError(
                        f"claude exited with code {exit_code}.\nLog: {log_text[:500]}"
                    )
                # Exit 0 but no sentinel yet — may still be writing
                time.sleep(self.poll_interval)
                elapsed += self.poll_interval
                if sentinel_file.exists():
                    content = sentinel_file.read_text(encoding="utf-8").strip()
                    if content.startswith("FAILED"):
                        raise RuntimeError(f"Claude task failed: {content}")
                    return content
                raise RuntimeError(
                    f"claude exited 0 but sentinel not written: {sentinel_file}"
                )

            time.sleep(self.poll_interval)
            elapsed += self.poll_interval

        raise RuntimeError(
            f"Timed out waiting for task sentinel after {self.timeout}s "
            f"(session: {self.session_name}, sentinel: {sentinel_file})."
        )

    def _wait_for_completion(self, output_file: Path, exit_file: Path) -> str:
        """Poll exit_file until claude finishes, then return output."""
        elapsed = 0.0
        while elapsed < self.timeout:
            if exit_file.exists():
                exit_code_str = exit_file.read_text(encoding="utf-8").strip()
                exit_code = int(exit_code_str) if exit_code_str.isdigit() else 1
                output_text = output_file.read_text(encoding="utf-8") if output_file.exists() else ""

                if exit_code != 0:
                    raise RuntimeError(
                        f"claude exited with code {exit_code}.\n"
                        f"Output: {output_text[:500]}"
                    )
                return output_text

            time.sleep(self.poll_interval)
            elapsed += self.poll_interval

        raise RuntimeError(
            f"Timed out waiting for claude after {self.timeout}s "
            f"(session: {self.session_name})."
        )
