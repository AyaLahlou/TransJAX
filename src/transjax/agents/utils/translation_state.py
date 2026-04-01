"""
Translation state manager — persistent, file-backed progress tracker.

Records which Fortran modules have been translated to JAX, in what order they
should be translated, and which module to suggest next.  State is stored as a
JSON file inside the convert output directory so it survives between runs.

State file: <output_dir>/translation_state.json

Schema
------
{
  "version": "1",
  "fortran_dir":  "/abs/path/to/fortran",
  "output_dir":   "/abs/path/to/jax_output",
  "analysis_dir": "/abs/path/to/analysis",
  "created_at":   "<ISO 8601>",
  "last_updated": "<ISO 8601>",
  "ordered_modules": [
    {
      "module":          "math_utils",
      "file":            "/abs/path/src/math_utils.f90",
      "rank":            1,
      "depth":           0,
      "n_internal_deps": 0,
      "status":          "success",   // pending | success | failed
      "translated_at":   "<ISO 8601 or null>",
      "tests_passed":    true,
      "repair_attempts": 0,
      "error_message":   null
    },
    ...
  ]
}

Status values
-------------
  pending   — not yet attempted
  success   — translated (and tests passed if tests were enabled)
  failed    — translation or test run failed
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_STATE_FILE = "translation_state.json"
_STATE_VERSION = "1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TranslationStateManager:
    """
    Manages persistent translation progress for a single output directory.

    Typical lifecycle
    -----------------
    1.  ``mgr = TranslationStateManager(output_dir)``
    2.  ``mgr.load()``                             # load or start fresh
    3.  ``mgr.initialize_from_order(file_order)``  # no-op if already populated
    4.  ``mgr.sync_from_output(output_dir)``       # detect pre-existing files
    5.  For each module: ``mgr.mark_module(name, status, ...)``
    6.  ``mgr.save()``
    """

    def __init__(self, output_dir: Path, fortran_dir: Path, analysis_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.fortran_dir = Path(fortran_dir)
        self.analysis_dir = Path(analysis_dir)
        self.state_path = self.output_dir / _STATE_FILE

        self._state: Dict[str, Any] = {}
        # name → index in ordered_modules list (for O(1) lookups)
        self._index: Dict[str, int] = {}

    # ---------------------------------------------------------------------- #
    # Load / save                                                              #
    # ---------------------------------------------------------------------- #

    _REQUIRED_KEYS = {"version", "ordered_modules", "created_at"}

    def load(self) -> None:
        """Load existing state from disk, or initialise an empty state."""
        if self.state_path.exists():
            try:
                loaded = json.loads(self.state_path.read_text())
                missing = self._REQUIRED_KEYS - set(loaded.keys())
                if missing:
                    logger.warning(
                        "State file %s is missing required keys %s — starting fresh.",
                        self.state_path, missing,
                    )
                else:
                    self._state = loaded
                    self._rebuild_index()
                    logger.info("Loaded translation state from %s", self.state_path)
                    return
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning(
                    "Could not parse %s (%s) — starting fresh.", self.state_path, exc
                )

        self._state = {
            "version": _STATE_VERSION,
            "fortran_dir": str(self.fortran_dir),
            "output_dir": str(self.output_dir),
            "analysis_dir": str(self.analysis_dir),
            "created_at": _now_iso(),
            "last_updated": _now_iso(),
            "ordered_modules": [],
        }
        self._index = {}

    def save(self) -> None:
        """Persist current state to disk."""
        self._state["last_updated"] = _now_iso()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2))
        logger.debug("Translation state saved to %s", self.state_path)

    # ---------------------------------------------------------------------- #
    # Initialisation from analysis                                             #
    # ---------------------------------------------------------------------- #

    def initialize_from_order(self, file_order: List[Dict[str, Any]]) -> int:
        """
        Populate ``ordered_modules`` from the file-level translation order.

        Each file entry in *file_order* may contain multiple modules; they are
        all assigned the same depth/rank as their containing file.

        This is a no-op if ``ordered_modules`` is already non-empty (i.e. the
        state was loaded from a previous run).

        Args:
            file_order: List of file entries from
                        ``FortranAnalyzer.get_file_translation_order()``.

        Returns:
            Number of modules added (0 if state was already populated).
        """
        if self._state.get("ordered_modules"):
            return 0  # already initialised — don't overwrite progress

        entries: List[Dict[str, Any]] = []
        for file_entry in file_order:
            for mod_name in file_entry.get("modules", []):
                entries.append(
                    {
                        "module": mod_name,
                        "file": file_entry.get("file", ""),
                        "rank": file_entry.get("rank", 0),
                        "depth": file_entry.get("depth", 0),
                        "n_internal_deps": file_entry.get("n_internal_deps", 0),
                        "status": "pending",
                        "translated_at": None,
                        "tests_passed": False,
                        "repair_attempts": 0,
                        "error_message": None,
                    }
                )

        self._state["ordered_modules"] = entries
        self._rebuild_index()
        logger.info(
            "Initialised translation state with %d modules from file order.", len(entries)
        )
        return len(entries)

    def initialize_from_module_list(
        self,
        module_names: List[str],
        dependency_levels: Optional[Dict[str, int]] = None,
    ) -> int:
        """
        Fallback initialisation when no file-level order is available.

        Modules are ordered by their ``dependency_levels`` value (lower = fewer
        deps = translate first).  If no levels are provided they are sorted
        alphabetically.

        Returns:
            Number of modules added (0 if state was already populated).
        """
        if self._state.get("ordered_modules"):
            return 0

        levels = dependency_levels or {}
        sorted_names = sorted(module_names, key=lambda m: (levels.get(m, 0), m))
        entries = [
            {
                "module": m,
                "file": "",
                "rank": i + 1,
                "depth": levels.get(m, 0),
                "n_internal_deps": levels.get(m, 0),
                "status": "pending",
                "translated_at": None,
                "tests_passed": False,
                "repair_attempts": 0,
                "error_message": None,
            }
            for i, m in enumerate(sorted_names)
        ]
        self._state["ordered_modules"] = entries
        self._rebuild_index()
        logger.info(
            "Initialised translation state with %d modules from dependency levels.",
            len(entries),
        )
        return len(entries)

    def sync_from_output(self, src_dir: Path) -> int:
        """
        Detect already-translated modules by scanning *src_dir* for .py files.

        Any module whose output Python file already exists is marked ``success``
        if it was still ``pending``.  Useful when re-running after a partial
        session where state was lost.

        Returns:
            Number of modules newly marked as success.
        """
        count = 0
        for entry in self._state.get("ordered_modules", []):
            if entry["status"] != "pending":
                continue
            mod = entry["module"]
            # Standard output pattern written by TranslatorAgent
            candidate = src_dir / f"{mod}.py"
            if candidate.exists():
                entry["status"] = "success"
                entry["translated_at"] = entry.get("translated_at") or _now_iso()
                count += 1
                logger.debug("Inferred '%s' already translated from %s", mod, candidate)
        if count:
            logger.info("Synced %d already-translated module(s) from %s", count, src_dir)
        return count

    # ---------------------------------------------------------------------- #
    # State mutations                                                          #
    # ---------------------------------------------------------------------- #

    def mark_module(
        self,
        module_name: str,
        status: str,
        *,
        tests_passed: bool = False,
        repair_attempts: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update the status of *module_name*.

        Args:
            module_name:    Module to update.
            status:         ``'success'`` or ``'failed'``.
            tests_passed:   Whether all pytest tests passed.
            repair_attempts: How many repair iterations were run.
            error_message:  Error description if status is ``'failed'``.
        """
        idx = self._index.get(module_name)
        if idx is None:
            # Module appeared mid-run (e.g. manual override); append it.
            entry: Dict[str, Any] = {
                "module": module_name,
                "file": "",
                "rank": len(self._state["ordered_modules"]) + 1,
                "depth": 0,
                "n_internal_deps": 0,
                "status": status,
                "translated_at": _now_iso(),
                "tests_passed": tests_passed,
                "repair_attempts": repair_attempts,
                "error_message": error_message,
            }
            self._state["ordered_modules"].append(entry)
            self._index[module_name] = len(self._state["ordered_modules"]) - 1
            return

        entry = self._state["ordered_modules"][idx]
        entry["status"] = status
        entry["translated_at"] = _now_iso()
        entry["tests_passed"] = tests_passed
        entry["repair_attempts"] = repair_attempts
        entry["error_message"] = error_message

    def reset_module(self, module_name: str) -> None:
        """Reset a module back to ``pending`` (use before force-retranslation)."""
        idx = self._index.get(module_name)
        if idx is not None:
            entry = self._state["ordered_modules"][idx]
            entry.update(
                status="pending",
                translated_at=None,
                tests_passed=False,
                repair_attempts=0,
                error_message=None,
            )

    # ---------------------------------------------------------------------- #
    # Queries                                                                  #
    # ---------------------------------------------------------------------- #

    def get_next_suggested(self) -> Optional[str]:
        """
        Return the name of the highest-priority pending module.

        Priority: lowest depth → lowest rank → alphabetical.

        Returns ``None`` if all modules are translated.
        """
        for entry in self._state.get("ordered_modules", []):
            if entry["status"] == "pending":
                return entry["module"]
        return None

    def get_pending_modules(self) -> List[str]:
        """Return all pending module names in translation order."""
        return [
            e["module"]
            for e in self._state.get("ordered_modules", [])
            if e["status"] == "pending"
        ]

    def get_ordered_modules(self) -> List[Dict[str, Any]]:
        """Return all module entries in translation order."""
        return list(self._state.get("ordered_modules", []))

    def get_module_entry(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Return the state entry for *module_name*, or None."""
        idx = self._index.get(module_name)
        return self._state["ordered_modules"][idx] if idx is not None else None

    def is_translated(self, module_name: str) -> bool:
        """Return True if the module has status ``success``."""
        entry = self.get_module_entry(module_name)
        return entry is not None and entry["status"] == "success"

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dict with counts by status."""
        modules = self._state.get("ordered_modules", [])
        by_status: Dict[str, int] = {"pending": 0, "success": 0, "failed": 0}
        for e in modules:
            by_status[e.get("status", "pending")] = (
                by_status.get(e.get("status", "pending"), 0) + 1
            )
        next_mod = self.get_next_suggested()
        next_entry = self.get_module_entry(next_mod) if next_mod else None

        return {
            "total": len(modules),
            "translated": by_status["success"],
            "failed": by_status["failed"],
            "pending": by_status["pending"],
            "percent_done": (
                round(100 * by_status["success"] / len(modules), 1) if modules else 0.0
            ),
            "next_suggested": next_mod,
            "next_depth": next_entry["depth"] if next_entry else None,
            "next_file": next_entry.get("file", "") if next_entry else None,
        }

    def exists(self) -> bool:
        """Return True if a state file already exists on disk."""
        return self.state_path.exists()

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _rebuild_index(self) -> None:
        self._index = {
            e["module"]: i
            for i, e in enumerate(self._state.get("ordered_modules", []))
        }
