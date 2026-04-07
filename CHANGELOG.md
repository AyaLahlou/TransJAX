# TransJAX Changelog

## Unreleased

### Added
- **Tmux Claude runner** (`agents/utils/tmux_runner.py`): Route all `claude` API calls
  through the `claude` CLI running inside a named tmux session. One session per module
  (`transjax-{module}`). Enables long-running jobs on HPC clusters that survive SSH
  disconnects. Enable with `--use-tmux` or `llm.use_tmux: true` in config.
- **RALPH repair loop** (`repair_agent.py`): Explicit Run→Assess→Loop→Patch→Halt stages
  with structured `{module}_ralph_log.json` output per repair attempt.
- **Slurm integration** (`agents/utils/slurm_runner.py`, `transjax slurm-submit`):
  Generate and submit one Slurm job per module. Each job creates a tmux session on the
  compute node and runs the full pipeline for that module.
- **Git coordination** (`agents/utils/git_coordinator.py`): Auto-commit translated output
  files to git after each successful module. Enable with `--auto-git-commit`.
  Entries are prepended to `CHANGELOG.md` automatically.
- **Doc generation** (`agents/utils/doc_generator.py`, `transjax generate-docs`):
  Semi-automatically generate `DESIGN.md` and `CLAUDE.md` from analyzer output
  (`analysis_results.json`, `translation_order.json`, `translation_state.json`).
  Follows the clax-style project documentation format.

### Changed
- `BaseAgent`: Added `use_tmux`, `tmux_poll_interval`, `tmux_timeout` parameters.
  Added `set_tmux_session()` method. Backward-compatible: `use_tmux=False` (default)
  keeps the existing Anthropic SDK path.
- `PipelineRunner`: Added `use_tmux` and `auto_git_commit` parameters. Each agent
  instantiation now calls `set_tmux_session()` when tmux mode is enabled.
- `transjax pipeline`: New flags `--use-tmux`, `--auto-git-commit`.

### New CLI commands
- `transjax slurm-submit` — generate and/or submit Slurm jobs per module
- `transjax generate-docs` — generate DESIGN.md and CLAUDE.md from analyzer output

## Previous

See git log for earlier history.
