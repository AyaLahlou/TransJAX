# TransJAX Development Guide

## Project Overview

TransJAX is a multi-agent LLM system that translates legacy Fortran scientific code to modern, differentiable JAX. It targets Earth System Model codebases (CTSM, CESM, MOM6, etc.) and validates each translation numerically against the original Fortran.

## Setup

```bash
pip install -e ".[dev]"
```

Authentication (one of):
```bash
claude login                     # Claude Pro/Max subscription (recommended)
# OR
export ANTHROPIC_API_KEY=sk-...  # Pay-per-use API key
```

The `BaseAgent` checks `CLAUDE_CODE_OAUTH_TOKEN` first, then `ANTHROPIC_API_KEY`. If neither is set it raises a `ValueError` at construction time.

## Common Commands

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/

# Type check
mypy src/

# CLI entry point
transjax --help
```

## Architecture

The system is organized around two top-level packages:

### `transjax.analyzer` — Static Analysis

Parses Fortran source, builds module dependency graphs, and decomposes code into translation units. Produces `analysis_results.json` and `translation_units.json`.

Key files:
- `analyzer/analyzer.py` — main orchestrator
- `analyzer/parser/fortran_parser.py` — Fortran parsing (fparser2 optional but recommended)
- `analyzer/analysis/call_graph_builder.py` — dependency graph
- `analyzer/analysis/translation_decomposer.py` — splits modules into per-subroutine units

### `transjax.agents` — Multi-Agent Pipeline

11 specialized agents, all inheriting from `BaseAgent`, coordinated by `OrchestratorAgent` or the higher-level `PipelineRunner`.

**Agent hierarchy:**

```
BaseAgent
├── TranslatorAgent       — Fortran → JAX translation
├── TestAgent             — auto-generates pytest suites
├── RepairAgent           — iterative test failure repair
├── FtestAgent            — Fortran functional test drivers
├── GoldenAgent           — captures golden I/O from compiled Fortran
├── ParityAgent           — numerical parity (jnp.allclose vs golden)
├── ParityRepairAgent     — repairs parity failures
├── IntegratorAgent       — builds model_run.py system integration
└── IntegrationRepairAgent — repairs integration test failures

OrchestratorAgent         — translate + test + repair loop
PipelineRunner            — full end-to-end (ftest → golden → translate → parity → integrate)
```

**Pipeline flow:**
1. `FtestAgent` — generate and compile thin Fortran driver programs
2. `GoldenAgent` — run drivers, record trusted I/O as JSON
3. `TranslatorAgent` — LLM translates Fortran → JAX (mode: `units` per-subroutine or `whole` module)
4. `TestAgent` → `RepairAgent` — generate pytest, repair failures iteratively
5. `ParityAgent` → `ParityRepairAgent` — verify JAX outputs match Fortran numerically
6. `IntegratorAgent` → `IntegrationRepairAgent` — build and test `model_run.py`

**Resumability:** `TranslationStateManager` writes `translation_state.json` to the output dir. Every step is checkpointed. Use `--next` flag to process one module at a time safely.

## Key Files

| File | Purpose |
|------|---------|
| `agents/base_agent.py` | Auth, Claude API calls, retry logic, token tracking |
| `agents/translator.py` | Fortran→JAX conversion (regex interface parsing, dtype mapping) |
| `agents/orchestrator.py` | translate + test + repair loop, dependency ordering |
| `agents/pipeline_runner.py` | full pipeline, `pipeline_state.json` |
| `agents/cli.py` | All CLI commands (click) |
| `agents/prompts/translation_prompts.py` | LLM prompts (system prompt, unit/whole module templates) |
| `agents/utils/config_loader.py` | YAML config with 4-level priority |
| `agents/utils/translation_state.py` | Persistent progress tracking |
| `agents/utils/default_config.yaml` | Default LLM + agent settings |
| `analyzer/analyzer.py` | Static analysis entry point |

## Configuration

Config is resolved in this priority order (highest wins):
1. CLI flags
2. `./config.yaml` (local project)
3. `~/.transjax/config.yaml` (user home)
4. `src/transjax/agents/utils/default_config.yaml` (package defaults)

Key defaults (`default_config.yaml`):
- Model: `claude-sonnet-4-6`
- Temperature: `0.0` (deterministic)
- Max tokens: `48000`
- Translation mode: `units` (per-subroutine, not whole-module)
- Parity tolerances: `rtol=1e-10`, `atol=1e-12`

## Code Conventions

- **Python 3.9+** — use `from typing import Dict, List, Optional` not `dict[str, ...]`
- **Black** — line length 100, applied automatically
- **Ruff** — rules E, F, W, I; ignores E501, E722, E701, F401
- **MyPy** — `disallow_untyped_defs = true`; all public methods need type hints
- **pathlib.Path** — always use instead of `os.path`
- **dataclasses** — for structured results (`TranslationResult`, `ModuleStatus`, etc.)
- **Rich** — for all console output (`from rich.console import Console`)
- **tenacity** — for retrying Claude API calls (`@retry(stop=stop_after_attempt(3), ...)`)
- **logging** — module-level `logger = logging.getLogger(__name__)`, not print statements

## Adding a New Agent

1. Subclass `BaseAgent` in `agents/your_agent.py`
2. Add prompts in `agents/prompts/your_agent_prompts.py`
3. Wire into `PipelineRunner` (`pipeline_runner.py`) or `OrchestratorAgent` as appropriate
4. Add CLI command in `agents/cli.py`
5. Export from `agents/__init__.py`

## Testing

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/transjax

# Tests must not call the real API — use ANTHROPIC_API_KEY=test-key-not-used
ANTHROPIC_API_KEY=test-key-not-used pytest tests/ -v
```

Test locations:
- `tests/test_imports.py` — package imports, version consistency
- `tests/agents/test_config_loader.py` — config loading
- `tests/agents/test_orchestrator.py` — orchestrator pipeline

## Output Directory Structure

```
output/
├── translation_state.json       # Per-module status (resumable)
├── pipeline_state.json          # Per-module per-step details
├── src/<module>.py              # Translated JAX modules
├── tests/test_<module>.py       # Generated pytest suites
├── reports/                     # Repair logs
├── ftest/<module>/
│   ├── ftest_report.json
│   ├── drivers/                 # Compiled Fortran drivers
│   └── tests/golden/            # Golden I/O JSON files
├── parity/<module>/
│   └── parity_report.json
└── integration/
    ├── model_run.py             # Python mirror of Fortran call sequence
    ├── test_integration.py
    └── docs/
```

## CI/CD

- **ci.yml** — runs on push/PR to `main`, `master`, `claude/*`, `feature/*`; tests Python 3.9/3.10/3.11; lints with ruff; builds wheel
- **publish.yml** — publishes to PyPI on git tags matching `v*.*.*`

Both workflows install with `pip install -e ".[dev]"` and run `pytest tests/ -v` with a dummy API key.
