# TransJAX

**Translate Fortran scientific code to differentiable JAX automatically using a multi-agent LLM pipeline.**

TransJAX combines static Fortran analysis with a neurosymbolic multi-agent system powered by Claude (Anthropic). It is designed for the scientific-computing community who want to modernise legacy Fortran numerics into differentiable, GPU-ready JAX code — with numerical correctness verified against the original Fortran at every step.

---

## Features

- **Static analysis** — parse any Fortran codebase, extract modules/subroutines/types, build dependency graphs, decompose into translation units.
- **LLM-powered translation** — translates Fortran → JAX with strict differentiability rules (pure functions, `jnp` arrays, `lax` primitives, no in-place mutation).
- **Functional test framework (FTest)** — auto-generates thin Fortran driver programs that expose each subroutine as an isolated, testable unit.
- **Golden I/O capture** — runs compiled Fortran drivers to record trusted input/output pairs as JSON golden files.
- **Numerical parity testing** — verifies that JAX outputs match Fortran golden data within configurable tolerances (`rtol`, `atol`).
- **Iterative repair** — three repair agents (translation, parity, integration) drive Claude in a fix → retest loop until tests pass or the iteration limit is reached.
- **System integration** — builds and tests a full `model_run.py` that mirrors the Fortran model's call sequence using the translated JAX modules.
- **Resumable pipeline** — persistent JSON state means the pipeline can be interrupted and resumed without re-doing completed work.

---

## Architecture

```
transjax analyze   →  dependency graph + translation order
        │
        ▼
transjax ftest     →  Fortran test drivers (compile + isolate each subroutine)
        │
        ▼
transjax golden    →  trusted I/O golden data (JSON) from compiled drivers
        │
        ▼
transjax convert   →  Fortran → JAX (TranslatorAgent + RepairAgent)
        │
        ▼
transjax test-parity → numerical parity: JAX ↔ Fortran golden (ParityAgent)
        │
        ▼
transjax parity-repair → iterative repair of parity failures (ParityRepairAgent)
        │
        ▼
transjax integrate →  system integration: model_run.py + tests (IntegratorAgent
                       + IntegrationRepairAgent)

  OR run everything in one shot:
transjax pipeline  →  steps ftest → golden → translate → parity for every module
                       (optionally followed by integrate with --integrate)
```

### Agent roles

| Agent | Responsibility |
|---|---|
| `TranslatorAgent` | Fortran → JAX module translation |
| `TestAgent` | pytest generation for translated modules |
| `RepairAgent` | iterative fix of failed translation tests |
| `FtestAgent` | Fortran functional test framework builder |
| `GoldenAgent` | trusted golden I/O capture from Fortran drivers |
| `ParityAgent` | numerical parity test generation and execution |
| `ParityRepairAgent` | iterative repair of parity failures |
| `IntegratorAgent` | system integration builder and tester |
| `IntegrationRepairAgent` | iterative repair of system integration failures |
| `OrchestratorAgent` | high-level translate → test → repair pipeline |
| `PipelineRunner` | full end-to-end pipeline (ftest → golden → translate → parity) |

---

## Installation

```bash
pip install transjax
```

For development / running the test suite:

```bash
git clone https://github.com/AyaLahlou/TransJAX.git
cd TransJAX
pip install -e ".[dev]"
```

---

## Quick start

### 1. Authenticate

**Option A — Claude Pro/Max subscription (recommended, zero per-token billing):**

```bash
claude login          # one-time setup; sets CLAUDE_CODE_OAUTH_TOKEN automatically
```

**Option B — Pay-per-use API key:**

```bash
transjax init                    # creates .env.template
cp .env.template .env            # open .env and set ANTHROPIC_API_KEY
```

### 2. Inspect the codebase

```bash
transjax analyze /path/to/fortran -o ./analysis
```

Produces `translation_order.json` and `analysis_results.json` — the dependency-ordered module list used by all downstream commands.

### 3. Translate

```bash
# Translate all modules (resumable — re-run to continue after interruption)
transjax convert /path/to/fortran -o ./jax_output --analysis-dir ./analysis

# Translate one module at a time
transjax convert /path/to/fortran --next --analysis-dir ./analysis

# View progress
transjax status ./jax_output
```

### 4. Run the full pipeline (recommended)

```bash
transjax pipeline /path/to/fortran \
    --analysis-dir ./analysis \
    --output ./pipeline_out \
    --gcm-model-name CTSM \
    --ftest-build-dir ./model_build   # omit if Fortran compiler unavailable
```

This runs FTest → Golden I/O → Translate → Parity for every module in dependency order, with automatic repair on parity failures.

### 5. Build the system integration

```bash
# After all modules are translated:
transjax integrate ./pipeline_out/jax/src \
    --fortran-dir /path/to/fortran \
    --output ./pipeline_out \
    --gcm-model-name CTSM

# Or add --integrate to the pipeline command to do it in one shot:
transjax pipeline /path/to/fortran \
    --analysis-dir ./analysis \
    --output ./pipeline_out \
    --integrate
```

---

## CLI reference

### `transjax analyze`

Analyse a Fortran codebase and produce the dependency-ordered translation plan.

```bash
transjax analyze FORTRAN_DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output / -o` | `<cwd>/transjax_analysis` | Output directory |
| `--template / -t` | `auto` | Project template: `auto`, `generic`, `scientific_computing`, `climate_model`, `ctsm` |
| `--no-graphs` | false | Skip GraphML/JSON dependency graph generation |
| `--verbose / -v` | false | Verbose logging |

---

### `transjax convert`

Translate a Fortran codebase to JAX (uses `TranslatorAgent` + `RepairAgent`).

```bash
transjax convert FORTRAN_DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output / -o` | `./jax_output` | Output directory |
| `--analysis-dir` | auto | Reuse an existing `transjax analyze` output directory |
| `--gcm-model` | — | ESM name injected into prompts (e.g. `CTSM`, `MOM6`) |
| `--mode` | `units` | `units` (subroutine-by-subroutine) or `whole` (full module in one call) |
| `--model` | from config | Claude model |
| `--api-key` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--max-repair-iterations` | 5 | Repair loop limit per module |
| `--skip-tests` | false | Skip test generation and repair |
| `--skip-repair` | false | Translate without repair loop |
| `--force` | false | Re-translate modules already marked successful |
| `--modules` | all | Comma-separated module filter |
| `--next` | false | Translate only the next pending module |
| `--yes / -y` | false | Skip confirmation prompt with `--next` |
| `--temperature` | 0.0 | LLM sampling temperature |
| `--verbose / -v` | false | Verbose logging |

---

### `transjax status`

View translation progress dashboard.

```bash
transjax status OUTPUT_DIR
```

---

### `transjax ftest`

Generate Fortran functional test drivers for isolated subroutine testing.

```bash
transjax ftest FORTRAN_DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output / -o` | `./ftest_output` | Output directory |
| `--build-dir` | — | Model build directory (`.o`/`.mod` files for linking) |
| `--compiler` | `nvfortran` | Fortran compiler command |
| `--netcdf-inc` | — | NetCDF include path (`-I`) |
| `--netcdf-lib` | — | NetCDF library path (`-L`) |
| `--module` | all | Module name filter (repeatable) |
| `--model` | from config | Claude model |
| `--verbose / -v` | false | Verbose logging |

---

### `transjax golden`

Run compiled Fortran drivers to capture golden I/O reference data.

```bash
transjax golden FTEST_OUTPUT_DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output / -o` | `<ftest_dir>/tests/golden` | Golden output directory |
| `--n-cases` | 5 | Number of cases to generate per subroutine |
| `--model` | from config | Claude model |
| `--verbose / -v` | false | Verbose logging |

---

### `transjax test-parity`

Generate and run numerical parity tests (JAX vs. Fortran golden data).

```bash
transjax test-parity PYTHON_FILE [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--golden-dir` | required | Directory with golden JSON files |
| `--output / -o` | `./parity_output` | Output directory for test files and report |
| `--ftest-report` | — | Path to `ftest_report.json` (enables programmatic test generation — recommended) |
| `--rtol` | `1e-10` | Relative tolerance for `jnp.allclose` |
| `--atol` | `1e-12` | Absolute tolerance |
| `--model` | from config | Claude model (used only in LLM fallback mode) |
| `--verbose / -v` | false | Verbose logging |

---

### `transjax parity-repair`

Iteratively repair a JAX module until parity tests pass.

```bash
transjax parity-repair PYTHON_FILE [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--fortran-file` | required | Original Fortran source (read-only reference) |
| `--golden-dir` | required | Directory with golden JSON files |
| `--ftest-report` | — | Path to `ftest_report.json` |
| `--output / -o` | `./parity_repair_output` | Output directory |
| `--rtol` | `1e-10` | Relative tolerance |
| `--atol` | `1e-12` | Absolute tolerance |
| `--max-iterations` | 5 | Maximum repair iterations |
| `--model` | from config | Claude model |
| `--verbose / -v` | false | Verbose logging |

---

### `transjax integrate`

Build and test the Python system integration for a fully translated ESM.

```bash
transjax integrate JAX_SRC_DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--fortran-dir` | required | Fortran source root (read-only reference) |
| `--output / -o` | `./pipeline_output` | Root output directory |
| `--gcm-model-name` | `generic ESM` | ESM name injected into prompts and docs |
| `--max-repair-iterations` | 5 | IntegrationRepairAgent iteration limit |
| `--model` | from config | Claude model |
| `--api-key` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--verbose / -v` | false | Verbose logging |

Generates:
- `<output>/integration/model_run.py` — Python integration driver
- `<output>/integration/test_integration.py` — pytest wrapper
- `<output>/integration/docs/System_integration_README.md`
- `<output>/integration/docs/integration_repair_PASS|FAIL.md` (if repair was needed)

---

### `transjax pipeline`

Run the full end-to-end pipeline for every module in translation order.

```bash
transjax pipeline FORTRAN_DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--analysis-dir` | required | Directory from `transjax analyze` |
| `--output / -o` | `./pipeline_output` | Root output directory |
| `--gcm-model-name` | `generic ESM` | ESM name |
| `--mode` | `units` | Translation mode (`units` or `whole`) |
| `--ftest-build-dir` | — | Model build directory for Ftest compilation |
| `--ftest-compiler` | `nvfortran` | Fortran compiler |
| `--ftest-netcdf-inc` | — | NetCDF include path |
| `--ftest-netcdf-lib` | — | NetCDF library path |
| `--golden-cases` | 5 | Golden I/O cases per subroutine |
| `--rtol` | `1e-10` | Parity relative tolerance |
| `--atol` | `1e-12` | Parity absolute tolerance |
| `--max-repair-iterations` | 5 | Parity repair iteration limit per module |
| `--module` | all | Module filter (repeatable: `--module A --module B`) |
| `--skip-ftest` | false | Skip FTest step |
| `--skip-golden` | false | Skip golden I/O generation |
| `--force` | false | Re-run all steps for every module |
| `--force-ftest` | false | Re-run FTest even if output exists |
| `--force-golden` | false | Re-run golden I/O even if output exists |
| `--force-translate` | false | Re-translate even if JAX file exists |
| `--force-parity` | false | Re-run parity even if previously passing |
| `--integrate` | false | Run `IntegratorAgent` after all modules complete |
| `--max-integration-repair-iterations` | 5 | Integration repair iteration limit |
| `--model` | from config | Claude model |
| `--api-key` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--verbose / -v` | false | Verbose logging |

**Per-module steps (in dependency order):**

```
[1/4] FTest     — generate Fortran test drivers & compile
[2/4] Golden    — run drivers to capture golden I/O reference data
[3/4] Translate — translate Fortran module to JAX/Python with Claude
[4/4] Parity    — run numerical parity tests; auto-repair if they fail
```

State is persisted after each module so the pipeline can be safely interrupted and resumed with the same command.

---

## Output layout

### `transjax convert`

```
jax_output/
├── src/                        # Translated JAX modules
│   └── <module>.py
├── tests/                      # Generated pytest files
│   └── test_<module>.py
├── docs/                       # Translation notes
├── reports/                    # Repair logs, root-cause analysis
│   └── root_cause_analysis_<module>.md
└── static_analysis/            # Fortran analyser output
    ├── analysis_results.json
    └── translation_units.json
```

### `transjax pipeline`

```
pipeline_output/
├── translation_state.json       # per-module translation status (resumable)
├── pipeline_state.json          # per-module per-step details
├── ftest/<module>/              # FTest drivers & golden JSON
│   ├── ftest_report.json
│   ├── drivers/
│   └── tests/golden/<module>_<sub>.json
├── jax/src/<module>.py          # Translated JAX modules
├── parity/<module>/             # Parity test files & repair report
│   ├── parity_report.json
│   └── docs/<module>_numerical_parity_repair_PASS|FAIL.md
└── integration/                 # System integration (with --integrate)
    ├── model_run.py
    ├── test_integration.py
    └── docs/
        ├── System_integration_README.md
        └── integration_repair_PASS|FAIL.md
```

---

## Python API

```python
from pathlib import Path
from transjax import OrchestratorAgent

# Simple translate + test + repair
orch = OrchestratorAgent(
    fortran_dir=Path("/path/to/fortran"),
    output_dir=Path("./jax_output"),
    gcm_model_name="CTSM",
)
summary = orch.run()
print(f"Translated: {summary['translated_count']}, passed: {summary['tests_passed']}")
```

```python
from transjax.agents.pipeline_runner import PipelineRunner
from pathlib import Path

runner = PipelineRunner(
    fortran_dir=Path("/path/to/fortran"),
    analysis_dir=Path("./analysis"),
    output_dir=Path("./pipeline_out"),
    gcm_model_name="CTSM",
    run_integrate=True,
)
summary = runner.run()
```

```python
from transjax.agents.integrator_agent import IntegratorAgent
from pathlib import Path

agent = IntegratorAgent()
result = agent.run(
    fortran_dir=Path("/path/to/fortran"),
    jax_src_dir=Path("./pipeline_out/jax/src"),
    output_dir=Path("./pipeline_out"),
    gcm_model_name="CTSM",
)
print("Integration:", result.final_status)
```

```python
from transjax import FortranAnalyzer, create_analyzer_for_project

# Analyse only
analyzer = create_analyzer_for_project("/path/to/fortran", template="auto")
results = analyzer.analyze()
print(f"Found {len(results['modules'])} modules")
```

---

## Configuration

TransJAX reads `~/.transjax/config.yaml` or a local `config.yaml`. All settings can be overridden by CLI flags.

Default LLM settings (`src/transjax/agents/utils/default_config.yaml`):

```yaml
llm:
  model: "claude-sonnet-4-6"
  temperature: 0.0        # deterministic for code generation
  max_tokens: 48000
  timeout: 600
```

---

## Development

```bash
# Install with all dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/

# Build a wheel
pip install build
python -m build --wheel
```

---

## License

BSD-3-Clause. See [LICENSE](LICENSE).

## Links

- **Repository**: https://github.com/AyaLahlou/TransJAX
- **Issue tracker**: https://github.com/AyaLahlou/TransJAX/issues
