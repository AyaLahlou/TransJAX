# TransJAX

**Translate Fortran scientific code to JAX automatically using LLM agents.**

TransJAX is a Python package that combines static Fortran analysis with
a multi-agent translation pipeline powered by Claude (Anthropic).  It is designed
for the scientific-computing community who want to modernise legacy Fortran
numerics into differentiable, GPU-ready JAX code.

---

## Features

- **Static analysis** — parse any Fortran codebase, extract modules/subroutines/types,
  build dependency graphs, decompose into translation units.
- **LLM-powered translation** — translates Fortran to JAX with strict differentiability
  rules (no Python control-flow inside jitted functions).
- **Automated tests** — generates `pytest` files for every translated module.
- **Repair loop** — iteratively fixes test failures using root-cause analysis.
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

### 1. Set your Anthropic API key or Claude OAuth token

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or copy the template and fill it in
transjax init && cp .env.template .env
```
Pro and Max subscribers can generate an OAuth token by running ```claude setup-token``` locally, which produces a ```CLAUDE_CODE_OAUTH_TOKEN```. This token can be used in place of ```ANTHROPIC_API_KEY``` specifically in Claude Code and the ```claude-code-action``` GitHub Action.
### 2. Inspect the codebase first 

```bash
transjax analyze /path/to/fortran_project --output /path/to/output_dir

```

### 3. Translate

```bash
transjax convert /path/to/fortran_code -o ./jax_output
```

---

## CLI reference

All functionality lives under the single `transjax` command:

```
Usage: transjax [OPTIONS] COMMAND [ARGS]...

  TransJAX — translate Fortran scientific code to JAX.

  Common workflow:
    transjax analyze /path/to/fortran_project --output /path/to/output_dir        
    transjax convert /path/to/fortran -o ./out  # translate + test + repair

Commands:
  analyze      Analyse a Fortran codebase without translating it.
  convert      Translate a Fortran codebase to JAX (full pipeline).
  init         Create a .env.template file in the current directory.
  show-config  Print the active configuration (YAML).
```

#### `transjax convert`

| Option | Default | Description |
|--------|---------|-------------|
| `--output / -o` | `./jax_output` | Output directory |
| `--model` | from config | Claude model name |
| `--api-key` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--max-repair-iterations` | 5 | Repair loop limit per module |
| `--skip-tests` | false | Skip test generation |
| `--skip-repair` | false | Skip repair loop |
| `--force` | false | Re-translate existing files |
| `--modules` | all | Comma-separated module filter |
| `--temperature` | 0.0 | LLM temperature |
| `--verbose / -v` | false | Verbose logging |

#### `transjax analyze`

| Option | Default | Description |
|--------|---------|-------------|
| `--output / -o` | `<src>/transjax_analysis` | Output directory |
| `--template / -t` | `auto` | Project template (auto, ctsm, scientific_computing, generic, …) |
| `--no-graphs` | false | Skip graph visualisation |
| `--verbose / -v` | false | Verbose logging |

---

## Python API

```python
from transjax import FortranAnalyzer, create_analyzer_for_project

# Analyse a Fortran project
analyzer = create_analyzer_for_project("/path/to/fortran", template="auto")
results = analyzer.analyze()
print(f"Found {len(results['modules'])} modules")

# Full translation pipeline
from transjax import OrchestratorAgent
from pathlib import Path

orch = OrchestratorAgent(
    fortran_dir=Path("/path/to/fortran"),
    output_dir=Path("./jax_output"),
    skip_tests=True,
)
summary = orch.run()
```

---

## Output structure

```
jax_output/
├── src/                        # Translated Python/JAX code
│   └── <source_dir>/
│       └── <module>.py
├── tests/                      # Generated pytest files
│   └── <source_dir>/
│       └── test_<module>.py
├── docs/                       # Translation notes (Markdown)
├── reports/                    # Logs, summaries, repair logs
│   └── translation_summary.json
└── static_analysis/            # Fortran analyser output
    ├── analysis_results.json
    └── translation_units.json
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

BSD-3-Clause.  See [LICENSE](LICENSE).

## Links

- **Repository**: https://github.com/AyaLahlou/TransJAX
- **Issue tracker**: https://github.com/AyaLahlou/TransJAX/issues
