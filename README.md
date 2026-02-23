# End-to-end package for translating Fortran code to Python JAX.

## Installation

### Step 1: Install Package in Development Mode

```bash
pip install -r requirements.txt
cd jax-agents
pip install -e .
```

This makes the `fortran-to-jax` command available globally.

### Step 2: Initialize a Project (Optional)

For new users who want configuration templates:

```bash
fortran-to-jax init
```

This creates:
- `config.yaml` (LLM configuration)
- `.env.template` (API key template)
- `requirements.txt` (dependencies)

### Step 3: Configure API Key

Create `.env` file:

```bash
echo "ANTHROPIC_API_KEY=your_actual_key_here" > .env
```

Or set environment variable:

```bash
export ANTHROPIC_API_KEY="your_actual_key_here"
```

## Usage Examples

### Basic Usage

```bash
# Translate all Fortran files in a directory
fortran-to-jax convert /path/to/fortran_code -o ./output
```

### Selective Module Translation

```bash
# Translate specific modules only
fortran-to-jax convert /path/to/fortran --modules clm_varctl,SoilStateType
```

### Advanced Options

```bash
# Use Claude Opus 4.5 with custom settings
fortran-to-jax convert ./fortran \
  --output ./jax_output \
  --model claude-opus-4-5 \
  --max-repair-iterations 10 \
  --verbose

# Skip tests (just translate)
fortran-to-jax convert ./fortran --skip-tests

# Force re-translation (ignore existing files)
fortran-to-jax convert ./fortran --force
```

### Check Current Configuration

```bash
fortran-to-jax show-config
```

## Output Directory Structure

After running, the output directory will contain:

```
jax_output/
├── src/                          # Translated Python/JAX code
│   ├── clm_src_main/
│   │   ├── clm_varctl.py
│   │   └── clm_varctl_params.py
│   └── clm_src_biogeophys/
│       └── SoilTemperatureMod.py
├── tests/                        # Generated pytest files
│   ├── clm_src_main/
│   │   └── test_clm_varctl.py
│   └── clm_src_biogeophys/
│       └── test_SoilTemperatureMod.py
├── docs/                         # Documentation
│   └── translation_notes/
│       ├── clm_varctl_notes.md
│       └── SoilTemperatureMod_notes.md
├── reports/                      # Logs and reports
│   ├── translation_summary.json
│   └── repair_logs/
│       └── SoilTemperatureMod/
│           ├── root_cause_analysis.md
│           └── final_test_report.txt
└── static_analysis/              # Fortran analyzer output
    ├── analysis_results.json
    └── translation_units.json
```

## Pipeline Workflow

When you run `fortran-to-jax convert`, the orchestrator executes:

```
1. Static Analysis
   ├─ Run Fortran analyzer on input directory
   ├─ Generate analysis_results.json
   └─ Generate translation_units.json

2. Module Ordering
   ├─ Parse dependencies from analysis
   ├─ Sort modules (leaves first, roots last)
   └─ Filter: skip already-translated (unless --force)

3. For Each Module:
   ├─ Translate Fortran → JAX (unit-by-unit iterative)
   ├─ Save to src/<source_dir>/<module>.py
   ├─ Generate tests → tests/<source_dir>/test_<module>.py
   ├─ Run pytest
   └─ If failures:
       ├─ Analyze root cause
       ├─ Generate corrected code
       ├─ Re-run tests
       └─ Repeat up to --max-repair-iterations times

4. Summary Report
   ├─ Display results table
   ├─ Save translation_summary.json
   └─ Exit code: 0 (success) or 1 (failures remain)
```

## Key Features Implemented

✅ **Unified CLI** - Single command for end-to-end translation
✅ **Automatic analyzer invocation** - No manual static analysis needed
✅ **Dependency-aware ordering** - Translates modules in correct order
✅ **Incremental translation** - Skips already-translated modules (use `--force` to override)
✅ **Automatic repair loop** - Iteratively fixes failed tests
✅ **Structured output** - Organized into src/, tests/, docs/, reports/
✅ **Progress tracking** - Rich console output with progress bars
✅ **Comprehensive reporting** - Detailed logs and root cause analyses
✅ **Flexible configuration** - CLI flags override config file

## Testing the Integration

### Quick Test

```bash
# 1. Install in dev mode
cd jax-agents
pip install -e .

# 2. Verify CLI is available
fortran-to-jax --version

# 3. Initialize config
mkdir test_project && cd test_project
fortran-to-jax init

# 4. Set API key
echo "ANTHROPIC_API_KEY=your_key" > .env

# 5. Run on a small Fortran module
fortran-to-jax convert /path/to/small_fortran_module -o ./output
```

### Verify Output

Check that output directory contains:
- `src/<source_dir>/<module>.py` - Translated code
- `tests/<source_dir>/test_<module>.py` - Generated tests
- `reports/translation_summary.json` - Summary report

### Run Tests Manually

```bash
cd output
pytest tests/ -v
```

## Troubleshooting

### Issue: `fortran-to-jax: command not found`

**Solution:** Reinstall in development mode:
```bash
pip uninstall jax-agents
cd jax-agents
pip install -e .
```

### Issue: `ANTHROPIC_API_KEY not found`

**Solution:** Set environment variable or create `.env` file:
```bash
export ANTHROPIC_API_KEY="your_key"
# or
echo "ANTHROPIC_API_KEY=your_key" > .env
```

### Issue: Static analysis fails

**Solution:** Ensure Fortran analyzer is importable:
```bash
python -c "from fortran_analyzer.analyzer import FortranAnalyzer"
```

If import fails, check that `fortran_analyzer/` is in the same parent directory as `jax-agents/`.

### Issue: Tests fail to run

**Solution:** Install test dependencies:
```bash
pip install pytest jax jaxlib numpy
```

### Issue: Module translation fails

**Solution:** Check verbose output:
```bash
fortran-to-jax convert /path/to/fortran --verbose
```

Review error logs in `output/reports/`.

## Migration from Old Workflow

### Before (Manual Multi-Step)

```bash
# Step 1: Analyze
python fortran_analyzer/cli.py /path/to/fortran

# Step 2: Translate
cd jax-agents
./run_translation_workflow.sh --translate --modules clm_varctl

# Step 3: Test
./run_translation_workflow.sh --test

# Step 4: Repair
python examples/repair_agent_example.py

# Step 5: Verify
pytest translated_modules/clm_varctl/tests/
```

### After (Single Command)

```bash
fortran-to-jax convert /path/to/fortran -o ./output
```

All steps are automated!