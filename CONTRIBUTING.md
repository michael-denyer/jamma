# Contributing to JAMMA

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker (optional, for running GEMMA validation tests)

## Development Setup

```bash
git clone https://github.com/michael-denyer/jamma.git
cd jamma
uv sync
uv run pre-commit install
```

This installs all runtime and dev dependencies and sets up pre-commit hooks (ruff lint + format).

## Running Tests

```bash
# Full suite (parallel by default via pytest-xdist)
uv run pytest tests/ -x

# Filter by keyword
uv run pytest tests/ -x -k lmm
uv run pytest tests/ -x -k hypothesis

# Skip slow tests
uv run pytest tests/ -x -m "not slow"
```

### Test Tiers

| Marker | Description | Typical Runtime |
|--------|-------------|-----------------|
| `tier0` | Fast unit tests, no external dependencies | <5s |
| `tier1` | Parity tests against GEMMA reference data | <60s |
| `tier2` / `slow` | Scale tests, requires large memory or long runtime | Minutes+ |

Tests run with `pytest-randomly` for order randomization. Use `--randomly-seed=<n>` to reproduce a specific ordering.

### GEMMA Validation

Some tests compare JAMMA output against GEMMA reference data in `tests/fixtures/`. To regenerate reference data or run GEMMA directly:

```bash
docker run -v $(pwd):/data gemma -h
```

## Code Style

**Formatter/Linter**: [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`)

Pre-commit hooks run ruff automatically on commit. To run manually:

```bash
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run pre-commit run --all-files  # All hooks
```

### Conventions

- **Docstrings**: Google style (see existing code for examples)
- **Type hints**: Required on all public functions
- **Line length**: 88 characters (ruff default)
- **Imports**: Sorted by ruff (isort rules)

## Branching and Pull Requests

1. Branch from `master` with a descriptive name:
   - `feat/description` for new features
   - `fix/description` for bug fixes
   - `perf/description` for performance improvements
   - `refactor/description` for refactoring

2. Write clear commit messages following conventional commits:
   ```
   feat: add multi-pass LOCO kinship batching
   fix: JAX device array leak on write exception
   perf: skip impute_and_center when no target chromosomes in chunk
   refactor: extract _yield_loco_matrices helper
   ```

3. Ensure all tests pass before opening a PR:
   ```bash
   uv run pytest tests/ -x
   uv run pre-commit run --all-files
   ```

4. PR descriptions should include:
   - **Summary**: What changed and why
   - **Test plan**: How to verify the changes

## Architecture

See [docs/CODEMAP.md](docs/CODEMAP.md) for the project architecture, module responsibilities, and data flow diagrams.

Key source layout:

```
src/jamma/
├── cli.py           # Click CLI entry point
├── core/            # JAX config, memory estimation, utilities
├── io/              # PLINK file readers, result writers
├── kinship/         # Kinship matrix computation (standard, streaming, LOCO)
├── lmm/             # LMM association (likelihood, optimization, runners)
├── utils/           # Shared utilities
└── validation/      # GEMMA comparison, tolerance config
```

## Reporting Issues

Open a [GitHub issue](https://github.com/michael-denyer/jamma/issues) with:

- JAMMA version (`jamma --version` or `python -c "import jamma; print(jamma.__version__)"`)
- Python version and OS
- Minimal reproduction steps
- Full error traceback (if applicable)

For performance issues, include dataset dimensions (samples x SNPs) and available memory.

## License

JAMMA is licensed under [GPL-3.0](LICENSE.md). By contributing, you agree that your contributions will be licensed under the same terms.
