# Contributing to JAMMA

## Prerequisites

- Python 3.11 -- 3.13. CI tests 3.11 and 3.12, and `build-wheels.yml` ships CPython 3.11, 3.12, and 3.13
- [uv](https://docs.astral.sh/uv/) (package manager)
- [prek](https://prek.j178.dev) (pre-commit hooks; v0.3.8+)
- A C compiler with OpenMP support (gcc, clang)
- A local [GEMMA 0.98.5](https://github.com/genetics-statistics/GEMMA) binary (optional, only needed for GEMMA-parity tests that compare against fresh GEMMA output)

## Development Setup

```bash
git clone https://github.com/michael-denyer/jamma.git
cd jamma
uv sync
prek install
```

This installs all runtime and dev dependencies and sets up the
[prek](https://prek.j178.dev/)-managed git hooks. Hooks include ruff (lint + format),
pyrefly (static types), clang-format, cppcheck, markdownlint, mermaid syntax
(`maid`), lychee link check, typos (spell check), actionlint (workflow lint),
zizmor (workflow security), shellcheck, vulture (dead-code),
refurb (refactor suggestions), plus JAMMA-specific
gates -- fixture-manifest verification, forbidden-patches AST check,
compile-flag-literal lint, route-through-`_build_support` enforcement,
`uv.lock` sync, and two pre-push checks (repo-wide `ruff format --check` and
C-extension freshness).

The pyrefly gate is absolute. The project sits at zero errors and there is no
baseline file, so a new error has to be fixed or given a narrow inline
`# type: ignore[code]` on the offending line. Run it with `uv run pyrefly check`.

## Running Tests

```bash
# Default suite -- excludes slow + tier2 + benchmark per pyproject addopts.
# Implicit pytest-xdist (-n 3) and pytest-timeout (--timeout=120).
uv run pytest tests/ -x

# Filter by keyword
uv run pytest tests/ -x -k lmm
uv run pytest tests/ -x -k hypothesis

# Run slow + tier2 too (override addopts)
uv run pytest tests/ -x -o 'addopts='
```

### Test Tiers

Every test file must declare a tier marker (or `slow`/`benchmark`) at module
or function level. A `pytest_configure` gate aborts the run if a file is
missing one. The `tier3` marker was removed in v5.3.0 -- if you are porting
an older plan, retag those tests to `tier2`.

| Marker | Description | Typical Runtime | Default suite |
|--------|-------------|-----------------|---------------|
| `tier0` | Fast unit tests, no external dependencies | <5s | run |
| `tier1` | Parity tests against GEMMA reference data | <60s | run |
| `tier2` / `slow` | Scale tests, large memory or long runtime | Minutes+ | excluded |
| `benchmark` | pytest-benchmark microbenchmarks | varies | excluded |

Tests run with `pytest-randomly` for order randomization. Use
`--randomly-seed=<n>` to reproduce a specific ordering.
`pytest-rerunfailures` is installed for marking known-flaky tests with
`@pytest.mark.flaky(reruns=N)`; the weekly `flaky-detect.yml` workflow
runs the suite under five distinct seeds and opens an issue on disagreement.

For full test guidance (mocking boundaries, fakes vs mocks, fixture
manifest, forbidden-patches gate), see [`docs/TESTING.md`](docs/TESTING.md).

### GEMMA Validation

Some tests compare JAMMA output against GEMMA reference data in
`tests/fixtures/`. Build [GEMMA 0.98.5](https://github.com/genetics-statistics/GEMMA)
locally (or use a prebuilt binary) and run it directly against the same
PLINK fixtures -- no Docker workflow is maintained in-tree. After
regenerating any fixture, refresh the manifest:

```bash
uv run python scripts/regenerate_fixture_manifest.py
```

`tests/fixtures/MANIFEST.toml` SHA-256-tracks all 55 fixtures; a stale
manifest fails the pre-commit gate.

## Code Style

**Formatter/Linter**: [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`)

Pre-commit hooks run ruff automatically on commit. To run manually:

```bash
uv run ruff check .        # Lint
uv run ruff format .       # Format
prek run --all-files       # All hooks
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

   ```text
   feat: add multi-pass LOCO kinship batching
   fix: streaming runner chunk boundary handling
   perf: skip impute_and_center when no target chromosomes in chunk
   refactor: extract _yield_loco_matrices helper
   ```

3. Ensure all tests pass before opening a PR:

   ```bash
   uv run pytest tests/ -x
   prek run --all-files
   ```

4. PR descriptions should include:
   - **Summary**: What changed and why
   - **Test plan**: How to verify the changes

## Architecture

See [docs/CODEMAP.md](docs/CODEMAP.md) for the project architecture, module responsibilities, and data flow diagrams.

Key source layout:

```text
src/jamma/
├── cli.py           # Click CLI entry point
├── core/            # Memory estimation, backend selection, utilities
├── io/              # PLINK file readers, result writers
├── kinship/         # Kinship matrix computation (standard, streaming, LOCO)
├── lmm/             # LMM association (likelihood, optimization, runners,
│                    # and the _lmm_accel C extension, built from the
│                    # _lmm_*.c files listed in LMM_ACCEL_SOURCES)
├── jlinalg/         # Vendor BLAS/LAPACK dispatch C layer + NumPy fallback
├── _build_support/  # Single source of truth for compile flags, sources,
│                    # link flags. Imported by all three compile entry points
│                    # (hatch_build.py, _compile_jlinalg.py, _compile_accel.py).
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
